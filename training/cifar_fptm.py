"""
CIFAR-10/100 trainer for FPTMConv with strong recipe:
- RandAugment, Mixup/CutMix, Random Erasing, Label Smoothing
- AdamW, cosine LR with warmup, grad clip, AMP, EMA
- Optional knowledge distillation from a torchvision ResNet-18 teacher
"""

import os, math, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from fptm.models import FPTMConv, FPTMConvDeep
from fptm.heads import compute_ece
from fptm.utils import set_seed

# ------------------------- Utils -------------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}

def smooth_one_hot(labels, classes, smoothing=0.1, device=None):
    with torch.no_grad():
        t = torch.full((labels.size(0), classes), fill_value=smoothing / (classes - 1), device=device)
        t.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    return t

def mixup_cutmix(x, y, num_classes, alpha_mix=0.8, alpha_cut=1.0, p_cutmix=0.5, label_smoothing=0.1, device=None):
    bs = x.size(0)
    onehot = smooth_one_hot(y, num_classes, label_smoothing, device=device)
    perm = torch.randperm(bs, device=device)
    x2, y2 = x[perm], onehot[perm]

    use_cutmix = torch.rand((), device=device) < p_cutmix
    if use_cutmix and alpha_cut > 0:
        lam = torch.distributions.Beta(alpha_cut, alpha_cut).sample().to(device)
        H, W = x.shape[-2:]
        cut_w = int(W * torch.sqrt(1 - lam))
        cut_h = int(H * torch.sqrt(1 - lam))
        cx = torch.randint(W, (1,), device=device)
        cy = torch.randint(H, (1,), device=device)
        x1 = torch.clamp(cx - cut_w // 2, 0, W)
        y1 = torch.clamp(cy - cut_h // 2, 0, H)
        x2c = torch.clamp(cx + cut_w // 2, 0, W)
        y2c = torch.clamp(cy + cut_h // 2, 0, H)
        x[:, :, y1:y2c, x1:x2c] = x2[:, :, y1:y2c, x1:x2c]
        lam = 1 - ((x2c - x1) * (y2c - y1) / (W * H))
        targets = onehot * lam + y2 * (1 - lam)
        return x, targets, float(lam)
    else:
        if alpha_mix > 0:
            lam = torch.distributions.Beta(alpha_mix, alpha_mix).sample().to(device)
        else:
            lam = torch.tensor(1.0, device=device)
        xmix = x * lam + x2 * (1 - lam)
        targets = onehot * lam + y2 * (1 - lam)
        return xmix, targets, float(lam)

def soft_ce_loss(logits, soft_targets):
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=1).mean()

def cosine_scheduler(base_lr, steps, warmup_steps=500, min_lr=1e-6):
    lr_list = []
    for t in range(steps):
        if t < warmup_steps:
            lr = base_lr * (t + 1) / warmup_steps
        else:
            progress = (t - warmup_steps) / max(1, steps - warmup_steps)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        lr_list.append(lr)
    return lr_list

def kd_loss(student_logits, teacher_logits, T=2.0):
    p_t = F.softmax(teacher_logits / T, dim=-1)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T*T)

# ------------------------- Teacher -------------------------
def build_teacher(num_classes: int):
    """Torchvision ResNet-18 teacher adapted to CIFAR (no initial downsampling)."""
    m = models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--mixup", type=float, default=0.8)
    ap.add_argument("--cutmix", type=float, default=1.0)
    ap.add_argument("--p_cutmix", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--erasing", type=float, default=0.25)
    ap.add_argument("--color_jitter", type=float, default=0.2)
    ap.add_argument("--randaugment_n", type=int, default=2)
    ap.add_argument("--randaugment_m", type=int, default=9)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true")
    # Deep model options
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--stages", type=str, default="128,256", help="Comma-separated num_clauses per stage, e.g., 128,256,256")
    ap.add_argument("--stage_heads", type=str, default="2,4", help="Comma-separated attention heads per stage")
    ap.add_argument("--stage_bottlenecks", type=str, default="0,0", help="Comma-separated bottleneck dims per stage (0 for none)")


    # FPTM hyperparams
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--num_clauses", type=int, default=256)
    ap.add_argument("--attention_heads", type=int, default=4)
    ap.add_argument("--automata_states", type=int, default=75)
    ap.add_argument("--normalize_mode", type=str, default="none", choices=["none", "minmax"])
    ap.add_argument("--reinforce_every", type=int, default=1)
    ap.add_argument("--specificity_s", type=float, default=3.0)

    # Distillation
    ap.add_argument("--distill", action="store_true")
    ap.add_argument("--teacher_ckpt", type=str, default="")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    num_classes = 10 if args.dataset == "cifar10" else 100

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=args.randaugment_n, magnitude=args.randaugment_m),
        transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter, 0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=args.erasing, scale=(0.02, 0.25), value='random'),
    ])
    test_tf = transforms.Compose([transforms.ToTensor()])

    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(args.data, train=True, download=True, transform=train_tf)
        test_set  = datasets.CIFAR10(args.data, train=False, download=True, transform=test_tf)
    else:
        train_set = datasets.CIFAR100(args.data, train=True, download=True, transform=train_tf)
        test_set  = datasets.CIFAR100(args.data, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=512, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    
    if args.deep:
        stages = [int(x) for x in args.stages.split(',') if x.strip()]
        heads = [int(x) for x in args.stage_heads.split(',') if x.strip()]
        bots  = [int(x) for x in args.stage_bottlenecks.split(',') if x.strip()]
        model = FPTMConvDeep(in_channels=3, image_size=32, patch_size=args.patch_size,
                             stages_num_clauses=stages, stages_heads=heads, stages_bottlenecks=bots,
                             num_classes=num_classes, epsilon=1e-6, automata_states=args.automata_states,
                             normalize_mode=args.normalize_mode).to(device)
    else:
        model = FPTMConv(in_channels=3, image_size=32, patch_size=args.patch_size,
                         num_clauses=args.num_clauses, num_classes=num_classes,
                         attention_heads=args.attention_heads, automata_states=args.automata_states,
                         normalize_mode=args.normalize_mode).to(device)


    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Teacher (optional)
    teacher = None
    if args.distill:
        teacher = build_teacher(num_classes).to(device)
        if args.teacher_ckpt and os.path.isfile(args.teacher_ckpt):
            teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
        # quick warmup of teacher for small datasets is recommended beforehand
        teacher.eval()

    # Opt & sched
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-8, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = EMA(model, decay=args.ema) if args.ema > 0 else None

    total_steps = args.epochs * len(train_loader)
    effective_lr = args.base_lr * (args.batch_size / 256.0)
    sched = cosine_scheduler(effective_lr, total_steps, warmup_steps=args.warmup_steps, min_lr=1e-6)

    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def evaluate(use_ema=False, tta_flip=True):
        was_training = model.training
        model.eval()
        if use_ema and ema is not None:
            ema.apply_shadow(model)
        correct, total, ece_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(x)
                    if tta_flip:
                        logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                ece_sum += compute_ece(logits, y) * y.size(0)
        if use_ema and ema is not None:
            ema.restore(model)
        if was_training: model.train()
        return correct / total, ece_sum / total

    step = 0
    best_acc = 0.0
    model.train()
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # LR per step
            for pg in opt.param_groups:
                pg["lr"] = sched[min(step, total_steps - 1)]

            # Mixup/CutMix + soft labels
            x_mix, soft_y, lam = mixup_cutmix(
                x, y, num_classes,
                alpha_mix=args.mixup, alpha_cut=args.cutmix, p_cutmix=args.p_cutmix,
                label_smoothing=args.label_smoothing, device=device
            )

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x_mix)
                if args.distill and teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x_mix)
                    loss = 0.5 * soft_ce_loss(logits, soft_y) + 0.5 * kd_loss(logits, t_logits, T=2.0)
                else:
                    loss = soft_ce_loss(logits, soft_y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            # Fuzzy reinforcement (periodic)
            if (step % max(1, args.reinforce_every)) == 0:
                with torch.no_grad():
                    hard_y = y  # use original hard labels for feedback
                    preds = logits.argmax(dim=-1)
                    model.reinforce(x, hard_y, preds, s=args.specificity_s)

            if ema is not None:
                ema.update(model)

            step += 1

        acc_ema, ece = evaluate(use_ema=True, tta_flip=True)
        best_acc = max(best_acc, acc_ema)
        print(f"Epoch {epoch+1}/{args.epochs} | loss {loss.item():.3f} | acc_ema {acc_ema*100:.2f}% | ECE {ece:.3f} | best {best_acc*100:.2f}%")

    print("Training done. Best EMA Acc:", best_acc)

if __name__ == "__main__":
    main()
