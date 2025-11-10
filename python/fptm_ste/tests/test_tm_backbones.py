import torch

from fptm_ste.resnet_tm import resnet_tm18
from fptm_ste.swin_tm import SwinTM, SwinTMStageConfig


def test_swin_tm_forward_backward():
    configs = (
        SwinTMStageConfig(
            embed_dim=16,
            depth=1,
            num_heads=1,
            window_size=7,
            shift_size=0,
            tm_hidden_dim=32,
            tm_clauses=16,
            head_clauses=12,
            dropout=0.0,
            drop_path=0.0,
            tau=0.5,
        ),
    )
    model = SwinTM(stage_configs=configs, num_classes=10, in_channels=1, image_size=(28, 28))
    x = torch.randn(2, 1, 28, 28, requires_grad=True)
    logits, stage_logits, clause_lists, final_clauses = model(x, use_ste=True)
    assert logits.shape == (2, 10)
    assert len(stage_logits) == len(configs)
    assert clause_lists[0].shape[0] == 2
    loss = logits.sum()
    loss.backward()


def test_resnet_tm_forward_shapes():
    model = resnet_tm18(num_classes=10, in_channels=1, tm_hidden=64, tm_clauses=32, tau=0.5)
    x = torch.randn(2, 1, 28, 28, requires_grad=True)
    logits, stage_logits, stage_clauses, final_clauses = model(x, use_ste=True)
    assert logits.shape == (2, 10)
    assert len(stage_logits) == 4
    for logit in stage_logits:
        assert logit.shape == (2, 10)
    loss = logits.sum()
    loss.backward()
