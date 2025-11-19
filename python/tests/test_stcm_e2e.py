import torch
import torch.nn.functional as F

import pytest

from fptm_ste import DeepTMNetwork, FuzzyPatternTM_STCM


def _train_simple(model, x, y, steps=8, lr=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        logits, _ = model(x, use_ste=True)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
    return losses, logits


def test_stcm_learns_conjunctive_rule():
    torch.manual_seed(4)
    samples = 64
    x = (torch.rand(samples, 4) > 0.5).float()
    targets = ((x[:, 0] > 0.5) & (x[:, 1] < 0.5)).long()

    model = FuzzyPatternTM_STCM(n_features=4, n_clauses=20, n_classes=2, operator="capacity")
    losses, logits = _train_simple(model, x, targets, steps=12, lr=0.08)

    assert losses[-1] < losses[0]
    preds = logits.argmax(dim=1)
    acc = (preds == targets).float().mean()
    assert acc >= 0.8


def test_capacity_and_product_improve_loss_on_fuzzy_data():
    torch.manual_seed(5)
    x = torch.rand(80, 6)
    y = ((x[:, 0] + (1 - x[:, 1])) > 1.1).long()

    cap = FuzzyPatternTM_STCM(n_features=6, n_clauses=18, n_classes=2, operator="capacity")
    prod = FuzzyPatternTM_STCM(n_features=6, n_clauses=18, n_classes=2, operator="product")

    cap_losses, _ = _train_simple(cap, x, y, steps=10, lr=0.05)
    prod_losses, _ = _train_simple(prod, x, y, steps=10, lr=0.05)

    assert cap_losses[-1] < cap_losses[0]
    assert prod_losses[-1] < prod_losses[0]
    assert abs(cap_losses[-1] - prod_losses[-1]) < 0.2


def test_continuous_vs_ternary_voting_accuracy_gap_small():
    torch.manual_seed(6)
    x = torch.rand(72, 8)
    y = (x[:, 0] > 0.4).long()

    cont = FuzzyPatternTM_STCM(n_features=8, n_clauses=16, n_classes=2, ternary_voting=False)
    tern = FuzzyPatternTM_STCM(n_features=8, n_clauses=16, n_classes=2, ternary_voting=True)

    _, cont_logits = _train_simple(cont, x, y, steps=10, lr=0.04)
    _, tern_logits = _train_simple(tern, x, y, steps=10, lr=0.04)

    cont_acc = (cont_logits.argmax(1) == y).float().mean()
    tern_acc = (tern_logits.argmax(1) == y).float().mean()
    assert abs(cont_acc.item() - tern_acc.item()) <= 0.15


def test_deep_tm_network_uses_stcm_layers():
    torch.manual_seed(7)
    x = torch.rand(32, 12)
    y = (x[:, 0] > 0.5).long()

    model = DeepTMNetwork(
        input_dim=12,
        hidden_dims=[16],
        n_classes=2,
        n_clauses=12,
        layer_cls=FuzzyPatternTM_STCM,
        layer_operator="capacity",
    )
    logits, _ = model(x, use_ste=True)
    assert logits.shape == (32, 2)
    assert torch.isfinite(logits).all()
    losses, logits = _train_simple(model, x, y, steps=6, lr=0.05)
    assert torch.isfinite(torch.tensor(losses[-1]))


@pytest.mark.parametrize(
    "operator,feature_dim,target_fn,threshold,n_clauses,steps,lr",
    [
        (
            "tqand",
            4,
            lambda x: ((x[:, 0] > 0.5) & (x[:, 1] > 0.5)).long(),
            0.84,
            40,
            20,
            0.07,
        ),
        (
            "txor",
            4,
            lambda x: (x[:, 0] != x[:, 1]).long(),
            0.75,
            32,
            30,
            0.08,
        ),
        (
            "tmaj",
            5,
            lambda x: ((x[:, 0] + x[:, 1] + x[:, 2]) > 1.5).long(),
            0.8,
            24,
            18,
            0.06,
        ),
    ],
)
def test_custom_operators_learn_boolean_rules(
    operator, feature_dim, target_fn, threshold, n_clauses, steps, lr
):
    torch.manual_seed(8)
    x = (torch.rand(96, feature_dim) > 0.5).float()
    y = target_fn(x)
    model = FuzzyPatternTM_STCM(
        n_features=feature_dim,
        n_clauses=n_clauses,
        n_classes=2,
        operator=operator,
        ternary_band=0.1,
    )
    losses, logits = _train_simple(model, x, y, steps=steps, lr=lr)
    assert losses[-1] < losses[0]
    acc = (logits.argmax(1) == y).float().mean()
    assert acc >= threshold

