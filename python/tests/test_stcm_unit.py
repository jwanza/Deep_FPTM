import torch
import torch.nn.functional as F
from fptm_ste import FuzzyPatternTM_STCM, FuzzyPatternTMFPTM


def test_stcm_forward_shapes_and_masks():
    torch.manual_seed(0)
    model = FuzzyPatternTM_STCM(
        n_features=12,
        n_clauses=16,
        n_classes=3,
        clause_dropout=0.1,
        literal_dropout=0.05,
    )
    x = torch.rand(5, 12)
    logits, clauses = model(x, use_ste=True)
    assert logits.shape == (5, 3)
    assert clauses.shape == (5, 16)

    mask = model._mask_from_logits(model.pos_logits, use_ste=False)
    assert torch.all((mask >= -1.0) & (mask <= 1.0))


def test_stcm_discretize_matches_logit_signs():
    model = FuzzyPatternTM_STCM(n_features=6, n_clauses=12, n_classes=2, ternary_band=0.1)
    with torch.no_grad():
        model.pos_logits.fill_(0.5)
        model.neg_logits.fill_(-0.5)

    bundle = model.discretize(threshold=0.1)
    assert len(bundle["positive"]) == model.n_clauses // 2
    assert all(len(clause) == model.n_features for clause in bundle["positive"])
    assert all(len(clause) == 0 for clause in bundle["positive_inv"])
    assert all(len(clause) == model.n_features for clause in bundle["negative_inv"])


def test_stcm_gradients_flow_through_masks():
    torch.manual_seed(1)
    model = FuzzyPatternTM_STCM(n_features=10, n_clauses=14, n_classes=2)
    x = torch.rand(4, 10)
    logits, _ = model(x, use_ste=True)
    targets = torch.zeros(x.shape[0], dtype=torch.long)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    assert model.pos_logits.grad is not None
    assert torch.sum(model.pos_logits.grad.abs()) > 0
    assert model.neg_logits.grad is not None


def test_stcm_literal_parameters_are_halved():
    n_features, n_clauses = 20, 24
    stcm = FuzzyPatternTM_STCM(n_features=n_features, n_clauses=n_clauses, n_classes=2)
    fptm = FuzzyPatternTMFPTM(n_features=n_features, n_clauses=n_clauses, n_classes=2)

    stcm_params = stcm.pos_logits.numel() + stcm.neg_logits.numel()
    fptm_params = (
        fptm.ta_pos.numel()
        + fptm.ta_neg.numel()
        + fptm.ta_pos_inv.numel()
        + fptm.ta_neg_inv.numel()
    )
    assert stcm_params * 2 == fptm_params


def test_capacity_and_product_operators_are_stable():
    torch.manual_seed(3)
    base_kwargs = dict(n_features=8, n_clauses=12, n_classes=2)
    cap = FuzzyPatternTM_STCM(operator="capacity", **base_kwargs)
    prod = FuzzyPatternTM_STCM(operator="product", **base_kwargs)
    with torch.no_grad():
        prod.pos_logits.copy_(cap.pos_logits)
        prod.neg_logits.copy_(cap.neg_logits)
    x = torch.rand(4, 8)

    cap_logits, cap_clauses = cap(x, use_ste=False)
    prod_logits, prod_clauses = prod(x, use_ste=False)

    assert cap_logits.shape == prod_logits.shape == (4, 2)
    assert torch.all(torch.isfinite(prod_logits))
    assert torch.all(torch.isfinite(cap_logits))

    half = cap_clauses.shape[1] // 2
    assert torch.all(cap_clauses[:, :half] >= 0)
    assert torch.all(cap_clauses[:, half:] <= 0)
    assert torch.all(prod_clauses[:, :half] >= 0)
    assert torch.all(prod_clauses[:, half:] <= 0)

