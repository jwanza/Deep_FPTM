import torch
from fptm_ste import SwinFeatureExtractor, MultiScaleTMEnsemble


def test_multiscale_tm_with_synthetic_features():
    # Avoid network: do not instantiate backbone here, just test ensemble
    B = 2
    feature_dims = [96, 192, 384, 768]
    feats = [
        torch.rand(B, feature_dims[0], 56, 56),
        torch.rand(B, feature_dims[1], 28, 28),
        torch.rand(B, feature_dims[2], 14, 14),
        torch.rand(B, feature_dims[3], 7, 7),
    ]
    head_configs = [
        {"stages": [0], "n_clauses": 100, "thresholds": 8, "binarizer": "dual"},
        {"stages": [1], "n_clauses": 80, "thresholds": 8, "binarizer": "dual"},
        {"stages": [2], "n_clauses": 60, "thresholds": 8, "binarizer": "dual"},
        {"stages": [3], "n_clauses": 40, "thresholds": 8, "binarizer": "dual"},
    ]
    ens = MultiScaleTMEnsemble(feature_dims, n_classes=10, head_configs=head_configs, backbone_type="swin", init_temperature=1.0)
    logits, scale_logits, clause_outputs = ens(feats, use_ste=True)
    assert logits.shape == (B, 10)
    assert len(scale_logits) == 4
    assert len(clause_outputs) == 4


def test_swin_feature_extractor_instantiation():
    # Smoke: instantiate without pretrained to avoid download in CI
    model = SwinFeatureExtractor(pretrained=False, freeze=True)
    x = torch.rand(1, 3, 224, 224)
    feats = model(x)
    assert isinstance(feats, list)
    assert len(feats) == 4
    assert feats[0].shape[2] in (56, 28)  # resolution check


