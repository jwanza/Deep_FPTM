import os
import json
import torch
from fptm_ste import FuzzyPatternTM_STE, export_compiled_to_json


def test_tm_forward_and_discretize(tmp_path):
    torch.manual_seed(0)
    B, F, C = 4, 32, 3
    tm = FuzzyPatternTM_STE(n_features=F, n_clauses=20, n_classes=C)
    x = torch.rand(B, F)
    logits, clauses = tm(x, use_ste=True)
    assert logits.shape == (B, C)
    assert clauses.shape == (B, 20)

    bundle = tm.discretize(threshold=0.5)
    for k in ["positive", "negative", "positive_inv", "negative_inv", "clauses_num"]:
        assert k in bundle

    out_json = os.path.join(tmp_path, "tm.json")
    export_compiled_to_json(bundle, class_labels=["0", "1", "2"], path=out_json, clauses_num=tm.n_clauses, L=16, LF=4)
    with open(out_json, "r") as f:
        data = json.load(f)
    assert "classes" in data and len(data["classes"]) == 3


