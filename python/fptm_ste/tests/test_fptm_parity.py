import pytest
import torch
import torch.nn as nn
from fptm_ste.tm import FuzzyPatternTMFPTM, FuzzyPatternTM_STE, FuzzyPatternTM_STCM
from fptm_ste.trainers import TsetlinMarginLoss

def test_fptm_team_per_class_init():
    # Valid init: half clauses (10) divisible by n_classes (2)
    model = FuzzyPatternTMFPTM(n_features=10, n_clauses=20, n_classes=2, team_per_class=True)
    assert model.voting is None
    assert model.team_per_class is True
    
    # Invalid init (half=10, n_classes=3 -> 10%3!=0)
    with pytest.raises(ValueError):
        FuzzyPatternTMFPTM(n_features=10, n_clauses=20, n_classes=3, team_per_class=True)

def test_fptm_team_per_class_forward():
    n_clauses = 4
    n_classes = 2
    # half=2, n_classes=2 -> divisible
    model = FuzzyPatternTMFPTM(n_features=4, n_clauses=n_clauses, n_classes=n_classes, team_per_class=True)
    x = torch.rand(1, 4)
    logits, votes = model(x)
    assert logits.shape == (1, n_classes)
    assert votes.shape == (1, n_clauses)

def test_prune_ste():
    model = FuzzyPatternTM_STE(n_features=10, n_clauses=10, n_classes=2)
    # Force one weight to be very small (logit < -5) and one large
    with torch.no_grad():
        model.ta_pos.data.fill_(10.0) # High prob
        model.ta_pos.data[0,0] = -2.0 # Lowish prob (sigmoid(-2) = 0.119)
    
    # Prune with threshold 0.2 (logit threshold ~ -1.38)
    # -2.0 < -1.38, so it should be pruned to -10.0
    model.prune(threshold=0.2)
    
    assert model.ta_pos.data[0,0] == -10.0
    assert model.ta_pos.data[0,1] == 10.0

def test_prune_stcm():
    model = FuzzyPatternTM_STCM(n_features=10, n_clauses=10, n_classes=2)
    with torch.no_grad():
        model.pos_logits.data.fill_(10.0)
        model.pos_logits.data[0,0] = 0.05 # Close to 0
    
    # Prune threshold 0.1 -> |0.05| < 0.1 -> should be 0
    model.prune(threshold=0.1)
    
    assert model.pos_logits.data[0,0] == 0.0
    assert model.pos_logits.data[0,1] == 10.0

def test_margin_loss():
    loss_fn = TsetlinMarginLoss(T=1.0)
    # Perfect prediction: Correct class score 2.0 (> T), Incorrect score -2.0 (< -T)
    logits = torch.tensor([[2.0, -2.0]])
    target = torch.tensor([0])
    loss = loss_fn(logits, target)
    # Loss correct: ReLU(1 - 2) = 0
    # Loss incorrect: ReLU(1 + (-2)) = 0
    # Total = 0
    assert loss.item() == 0.0
    
    # Bad prediction: Correct class score 0.0 (< T), Incorrect score 0.0 (> -T)
    logits = torch.tensor([[0.0, 0.0]])
    loss = loss_fn(logits, target)
    # Loss correct: ReLU(1 - 0) = 1
    # Loss incorrect: ReLU(1 + 0) = 1
    # Total = 2
    assert loss.item() == 2.0

