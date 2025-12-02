"""Unit tests for get_masks() method in TM classes."""
import torch
from fptm_ste.tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM


def test_ste_get_masks_shape_and_count():
    """Test that FuzzyPatternTM_STE.get_masks() returns correct number and shape of tensors."""
    n_features = 96
    n_clauses = 128
    n_classes = 10
    
    tm = FuzzyPatternTM_STE(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        tau=0.5
    )
    
    masks = tm.get_masks(use_ste=True)
    assert len(masks) == 4, f"Expected 4 masks, got {len(masks)}"
    
    p_pos, p_neg, p_pos_inv, p_neg_inv = masks
    half = n_clauses // 2
    
    assert p_pos.shape == (half, n_features), f"p_pos shape {p_pos.shape} != ({half}, {n_features})"
    assert p_neg.shape == (half, n_features), f"p_neg shape {p_neg.shape} != ({half}, {n_features})"
    assert p_pos_inv.shape == (half, n_features), f"p_pos_inv shape {p_pos_inv.shape} != ({half}, {n_features})"
    assert p_neg_inv.shape == (half, n_features), f"p_neg_inv shape {p_neg_inv.shape} != ({half}, {n_features})"
    
    print("✓ FuzzyPatternTM_STE.get_masks() returns correct shapes")


def test_ste_get_masks_range():
    """Test that FuzzyPatternTM_STE.get_masks() returns values in [0,1] range."""
    n_features = 96
    n_clauses = 128
    n_classes = 10
    
    tm = FuzzyPatternTM_STE(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        tau=0.5
    )
    
    masks = tm.get_masks(use_ste=True)
    
    for i, mask in enumerate(masks):
        assert mask.min() >= 0.0, f"Mask {i} has values < 0: {mask.min()}"
        assert mask.max() <= 1.0, f"Mask {i} has values > 1: {mask.max()}"
    
    print("✓ FuzzyPatternTM_STE.get_masks() returns values in [0,1] range")


def test_stcm_get_masks_shape_and_count():
    """Test that FuzzyPatternTM_STCM.get_masks() returns correct number and shape of tensors."""
    n_features = 96
    n_clauses = 128
    n_classes = 10
    
    tm = FuzzyPatternTM_STCM(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        operator="capacity",
        ternary_voting=False,
        ternary_band=0.1,
        ste_temperature=1.0
    )
    
    masks = tm.get_masks(use_ste=True)
    assert len(masks) == 4, f"Expected 4 masks, got {len(masks)}"
    
    pos_pos, neg_pos, pos_inv, neg_inv = masks
    half = n_clauses // 2
    
    assert pos_pos.shape == (half, n_features), f"pos_pos shape {pos_pos.shape} != ({half}, {n_features})"
    assert neg_pos.shape == (half, n_features), f"neg_pos shape {neg_pos.shape} != ({half}, {n_features})"
    assert pos_inv.shape == (half, n_features), f"pos_inv shape {pos_inv.shape} != ({half}, {n_features})"
    assert neg_inv.shape == (half, n_features), f"neg_inv shape {neg_inv.shape} != ({half}, {n_features})"
    
    print("✓ FuzzyPatternTM_STCM.get_masks() returns correct shapes")


def test_stcm_get_masks_range():
    """Test that FuzzyPatternTM_STCM.get_masks() returns split masks in [0,1] range."""
    n_features = 96
    n_clauses = 128
    n_classes = 10
    
    tm = FuzzyPatternTM_STCM(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        operator="capacity",
        ternary_voting=False,
        ternary_band=0.1,
        ste_temperature=1.0
    )
    
    masks = tm.get_masks(use_ste=True)
    
    for i, mask in enumerate(masks):
        assert mask.min() >= 0.0, f"Mask {i} has values < 0: {mask.min()}"
        assert mask.max() <= 1.0, f"Mask {i} has values > 1: {mask.max()}"
    
    print("✓ FuzzyPatternTM_STCM.get_masks() returns split masks in [0,1] range")


def test_ste_get_masks_no_grad():
    """Test that get_masks() works without gradient tracking (inference mode)."""
    n_features = 96
    n_clauses = 128
    n_classes = 10
    
    tm = FuzzyPatternTM_STE(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        tau=0.5
    )
    
    with torch.no_grad():
        masks = tm.get_masks(use_ste=False)
        assert len(masks) == 4
    
    print("✓ FuzzyPatternTM_STE.get_masks() works without gradients")


def test_stcm_get_masks_no_grad():
    """Test that get_masks() works without gradient tracking (inference mode)."""
    n_features = 96
    n_clauses = 128
    n_classes = 10
    
    tm = FuzzyPatternTM_STCM(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        operator="capacity",
        ternary_voting=False,
        ternary_band=0.1,
        ste_temperature=1.0
    )
    
    with torch.no_grad():
        masks = tm.get_masks(use_ste=False)
        assert len(masks) == 4
    
    print("✓ FuzzyPatternTM_STCM.get_masks() works without gradients")


if __name__ == "__main__":
    print("\n=== Testing get_masks() for FuzzyPatternTM_STE ===")
    test_ste_get_masks_shape_and_count()
    test_ste_get_masks_range()
    test_ste_get_masks_no_grad()
    
    print("\n=== Testing get_masks() for FuzzyPatternTM_STCM ===")
    test_stcm_get_masks_shape_and_count()
    test_stcm_get_masks_range()
    test_stcm_get_masks_no_grad()
    
    print("\n✅ All get_masks() tests passed!")

