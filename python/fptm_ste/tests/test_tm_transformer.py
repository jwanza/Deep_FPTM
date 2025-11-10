import torch
from fptm_ste import TM_TransformerBlock, UnifiedTMTransformer


def test_tm_transformer_block():
    block = TM_TransformerBlock(d_model=64, n_heads=4, n_clauses=50)
    x = torch.rand(2, 16, 64)
    y = block(x, use_ste=True)
    assert y.shape == x.shape


def test_unified_tm_transformer():
    model = UnifiedTMTransformer(vocab_size=100, d_model=64, n_heads=4, num_layers=2, max_len=32, n_clauses=50)
    x = torch.randint(0, 100, (2, 16))
    y = model(x, use_ste=True)
    assert y.shape == (2, 16, 100)


