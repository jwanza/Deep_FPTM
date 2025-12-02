# Quick Demo Commands for Deep-CTM

## 1. Baseline (Fast, ~96% in 10 epochs)

```bash
cd /nvme0n1-disk/shared/joel/FuzzyPatternTM_4GTM
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/nvme0n1-disk/shared/joel/FuzzyPatternTM_4GTM/python

python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist \
  --models deep_ctm \
  --epochs 10 \
  --batch-size 128 \
  --deepctm-channels 32,64,128 \
  --deepctm-core stcm \
  --deepctm-aux-weight 0.5 \
  --deepctm-dropout 0.1 \
  --stcm-operator capacity \
  --stcm-ternary-band 0.1
```

**Expected:** ~96.5% test accuracy in 10 epochs (~5 minutes on H100)

## 2. With 4 Blocks (Better, ~97.5% in 10 epochs)

```bash
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist \
  --models deep_ctm \
  --epochs 10 \
  --batch-size 128 \
  --deepctm-channels 32,64,128,256 \
  --deepctm-kernels 5,5,3,3 \
  --deepctm-pools 2,2,2,2 \
  --deepctm-clauses 128,128,128,128 \
  --deepctm-head-clauses 256 \
  --deepctm-core stcm \
  --deepctm-aux-weight 0.5 \
  --deepctm-dropout 0.1 \
  --stcm-operator capacity \
  --stcm-ternary-band 0.1
```

**Expected:** ~97.5% test accuracy in 10 epochs

## 3. With Attention Head (Best, ~98% in 10 epochs)

```bash
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist \
  --models deep_ctm \
  --epochs 10 \
  --batch-size 128 \
  --deepctm-channels 32,64,128,256 \
  --deepctm-kernels 5,5,3,3 \
  --deepctm-pools 2,2,2,2 \
  --deepctm-clauses 128,128,128,128 \
  --deepctm-head-clauses 256 \
  --deepctm-core stcm \
  --deepctm-aux-weight 0.5 \
  --deepctm-dropout 0.1 \
  --stcm-operator capacity \
  --stcm-ternary-band 0.1 \
  --head-attention \
  --head-attention-dim 256 \
  --head-attention-heads 4 \
  --head-attention-dropout 0.1
```

**Expected:** ~98% test accuracy in 10 epochs

## 4. Hybrid Heads (TM + Attention + Linear)

```bash
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist \
  --models deep_ctm \
  --epochs 10 \
  --batch-size 128 \
  --deepctm-channels 32,64,128,256 \
  --deepctm-kernels 5,5,3,3 \
  --deepctm-pools 2,2,2,2 \
  --deepctm-clauses 128,128,128,128 \
  --deepctm-head-clauses 256 \
  --deepctm-core stcm \
  --deepctm-aux-weight 0.5 \
  --deepctm-dropout 0.1 \
  --stcm-operator capacity \
  --stcm-ternary-band 0.1 \
  --head-type stcm \
  --head-attention \
  --head-attention-dim 256 \
  --head-attention-heads 4 \
  --head-linear \
  --head-linear-hidden 512
```

**Expected:** ~98% test accuracy in 10 epochs, automatic head mixing

## 5. SOTA Configuration (99% target, 100 epochs)

```bash
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist \
  --models deep_ctm \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.001 \
  --deepctm-channels 32,64,128,256 \
  --deepctm-kernels 5,5,3,3 \
  --deepctm-strides 1,1,1,1 \
  --deepctm-pools 2,2,2,2 \
  --deepctm-clauses 256,256,256,256 \
  --deepctm-head-clauses 512 \
  --deepctm-tau 0.5 \
  --deepctm-dropout 0.15 \
  --deepctm-aux-weight 0.6 \
  --deepctm-core stcm \
  --stcm-operator capacity \
  --stcm-ternary-voting \
  --stcm-ternary-band 0.1 \
  --stcm-ste-temperature 1.0 \
  --head-type stcm \
  --head-attention \
  --head-attention-dim 256 \
  --head-attention-heads 4 \
  --head-attention-dropout 0.1 \
  --head-linear \
  --head-linear-hidden 512 \
  --head-linear-dropout 0.1
```

**Expected Timeline:**
- Epoch 20: ~97.5%
- Epoch 50: ~98.5%
- Epoch 100: ~99.0%

## Monitoring Output

You'll see output like:
```
Deep-CTM | epoch 10/10 | loss=0.3369 | train_acc=0.9673 | test_acc=0.9654 | best_acc=0.9654
         | block_1_te=0.668, block_2_te=0.803, block_3_te=0.971, head_tm_te=0.965
         | block_1_tr=0.668, block_2_tr=0.820, block_3_tr=0.977, head_tm_tr=0.977
```

Key metrics:
- `test_acc=0.9654`: Final accuracy (96.54%)
- `block_1_te=0.668`: Block 1 can classify at 66.8% (shows it's learning!)
- `block_3_te=0.971`: Block 3 reaches 97.1% (nearly perfect features)

## Quick Performance Test

To verify the optimization is working:
```bash
# Test base vs optimized speed
time python python/fptm_ste/tests/test_conv_optimized_equivalence.py
```

Should show:
- Forward pass: max_diff < 5e-7 ✅
- Backward pass: max_diff < 3e-11 ✅
- All tests pass in ~2 seconds

