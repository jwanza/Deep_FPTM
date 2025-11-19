
target = 'python/fptm_ste/tests/run_mnist_equiv.py'
lines = open(target).readlines()

# Check where ema_decay_value is used
start_idx = -1
for i in range(len(lines)):
    if "ema_param.data.mul_(ema_decay_value)" in lines[i]:
        start_idx = i
        break

if start_idx != -1:
    # Find definition of run_variant_transformer
    def_idx = -1
    for i in range(start_idx, 0, -1):
         if "def run_variant_transformer" in lines[i]:
             def_idx = i
             break
    
    # Check if ema_decay is in args
    # It is: `ema_decay: float,` (line 3789 approx)
    
    # Replace variable usage or ensure correct name
    # In args it is `ema_decay`. In body it is `ema_decay_value`.
    
    # Replace usage in lines around start_idx
    for j in range(start_idx - 5, start_idx + 5):
         lines[j] = lines[j].replace("ema_decay_value", "ema_decay")
    
    print("Fixed ema_decay variable name.")
    with open(target, 'w') as f:
        f.writelines(lines)
else:
    print("EMA logic not found.")

