
target = 'python/fptm_ste/tests/run_mnist_equiv.py'
lines = open(target).readlines()

# Find where run_variant_transformer is defined
start_idx = -1
for i in range(len(lines)):
    if "def run_variant_transformer" in lines[i]:
        start_idx = i
        break

if start_idx != -1:
    # Look for epoch loop
    epoch_loop = -1
    for i in range(start_idx, start_idx + 500):
        if "for epoch in range(epochs):" in lines[i]:
             epoch_loop = i
             break
    
    if epoch_loop != -1:
        # Look for initialization block inside epoch loop
        # It should be before `for batch_idx, (x, y)`
        
        # I need to insert `total_self_distill = 0.0` inside epoch loop.
        
        # Find running_loss = 0.0
        for j in range(epoch_loop, epoch_loop + 100):
             if "running_loss = 0.0" in lines[j]:
                  # Check if total_self_distill is nearby
                  if "total_self_distill = 0.0" not in lines[j+4] and "total_self_distill = 0.0" not in lines[j+5]:
                       # Insert it
                       indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                       lines.insert(j+1, indent + "total_self_distill = 0.0\n")
                       lines.insert(j+2, indent + "self_distill_count = 0\n")
                       print("Inserted total_self_distill init.")
                       with open(target, 'w') as f:
                           f.writelines(lines)
                       break
else:
    print("run_variant_transformer not found.")

