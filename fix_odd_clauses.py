
target = 'python/fptm_ste/tests/run_mnist_equiv.py'
lines = open(target).readlines()

# Find where _scale_clause_counts is called or defined
# We want to ensure it returns even numbers.

start_idx = -1
for i in range(len(lines)):
    if "def _scale_clause_counts" in lines[i]:
        start_idx = i
        break

if start_idx != -1:
    # Find where new_counts.append happens
    for j in range(start_idx, start_idx + 20):
        if "new_counts.append(max(8, int(round(base * combined))))" in lines[j]:
            # Change to ensure even
            lines[j] = "        val = max(8, int(round(base * combined)))\n        if val % 2 != 0: val += 1\n        new_counts.append(val)\n"
    
    # Find where final scaling happens
    for j in range(start_idx + 20, start_idx + 30):
        if "new_counts = [max(8, int(round(c * scale))) for c in new_counts]" in lines[j]:
             # We need to fix this too.
             lines[j] = "            new_counts = [max(8, int(round(c * scale))) for c in new_counts]\n            new_counts = [c + 1 if c % 2 != 0 else c for c in new_counts]\n"

    print("Fixed clause scaling to enforce even numbers.")
    with open(target, 'w') as f:
        f.writelines(lines)
else:
    print("Function not found.")

