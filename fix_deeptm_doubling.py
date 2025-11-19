
target = 'python/fptm_ste/deep_tm.py'
lines = open(target).readlines()

# Find classifier_kwargs initialization
start_idx = -1
for i in range(len(lines)):
    if "classifier_kwargs = dict(" in lines[i]:
        start_idx = i
        break

if start_idx != -1:
    # Find n_clauses line
    for j in range(start_idx, start_idx + 10):
        if "n_clauses=n_clauses * 2," in lines[j]:
             lines[j] = lines[j].replace("* 2", "")
             print("Removed * 2 doubling from DeepTMNetwork.")
             break
             
    with open(target, 'w') as f:
        f.writelines(lines)
else:
    print("classifier_kwargs not found.")

