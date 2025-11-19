
target = 'python/fptm_ste/tests/run_mnist_equiv.py'
lines = open(target).readlines()

for i in range(len(lines)):
    if '"--transformer-backend",' in lines[i]:
        # The next line usually contains choices
        if "choices=[" in lines[i+1]:
             lines[i+1] = '        choices=["ste", "deeptm", "stcm", "deep_stcm"],\n'
             print("Updated transformer-backend choices.")
             with open(target, 'w') as f:
                 f.writelines(lines)
             break

