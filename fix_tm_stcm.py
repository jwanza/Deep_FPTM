
target = 'python/fptm_ste/tm.py'
lines = open(target).readlines()

# Fix __init__ to restore operator_impl
for i in range(480, 500):
    if "self.operator = operator" in lines[i]:
        # Check next lines
        # Original code had:
        # operator = operator.lower()
        # custom_ops = set(available_ternary_operators())
        # base_ops = {"capacity", "product"}
        # if operator not in base_ops | custom_ops: ...
        # self.operator = operator
        # self.operator_impl = build_ternary_operator(operator)
        
        # Current code has simplified operator check.
        pass

# Fix _strength to use operator_impl
for i in range(540, 560):
    if "def _strength(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:" in lines[i]:
        # Insert implementation check
        lines.insert(i+3, "        if hasattr(self, 'operator_impl') and self.operator_impl is not None:\n")
        lines.insert(i+4, "            matches = self._match_triplet(x, mask_pos, mask_inv)\n")
        lines.insert(i+5, "            return self.operator_impl(*matches)\n")
        print("Restored operator_impl logic in _strength.")
        break

with open(target, 'w') as f:
    f.writelines(lines)

