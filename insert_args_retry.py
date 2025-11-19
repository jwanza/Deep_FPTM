
target = 'python/fptm_ste/tests/run_mnist_equiv.py'
lines = open(target).readlines()
insert_idx = 1946 - 1 # Approximate location
for i in range(len(lines)):
    if "return parser" in lines[i]:
        insert_idx = i
        break

block = """
    parser.add_argument(
        "--distill-contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for contrastive feature distillation (student clauses vs teacher features).",
    )
    parser.add_argument(
        "--distill-contrastive-temp",
        type=float,
        default=0.07,
        help="Temperature for contrastive feature distillation.",
    )
"""
lines.insert(insert_idx, block)
with open(target, 'w') as f:
    f.writelines(lines)
print("Inserted distill args again.")

