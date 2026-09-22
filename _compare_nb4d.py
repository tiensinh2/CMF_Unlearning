import sys, io, csv, statistics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def read_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def avg(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key,'') not in ('', None)]
    return statistics.mean(vals) if vals else float('nan')

f1 = 'notebooks/Cifar_10/Result_nb4d_after postdoc/results_4d_cmf_freezeW_stage3_with_only_cmf_again.xlsx'
f2 = 'notebooks/Cifar_10/Result_nb4d_after postdoc/results_4d_cmf_freezeW_stage3_with_per_epochs_modify.xlsx'

d1 = read_csv(f1)
d2 = read_csv(f2)

print('=== File 1: only_cmf_again (s3 freeze-W, CMF-only stage3) ===')
print('Columns:', list(d1[0].keys()) if d1 else 'empty')
print('Rows:', len(d1))
# check unique config columns
if d1:
    for col in ['s3_epochs', 'k_posthoc', 'phase2_data', 'lr']:
        if col in d1[0]:
            vals = sorted(set(r[col] for r in d1))
            print(f'  {col} values: {vals}')
methods1 = sorted(set(r['method'] for r in d1))
for m in methods1:
    rows = [r for r in d1 if r['method'] == m]
    line = f'  {m}: out_ret={avg(rows,"output_retain_acc"):.2f}  out_fgt={avg(rows,"output_forget_acc"):.2f}  ncc_ret={avg(rows,"ncc_retain_acc"):.2f}  ncc_fgt={avg(rows,"ncc_forget_acc"):.2f}'
    print(line)

print()
print('=== File 2: per_epochs_modify (s3 freeze-W, per-epoch lr modification) ===')
print('Columns:', list(d2[0].keys()) if d2 else 'empty')
print('Rows:', len(d2))
if d2:
    for col in ['s3_epochs', 'k_posthoc', 'phase2_data', 'lr']:
        if col in d2[0]:
            vals = sorted(set(r[col] for r in d2))
            print(f'  {col} values: {vals}')
methods2 = sorted(set(r['method'] for r in d2))
for m in methods2:
    rows = [r for r in d2 if r['method'] == m]
    line = f'  {m}: out_ret={avg(rows,"output_retain_acc"):.2f}  out_fgt={avg(rows,"output_forget_acc"):.2f}  ncc_ret={avg(rows,"ncc_retain_acc"):.2f}  ncc_fgt={avg(rows,"ncc_forget_acc"):.2f}'
    print(line)

print()
print('=== PAPER reference (Table 3, CIFAR-10, 1 class) ===')
paper = {
    'Retrain':          {'ncc_ret': 93.31, 'ncc_fgt': 47.06, 'out_ret': 94.74},
    'random_label+CMF': {'ncc_ret': 94.25, 'ncc_fgt': 82.04, 'out_ret': 94.27},
    'salun+CMF':        {'ncc_ret': 94.32, 'ncc_fgt': 79.50, 'out_ret': 94.33},
    'NegGrad++CMF':     {'ncc_ret': 91.99, 'ncc_fgt': 57.83, 'out_ret': 91.87},
    'scrub+CMF':        {'ncc_ret': 92.48, 'ncc_fgt': 35.53, 'out_ret': 92.51},
    'unsir/tarun+CMF':  {'ncc_ret': 91.87, 'ncc_fgt':  9.29, 'out_ret': 91.79},
}
for k, v in paper.items():
    print(f'  {k}: out_ret={v["out_ret"]:.2f}  ncc_ret={v["ncc_ret"]:.2f}  ncc_fgt={v["ncc_fgt"]:.2f}')

# Also break down file2 by config if there are multiple
print()
print('=== File 2 breakdown by s3_epochs and phase2_data ===')
if d2 and 's3_epochs' in d2[0]:
    for s3 in sorted(set(r['s3_epochs'] for r in d2)):
        for ph in sorted(set(r.get('phase2_data','') for r in d2)):
            rows = [r for r in d2 if r['s3_epochs']==s3 and r.get('phase2_data','')==ph]
            if not rows:
                continue
            print(f'  s3_epochs={s3}, phase2_data={ph}, n={len(rows)}')
            for m in sorted(set(r['method'] for r in rows)):
                mr = [r for r in rows if r['method']==m]
                line = f'    {m}: out_ret={avg(mr,"output_retain_acc"):.2f}  out_fgt={avg(mr,"output_forget_acc"):.2f}  ncc_ret={avg(mr,"ncc_retain_acc"):.2f}  ncc_fgt={avg(mr,"ncc_forget_acc"):.2f}'
                print(line)
