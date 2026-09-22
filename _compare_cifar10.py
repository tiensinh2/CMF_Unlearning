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

# Oracle (Retrain)
oracle = read_csv('notebooks/Cifar_10/Result_nb2/results_oracle_paper.xlsx')
print('=== ORACLE (Retrain, nb2) ===')
r = avg(oracle, 'output_retain_acc')
ncc_r = avg(oracle, 'ncc_retain_acc')
ncc_f = avg(oracle, 'ncc_forget_acc')
print(f'  output_retain={r:.2f}, ncc_retain={ncc_r:.2f}, ncc_forget={ncc_f:.2f}')

# No CMF methods (nb3)
nocmf = read_csv('notebooks/Cifar_10/Result_nb3/results_nocmf_paper.xlsx')
methods_nocmf = sorted(set(r2['method'] for r2 in nocmf))
print('\n=== NO-CMF methods (nb3) ===')
for m in methods_nocmf:
    rows = [r2 for r2 in nocmf if r2['method']==m]
    print(f'  {m}: output_retain={avg(rows,"output_retain_acc"):.2f}, ncc_retain={avg(rows,"ncc_retain_acc"):.2f}, ncc_forget={avg(rows,"ncc_forget_acc"):.2f}')

# CMF Static (nb4a)
static = read_csv('notebooks/Cifar_10/Result_nb4_static/results_4a_cmf_static_new.xlsx')
methods_static = sorted(set(r2['method'] for r2 in static))
print('\n=== CMF STATIC (nb4a) ===')
for m in methods_static:
    rows = [r2 for r2 in static if r2['method']==m]
    print(f'  {m}: output_retain={avg(rows,"output_retain_acc"):.2f}, ncc_retain={avg(rows,"ncc_retain_acc"):.2f}, ncc_forget={avg(rows,"ncc_forget_acc"):.2f}')

# CMF Posthoc (nb4b)
posthoc = read_csv('notebooks/Cifar_10/Result_nb4_postdoc/results_4b_cmf_posthoc.xlsx')
methods_ph = sorted(set(r2['method'] for r2 in posthoc))
print('\n=== CMF POST-HOC (nb4b) all configs avg ===')
for m in methods_ph:
    rows = [r2 for r2 in posthoc if r2['method']==m]
    print(f'  {m}: output_retain={avg(rows,"output_retain_acc"):.2f}, ncc_retain={avg(rows,"ncc_retain_acc"):.2f}, ncc_forget={avg(rows,"ncc_forget_acc"):.2f}')

# nb4d
nb4d = read_csv('notebooks/Cifar_10/Result_nb4d_after postdoc/results_4d_cmf_freezeW_stage3_with_only_cmf_again.xlsx')
methods_d4d = sorted(set(r2['method'] for r2 in nb4d))
print('\n=== CMF FREEZE-W Stage3 (nb4d) ===')
for m in methods_d4d:
    rows = [r2 for r2 in nb4d if r2['method']==m]
    print(f'  {m}: output_retain={avg(rows,"output_retain_acc"):.2f}, ncc_retain={avg(rows,"ncc_retain_acc"):.2f}, ncc_forget={avg(rows,"ncc_forget_acc"):.2f}')
