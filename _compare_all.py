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

def by_method(rows, methods=None):
    if methods is None:
        methods = sorted(set(r['method'] for r in rows))
    result = {}
    for m in methods:
        mr = [r for r in rows if r['method'] == m]
        result[m] = {
            'out_ret': avg(mr, 'output_retain_acc'),
            'out_fgt': avg(mr, 'output_forget_acc'),
            'ncc_ret': avg(mr, 'ncc_retain_acc'),
            'ncc_fgt': avg(mr, 'ncc_forget_acc'),
        }
    return result

# Load all datasets
oracle  = read_csv('notebooks/Cifar_10/Result_nb2/results_oracle_paper.xlsx')
nocmf   = read_csv('notebooks/Cifar_10/Result_nb3/results_nocmf_paper.xlsx')
static  = read_csv('notebooks/Cifar_10/Result_nb4_static/results_4a_cmf_static_new.xlsx')
posthoc = read_csv('notebooks/Cifar_10/Result_nb4_postdoc/results_4b_cmf_posthoc.xlsx')
nb4d1   = read_csv('notebooks/Cifar_10/Result_nb4d_after postdoc/results_4d_cmf_freezeW_stage3_with_only_cmf_again.xlsx')
nb4d2   = read_csv('notebooks/Cifar_10/Result_nb4d_after postdoc/results_4d_cmf_freezeW_stage3_with_per_epochs_modify.xlsx')

METHODS = ['grad_ascent_descent', 'random_label', 'salun', 'scrub', 'tarun']

# compute
oracle_v = {'out_ret': avg(oracle,'output_retain_acc'), 'out_fgt': avg(oracle,'output_forget_acc'),
            'ncc_ret': avg(oracle,'ncc_retain_acc'),    'ncc_fgt': avg(oracle,'ncc_forget_acc')}

nocmf_v   = by_method(nocmf, METHODS)
static_v  = by_method(static, METHODS)
posthoc_v = by_method(posthoc, METHODS)
nb4d1_v   = by_method(nb4d1, METHODS)
nb4d2_v   = by_method(nb4d2, METHODS)

PAPER_NOCMF = {
    'grad_ascent_descent': {'ncc_ret': 91.33, 'ncc_fgt': 52.00, 'out_ret': 92.85, 'out_fgt': 0.00},
    'random_label':        {'ncc_ret': 92.25, 'ncc_fgt': 80.25, 'out_ret': 92.93, 'out_fgt': 0.00},
    'salun':               {'ncc_ret': 91.31, 'ncc_fgt': 93.70, 'out_ret': 93.19, 'out_fgt': 0.00},
    'scrub':               {'ncc_ret': 89.96, 'ncc_fgt': 55.30, 'out_ret': 91.37, 'out_fgt': 0.00},
    'tarun':               {'ncc_ret': 89.73, 'ncc_fgt': 62.72, 'out_ret': 91.84, 'out_fgt': 0.48},
}
PAPER_CMF = {
    'random_label': {'ncc_ret': 94.25, 'ncc_fgt': 82.04, 'out_ret': 94.27, 'out_fgt': 80.70},
    'salun':        {'ncc_ret': 94.32, 'ncc_fgt': 79.50, 'out_ret': 94.33, 'out_fgt': 78.07},
    'scrub':        {'ncc_ret': 92.48, 'ncc_fgt': 35.53, 'out_ret': 92.51, 'out_fgt': 33.78},
    'tarun':        {'ncc_ret': 91.87, 'ncc_fgt':  9.29, 'out_ret': 91.79, 'out_fgt': 12.91},
    'grad_ascent_descent': {'ncc_ret': 91.99, 'ncc_fgt': 57.83, 'out_ret': 91.87, 'out_fgt': 54.50},
}
PAPER_RETRAIN = {'ncc_ret': 93.31, 'ncc_fgt': 47.06, 'out_ret': 94.74, 'out_fgt': 0.00}

METHOD_LABELS = {
    'grad_ascent_descent': 'NegGrad+',
    'random_label':        'Random-label',
    'salun':               'SalUn',
    'scrub':               'SCRUB',
    'tarun':               'UNSIR/Tarun',
}

VARIANTS = [
    ('Paper (no CMF)',    PAPER_NOCMF,  False),
    ('nb3 (no CMF)',      nocmf_v,      False),
    ('Paper + CMF',       PAPER_CMF,    True),
    ('nb4a CMF Static',   static_v,     True),
    ('nb4b CMF Posthoc',  posthoc_v,    True),
    ('nb4d-1 FreezeW CMF-only', nb4d1_v, True),
    ('nb4d-2 FreezeW PerEpoch', nb4d2_v, True),
]

print('RETRAIN TARGET: out_ret=%.2f  out_fgt=%.2f  ncc_ret=%.2f  ncc_fgt=%.2f' % (
    PAPER_RETRAIN['out_ret'], PAPER_RETRAIN['out_fgt'],
    PAPER_RETRAIN['ncc_ret'], PAPER_RETRAIN['ncc_fgt']))
print('OUR ORACLE nb2: out_ret=%.2f  out_fgt=%.2f  ncc_ret=%.2f  ncc_fgt=%.2f' % (
    oracle_v['out_ret'], oracle_v['out_fgt'], oracle_v['ncc_ret'], oracle_v['ncc_fgt']))
print()

for m in METHODS:
    label = METHOD_LABELS[m]
    print(f'======= {label} =======')
    print(f'{"Variant":<30} {"out_ret":>8} {"out_fgt":>8} {"ncc_ret":>8} {"ncc_fgt":>8}')
    print('-'*66)
    for vname, vdata, is_cmf in VARIANTS:
        if m not in vdata:
            continue
        d = vdata[m]
        print(f'{vname:<30} {d["out_ret"]:>8.2f} {d["out_fgt"]:>8.2f} {d["ncc_ret"]:>8.2f} {d["ncc_fgt"]:>8.2f}')
    print()
