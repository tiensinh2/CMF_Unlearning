import pandas as pd
import numpy as np

df_new = pd.read_csv('notebooks/Result_nb4_static/results_4a_cmf_static_new.xlsx')
df_old = pd.read_csv('notebooks/Result_nb4_static/results_4a_cmf_static.xlsx')

# Paper Table 3 values (CMF, CIFAR-10, 1 class)
paper_cmf = {
    'random_label':        {'out_ret':94.27,'out_fgt':80.70,'lp_ret':94.19,'lp_fgt':85.71,'ncc_ret':94.25,'ncc_fgt':82.04},
    'salun':               {'out_ret':94.33,'out_fgt':78.07,'lp_ret':94.26,'lp_fgt':84.68,'ncc_ret':94.32,'ncc_fgt':79.50},
    'grad_ascent_descent': {'out_ret':91.87,'out_fgt':54.50,'lp_ret':92.35,'lp_fgt':68.02,'ncc_ret':91.99,'ncc_fgt':57.83},
    'scrub':               {'out_ret':92.51,'out_fgt':33.78,'lp_ret':92.48,'lp_fgt':60.68,'ncc_ret':92.48,'ncc_fgt':35.53},
}

# Paper Table 1 values (no CMF, CIFAR-10, 1 class)
paper_no_cmf = {
    'random_label': {'out_ret':92.93,'out_fgt':0.00,'lp_ret':92.65,'lp_fgt':92.49,'ncc_ret':92.25,'ncc_fgt':80.25},
    'salun':        {'out_ret':93.19,'out_fgt':0.00,'lp_ret':93.05,'lp_fgt':92.57,'ncc_ret':91.31,'ncc_fgt':93.70},
    'scrub':        {'out_ret':91.37,'out_fgt':0.00,'lp_ret':91.71,'lp_fgt':74.79,'ncc_ret':89.96,'ncc_fgt':55.30},
}

metrics = ['output_retain_acc','output_forget_acc','probe_retain_acc','probe_forget_acc','ncc_retain_acc','ncc_forget_acc']
mk_keys = ['out_ret','out_fgt','lp_ret','lp_fgt','ncc_ret','ncc_fgt']
label_m = {'grad_ascent_descent':'NegGrad+','random_label':'Random-label','salun':'SalUn','scrub':'SCRUB','tarun':'TARUN'}

print("="*90)
print("SUMMARY: Experiment (new) means vs Paper Table 3 (CMF, CIFAR-10, 1 class)")
print("="*90)
print(f"{'Method':<20} | {'Metric':<12} | {'Exp(new)':>9} | {'Exp(old)':>9} | {'Paper':>9} | {'new-paper':>10}")
print('-'*90)
for meth in ['grad_ascent_descent','random_label','salun','scrub']:
    sub_n = df_new[df_new['method']==meth]
    sub_o = df_old[df_old['method']==meth]
    en = sub_n[metrics].mean().values
    eo = sub_o[metrics].mean().values
    pv = [paper_cmf[meth][k] for k in mk_keys]
    for i, (e_n, e_o, p, k) in enumerate(zip(en, eo, pv, mk_keys)):
        diff = e_n - p
        print(f"{label_m[meth]:<20} | {k:<12} | {e_n:9.2f} | {e_o:9.2f} | {p:9.2f} | {diff:+10.2f}")
    print()

print()
print("="*90)
print("DIFF summary: new vs old (significant changes > 0.5%)")
print("="*90)
methods_all = ['grad_ascent_descent','random_label','salun','scrub','tarun']
for meth in methods_all:
    sub_n = df_new[df_new['method']==meth][metrics].mean()
    sub_o = df_old[df_old['method']==meth][metrics].mean()
    diff = sub_n - sub_o
    sigs = [(k, diff[c]) for k,c in zip(mk_keys,metrics) if abs(diff[c])>0.5]
    if sigs:
        print(f"  {label_m[meth]}: " + "  ".join(f"{k}:{d:+.2f}" for k,d in sigs))
    else:
        print(f"  {label_m[meth]}: no significant changes (all diffs < 0.5%)")
