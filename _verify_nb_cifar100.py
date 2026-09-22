import json

for nb in ['04_cmf_static', '04b_cmf_posthoc', '04d_cmf_freeze_w_stage3', '04e_cmf_freeze_w_stage3_static']:
    path = f'notebooks/{nb}.ipynb'
    data = json.load(open(path))
    all_lines = []
    for c in data['cells']:
        src = c['source']
        if isinstance(src, list):
            all_lines.extend(src)
        else:
            all_lines.append(src)

    forget  = [l.strip() for l in all_lines if 'FORGET_CLASSES' in l and '=' in l and '#' not in l[:3]]
    dataset = [l.strip() for l in all_lines if 'CIFAR10' in l or 'CIFAR100' in l]
    fallback_dataset = [l.strip() for l in all_lines if "DATASET     = 'cifar" in l]
    fallback_num     = [l.strip() for l in all_lines if "NUM_CLASSES = " in l and "'cifar" not in l]

    print(f'\n=== {nb} ===')
    print(f'  FORGET_CLASSES : {forget[0][:70] if forget else "NOT FOUND"}')
    print(f'  Fallback DATASET   : {fallback_dataset[0][:60] if fallback_dataset else "NOT FOUND"}')
    print(f'  Fallback NUM_CLASS : {fallback_num[0][:60] if fallback_num else "NOT FOUND"}')
    print(f'  CIFAR100 loaders   : {sum(1 for l in dataset if "CIFAR100" in l)} found')
    print(f'  CIFAR10  loaders   : {sum(1 for l in dataset if "CIFAR10(" in l or "CIFAR10(" in l)} found (should be 0)')
