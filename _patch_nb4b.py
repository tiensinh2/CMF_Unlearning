import json

with open('notebooks/04b_cmf_posthoc.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = {'config': False, 's1': False}

for cell in nb['cells']:
    src_list = cell.get('source', [])
    if isinstance(src_list, list):
        src = ''.join(src_list)
    else:
        src = src_list

    # ── Config cell: add NB4A_DATASET_DIR ──────────────────────────────────
    if 'CKPT_ROOT_NB4A' in src and 'NB4A_DATASET_DIR' not in src:
        old = "CKPT_ROOT_NB4A = '/kaggle/working/checkpoints/cmf_paper'"
        new = (
            "# NB4a checkpoints: first try a Kaggle dataset input (attach NB4a output as\n"
            "# a dataset to this notebook, then update the slug in NB4A_DATASET_DIR below).\n"
            "NB4A_DATASET_DIR = '/kaggle/input/datasets/btk23021592/cmf-notebook4a'\n"
            "CKPT_ROOT_NB4A = '/kaggle/working/checkpoints/cmf_paper'"
        )
        src = src.replace(old, new, 1)
        # Add existence print
        old2 = "print(f'STAGE={STAGE}"
        new2 = "print(f'NB4A_DATASET_DIR exists: {os.path.isdir(NB4A_DATASET_DIR)}')\n" + old2
        src = src.replace(old2, new2, 1)
        if isinstance(src_list, list):
            cell['source'] = src.splitlines(keepends=True)
        else:
            cell['source'] = src
        patched['config'] = True

    # ── Loop cell: prepend dataset-input paths to s1_candidates ────────────
    if 's1_candidates' in src and 'NB4A_DATASET_DIR' not in src:
        old = (
            "s1_candidates = [\n"
            "                    f'{CKPT_ROOT_NB4A}/cmf_static/{s1_tag}.pt',\n"
            "                    f'./notebooks/Result_nb4/checkpoints/{s1_tag}.pt',\n"
            "                    f'./checkpoints/cmf_paper/cmf_static/{s1_tag}.pt',\n"
            "                ]"
        )
        new = (
            "s1_candidates = [\n"
            "                    f'{NB4A_DATASET_DIR}/cmf_static/{s1_tag}.pt',\n"
            "                    f'{NB4A_DATASET_DIR}/checkpoints/cmf_paper/cmf_static/{s1_tag}.pt',\n"
            "                    f'{CKPT_ROOT_NB4A}/cmf_static/{s1_tag}.pt',\n"
            "                    f'./notebooks/Result_nb4/checkpoints/{s1_tag}.pt',\n"
            "                    f'./checkpoints/cmf_paper/cmf_static/{s1_tag}.pt',\n"
            "                ]"
        )
        if old in src:
            src = src.replace(old, new, 1)
            if isinstance(src_list, list):
                cell['source'] = src.splitlines(keepends=True)
            else:
                cell['source'] = src
            patched['s1'] = True
        else:
            print("WARNING: s1_candidates old pattern not found exactly. Showing context:")
            idx = src.find('s1_candidates')
            print(repr(src[idx:idx+400]))

with open('notebooks/04b_cmf_posthoc.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Patched:", patched)
print("Saved notebooks/04b_cmf_posthoc.ipynb")
