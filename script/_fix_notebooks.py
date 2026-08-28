"""
Fix the clone URL and sys.path in the 4 ReGUn notebooks so they
work when run from /kaggle/working (the notebook is in notebooks/ in the repo,
but the working directory is set to the repo root after cloning).
"""
import json, os, re

GITHUB_URL = 'https://github.com/tiensinh2/CMF_Unlearning.git'
REPO_DIR   = '/kaggle/working/CMF_Unlearning'

NOTEBOOKS = [
    'notebooks/01_train_full.ipynb',
    'notebooks/02_retrain_oracle.ipynb',
    'notebooks/03_unlearn_group_a_no_cmf.ipynb',
    'notebooks/04_unlearn_group_b_cmf.ipynb',
]

# The canonical clone cell source we want in every notebook
CLONE_CELL_SOURCE = [
    "REPO_DIR = '/kaggle/working/CMF_Unlearning'\n",
    "\n",
    "if not os.path.isdir(REPO_DIR):\n",
    "    sh(f'git clone " + GITHUB_URL + " {REPO_DIR}')\n",
    "else:\n",
    "    sh(f'git -C {REPO_DIR} pull origin main')\n",
    "\n",
    "os.chdir(REPO_DIR)\n",
    "sys.path.insert(0, REPO_DIR)\n",
    "print('Working directory:', os.getcwd())",
]

for nb_path in NOTEBOOKS:
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = cell['source']
        src_joined = ''.join(src)
        # Identify the clone cell by its signature content
        if "REPO_DIR = '/kaggle/working/CMF_Unlearning'" in src_joined and 'git clone' in src_joined:
            cell['source'] = CLONE_CELL_SOURCE
            print(f"  Fixed clone cell in {nb_path}")

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'Saved: {nb_path}')

print('\nAll notebooks updated.')
