"""Execute both notebooks in order using the current Python environment."""
import json
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / '.runtime'
kernel = runtime / 'kernels' / 'python3'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Python 3', 'language': 'python'}))
os.environ['JUPYTER_PATH'] = str(runtime)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
for name in ['01_eda.ipynb', '02_modeling.ipynb']:
    path = ROOT / 'notebooks' / name
    print('Executing', name, flush=True)
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=14400, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)
    print('Completed', name, flush=True)
