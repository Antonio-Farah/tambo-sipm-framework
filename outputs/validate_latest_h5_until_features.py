from pathlib import Path

import nbformat
from nbclient import NotebookClient


source = Path("notebooks/04_latest_h5_real_vs_sim_peak_matching.ipynb")
nb = nbformat.read(source, as_version=4)

# Execute through the feature-extraction cell only. This catches the failure
# reported by the user without running the expensive CORSIKA simulation cells.
nb.cells = nb.cells[:12]

client = NotebookClient(
    nb,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": str(source.parent)}},
)
client.execute()

out = Path("outputs/latest_h5_until_features_validation.ipynb")
nbformat.write(nb, out)
print(f"validated through feature extraction: {out}")
