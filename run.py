#!/usr/bin/env python
"""Random clustering baseline for Omnibenchmark."""
import argparse, json, os, time
import numpy as np
import pandas as pd
import anndata as ad

# Parse command line arguments (same names as the scanpy module)
parser = argparse.ArgumentParser(description="Random clustering baseline")
parser.add_argument(
    "--output_dir", "-o", type=str, required=True, help="Output directory"
)
parser.add_argument("--name", "-n", type=str, required=True, help="Dataset name")
parser.add_argument(
    "--data.h5ad", dest="data_h5ad", type=str, required=True, help="Input h5ad file"
)
parser.add_argument(
    "--data.clusters_truth_num",
    dest="clusters_truth_num_path",
    type=str,
    required=True,
    help="input true number of clusters path",
)
parser.add_argument(
    "--d_cluster",
    type=int,
    default=0,
    help="delta number of clusters with respect to true number of clusters",
)
parser.add_argument(
    "--seed", type=int, default=42, help="seed for the random number generator"
)
args, _ = parser.parse_known_args()

# Random clustering
t0 = time.time()
cells = ad.read_h5ad(args.data_h5ad).obs_names
with open(args.clusters_truth_num_path) as f:
    k_true = int(f.read().strip())
k = max(1, k_true + args.d_cluster)

rng = np.random.default_rng(args.seed)
labels = rng.integers(1, k + 1, size=len(cells))
t_cluster = time.time() - t0

os.makedirs(args.output_dir, exist_ok=True)
out = lambda suffix: os.path.join(args.output_dir, f"{args.name}.{suffix}")

# clusters.tsv: same columns as scanpy/OSCA
pd.DataFrame({"cell_id": cells, "louvain": labels, "leiden": labels}) \
  .to_csv(out("clusters.tsv"), sep="\t", index=False)

# pca.tsv: random coordinates, same layout (cell_id, PC1..PC30)
pca = pd.DataFrame(rng.normal(size=(len(cells), 30)), index=cells,
                   columns=[f"PC{i+1}" for i in range(30)])
pca.index.name = "cell_id"
pca.to_csv(out("pca.tsv"), sep="\t")

# timings.json: same keys as scanpy; steps a baseline doesn't do are 0
steps = ["find_mit_gene", "filter", "normalization", "hvg", "scaling",
         "pca", "t_sne", "umap"]
timings = {"gpu_load": None, **{s: 0.0 for s in steps},
           "louvain": t_cluster, "leiden": t_cluster}
with open(out("timings.json"), "w") as f:
    json.dump(timings, f, indent=2)

# clustering_info.json: same structure; no resolution search for random labels
info = {"resolutions": {"louvain": 0.0, "leiden": 0.0},
        "num_runs": {"louvain": 1, "leiden": 1}}
with open(out("clustering_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print(f"Random clustering: {len(cells)} cells, k={k}, seed={args.seed}")
