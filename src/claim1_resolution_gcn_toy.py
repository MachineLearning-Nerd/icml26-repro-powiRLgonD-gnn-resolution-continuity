#!/usr/bin/env python3
"""Clean-room local CPU resolution-shift fixture for live Claim 1.

This is deliberately a reduced synthetic graph-regression experiment, not QM7.
It tests the paper's qualitative discontinuity mechanism: a readout trained on
fine graph GCN embeddings loses accuracy when evaluated on a quotient graph.
"""
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np


def gcn_embedding(x: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """One fixed renormalized-adjacency GCN layer + mean readout embedding."""
    a = adj + np.eye(adj.shape[0])
    d = a.sum(axis=1)
    norm = a / np.sqrt(d[:, None] * d[None, :])
    h = np.maximum(norm @ x, 0.0)
    return h.mean(axis=0)


def ring_adj(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        a[i, (i - 1) % n] = a[i, (i + 1) % n] = 1.0
    return a


def coarsen_pairs(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, 2, x.shape[1]).mean(axis=1)


def fixture(seed: int, train_n=300, test_n=200, fine_n=16):
    rng = np.random.default_rng(seed)
    af, ac = ring_adj(fine_n), ring_adj(fine_n // 2)
    # Latent target is scale invariant.  Fine node signal has alternating
    # high-frequency detail; quotient averaging removes it before GCN passes.
    def sample(count):
        z = rng.uniform(-1.5, 1.5, size=count)
        phase = rng.uniform(0, 2*np.pi, size=count)
        xs, ys = [], []
        nodes = np.arange(fine_n)
        for zi, pi in zip(z, phase):
            x0 = zi + 0.85*np.sin(2*np.pi*nodes/fine_n + pi) + 0.35*((-1.0)**nodes)
            x1 = np.cos(2*np.pi*nodes/fine_n + pi)
            xs.append(np.column_stack([x0, x1]))
            ys.append(2.0*zi + 0.2*np.sin(pi))
        return np.asarray(xs), np.asarray(ys)
    xtr, ytr = sample(train_n)
    xte, yte = sample(test_n)
    ef = np.array([gcn_embedding(x, af) for x in xtr])
    # affine linear regression readout learned only at fine resolution
    design = np.c_[np.ones(train_n), ef]
    w = np.linalg.lstsq(design, ytr, rcond=None)[0]
    ef_test = np.array([gcn_embedding(x, af) for x in xte])
    ec_test = np.array([gcn_embedding(coarsen_pairs(x), ac) for x in xte])
    pf, pc = np.c_[np.ones(test_n), ef_test] @ w, np.c_[np.ones(test_n), ec_test] @ w
    # A destructive matched control: preserve fine inputs but permute graph
    # signal positions for cross evaluation, expected to be even less aligned.
    perm = np.roll(np.arange(fine_n), 1)
    ep = np.array([gcn_embedding(coarsen_pairs(x[perm]), ac) for x in xte])
    pp = np.c_[np.ones(test_n), ep] @ w
    return {
        "seed": seed, "same_resolution_mae": float(np.mean(np.abs(pf-yte))),
        "cross_resolution_mae": float(np.mean(np.abs(pc-yte))),
        "permuted_coarse_mae": float(np.mean(np.abs(pp-yte))),
        "embedding_l2_mean": float(np.mean(np.linalg.norm(ef_test-ec_test, axis=1))),
        "fine_nodes": fine_n, "coarse_nodes": fine_n//2, "train_n": train_n, "test_n": test_n,
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',nargs='+',type=int,default=[101,202,303,404,505]); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); start=time.time()
    rows=[fixture(s) for s in args.seeds]
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={"verdict":"toy","scope":"clean-room synthetic ring-graph regression; reduced fixture, not QM7/Table 1", "seeds":args.seeds,
      "same_resolution_mae_mean":float(np.mean([r['same_resolution_mae'] for r in rows])),
      "cross_resolution_mae_mean":float(np.mean([r['cross_resolution_mae'] for r in rows])),
      "permuted_coarse_mae_mean":float(np.mean([r['permuted_coarse_mae'] for r in rows])),
      "cross_exceeds_same_all_seeds":bool(all(r['cross_resolution_mae']>r['same_resolution_mae'] for r in rows)),
      "runtime_seconds":time.time()-start,"python":sys.version,"platform":platform.platform()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    config={"seeds":args.seeds,"fine_nodes":16,"coarse_nodes":8,"train_n":300,"test_n":200,"method":"one fixed renormalized-adjacency GCN layer plus learned fine-resolution linear readout","control":"node-position permuted coarse input"}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest=[]
    for name in ['results.csv','summary.json','config.json']:
        manifest.append(f"{sha(out/name)}  {name}")
    (out/'SHA256SUMS').write_text('\n'.join(manifest)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
