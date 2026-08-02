import importlib.util
from pathlib import Path
p=Path(__file__).parents[1]/'src'/'claim1_resolution_gcn_toy.py'
spec=importlib.util.spec_from_file_location('claim1',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_coarsening_shape():
    import numpy as np
    assert m.coarsen_pairs(np.zeros((16,2))).shape == (8,2)

def test_cross_resolution_shift_fixture():
    r=m.fixture(101, train_n=80, test_n=60)
    assert r['cross_resolution_mae'] > r['same_resolution_mae']
    assert r['embedding_l2_mean'] > 0
