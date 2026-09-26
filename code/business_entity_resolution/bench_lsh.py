import time
import numpy as np
from datasketch import MinHash, MinHashLSH
from scipy.sparse import csr_matrix

def bench_lsh(n_rows):
    lsh = MinHashLSH(threshold=0.5, num_perm=64)
    t0 = time.time()
    for i in range(n_rows):
        m = MinHash(num_perm=64)
        m.update(b"abc")
        m.update(b"def")
        lsh.insert(str(i), m)
    print(f"LSH insert {n_rows} rows: {time.time()-t0:.2f}s")

def bench_sparse(n_pairs):
    X1 = csr_matrix((100000, 50000), dtype=np.float32)
    X2 = csr_matrix((100000, 50000), dtype=np.float32)
    # create some fake pairs
    s1_idx = np.random.randint(0, 100000, n_pairs)
    s23_idx = np.random.randint(0, 100000, n_pairs)
    t0 = time.time()
    X1_sub = X1[s1_idx]
    X2_sub = X2[s23_idx]
    scores = X1_sub.multiply(X2_sub).sum(axis=1).A1
    print(f"Sparse multiply {n_pairs} pairs: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    bench_lsh(250000)
    bench_sparse(1000000)
