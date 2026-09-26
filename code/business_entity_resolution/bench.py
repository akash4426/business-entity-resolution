import time
from rapidfuzz import fuzz
import concurrent.futures

a = ["Apple Inc"] * 100000
b = ["Apple Incorporated"] * 100000

t0 = time.time()
res1 = [fuzz.token_set_ratio(x, y) for x, y in zip(a, b)]
print("List comp:", time.time() - t0)

t0 = time.time()
def compute(n1_n2):
    return fuzz.token_set_ratio(n1_n2[0], n1_n2[1])
with concurrent.futures.ThreadPoolExecutor() as executor:
    res2 = list(executor.map(compute, zip(a, b), chunksize=10000))
print("ThreadPool:", time.time() - t0)

t0 = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    res3 = list(executor.map(compute, zip(a, b), chunksize=10000))
print("ProcessPool:", time.time() - t0)
