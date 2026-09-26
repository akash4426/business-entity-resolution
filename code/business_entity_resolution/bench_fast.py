import time
from rapidfuzz import fuzz
import concurrent.futures

a = ["Apple Inc"] * 1000000
b = ["Apple Incorporated"] * 1000000

t0 = time.time()
res1 = [fuzz.ratio(x, y) for x, y in zip(a, b)]
print("List comp:", time.time() - t0)
