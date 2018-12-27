import numpy as np


def next_x(x):
    x += np.random.normal(scale=.0625)
    if x < 0:
        return 0.
    return xnext


def walk():
    iters = 0
    while x <= 1.:
        x = next_x(x)
        iters += 1
    return nr_iters


walks = np.array([walk() for i in range(1000)])

print(np.mean(walks))
