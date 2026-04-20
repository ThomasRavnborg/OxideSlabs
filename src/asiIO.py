import numpy as np

def save_asi(A_inv, filename="active_set.asi"):
    """
    Save active set inverse matrix (M x M)
    """
    A_inv = np.asarray(A_inv)

    with open(filename, "w") as f:
        n, m = A_inv.shape
        f.write(f"{n} {m}\n")

        for val in A_inv.ravel():
            f.write(f"{val}\n")

def load_asi(filename):
    """
    Load active set inverse matrix (M x M)
    """
    with open(filename, "r") as f:
        n, m = map(int, f.readline().split())

        data = [
            float(f.readline())
            for _ in range(n * m)
        ]

    return np.array(data).reshape(n, m)