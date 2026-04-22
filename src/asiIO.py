import numpy as np


def save_asi(asi_dict, filename="active_set.asi"):
    with open(filename, "w") as f:
        for atom_type, A_inv in asi_dict.items():
            n, m = A_inv.shape
            f.write(f"{atom_type} {n} {m}\n")
            for val in A_inv.ravel(order="C"):
                f.write(f"{val}\n")


def load_asi(asi):
    ret = {}
    with open(asi, "r") as f:
        while True:
            B = []
            line1 = f.readline()
            if len(line1) == 0:
                break
            line1 = line1.split(" ")
            element, shape1, shape2 = int(line1[0]), int(line1[1]), int(line1[2])
            for _ in range(shape1 * shape2):
                B.append(float(f.readline()))
            ret[element] = np.array(B).reshape((shape1, shape2))
    return ret