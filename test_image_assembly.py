import time
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import random
from image_assembly import jigsaw_kruskals


def verify_reconstruction_matrix(matrix: np.ndarray, n: int) -> bool:
    passes = True
    to_find = set(list(range(n)))
    flattened = list(matrix[:, :, 0].flatten())
    for i, val in enumerate(flattened):
        if val == -1:
            continue
        if val in to_find:
            to_find.remove(val)
            flattened[i] = -1
            continue
        else:
            print(f"{val} was not found in the reconstruction matrix")
            passes = False
    for val in flattened:
        if val != -1:
            print(f"found an unexpected number i the reconstruction matrix: {val}")
            passes = False
    return passes


class KruskalsTest(TestCase):
    def test_jigsaw_kruskals(self):
        domain = list()
        averages = list()
        for n in range(10, 50):
            domain.append(n)
            times = list()
            for trial in range(10):
                test_graph = np.array([[[random.randrange(101) for r in range(16)] for b in range(n)] for a in range(n)], dtype=float)
                time_start = time.time()
                reconstruction_matrix = jigsaw_kruskals(test_graph)
                times.append(time.time() - time_start)
                if not verify_reconstruction_matrix(reconstruction_matrix, n):
                    self.fail()
            average = sum(times) / len(times)
            averages.append(average)
            space = "\t"
            if len(str(n)) == 1:
                space += "\t"
            print(f"size: {n}{space}average time: {average}")
        plt.plot(domain, averages)
        plt.show()
