from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import random
from functions import assemble_patches, get_test_patches, load_image_from_disk, show_image, verify_reconstruction_matrix
from kruskal import assemble_image, build_graph, kruskals_reconstruction
import time


class KruskalsTest(TestCase):
	def test_build_graph(self):
		test_patches = get_test_patches()
		show_image(assemble_patches(test_patches, 2))
		matrix = build_graph(test_patches)
		self.assertTupleEqual(matrix.shape, (4, 4, 16), "matrix wrong shape")
		# show_image(combine_patches(test_patches[0], test_patches[1], 0))
		self.assertGreater(matrix[0, 1, 0], matrix[0, 1, 2], "black with rotated red should score better than black with just red")
		self.assertGreater(matrix[0, 1, 2], matrix[0, 1, 10], "rotated black with rotated red should score better than just black with rotated red")
		self.assertEqual(matrix[0, 1, 2], matrix[1, 0, 2], "black with red should get the same score regardless of which is listed first")
		self.assertLess(matrix[2, 3, 7], matrix[1, 3, 11], "similar shades should have a better score then blatantly different colors")
		# self.assertLess(matrix[2, 3, 13], matrix[3, 2, 0], "identical interesting edges should give a lower score than identical boring edges")
		self.assertEqual(np.amin(matrix), matrix[2, 3, 13], "identical and interesting edges should give the lowest score")

	def test_jigsaw_kruskals(self):
		domain = list()
		averages = list()
		for n in range(10, 30):
			domain.append(n)
			times = list()
			random.seed(4)
			for trial in range(10):
				test_graph = np.array([[[random.randrange(101) for _ in range(16)] for _ in range(n)] for _ in range(n)], dtype=float)
				time_start = time.time()
				reconstruction_matrix = kruskals_reconstruction(test_graph)
				times.append(time.time() - time_start)
				self.assertTrue(verify_reconstruction_matrix(reconstruction_matrix, n), "reconstruction matrix is invalid")
			average = sum(times) / len(times)
			averages.append(average)
			space = "\t"
			if len(str(n)) == 1:
				space += "\t"
			print(f"size: {n}{space}average time: {average}")
		plt.plot(domain, averages)
		plt.title("time to run jigsaw_kruskals by input size")
		plt.show()

	def test_assemble_image(self):
		original = load_image_from_disk("TestImages/TestPatches.png")
		test_patches = get_test_patches()
		# cycle black, red, and dark blue clockwise
		# rotate black by 180, rotate red by 90
		test_patches_shuffled = [test_patches[2], np.rot90(test_patches[0], 2), np.rot90(test_patches[1], 1), test_patches[3]]
		# make the reconstruction matrix
		reconstruction = np.array([[[1, 2], [2, 3]], [[0, 0], [3, 0]]])
		actual = assemble_image(test_patches_shuffled, reconstruction)
		show_image(original)
		show_image(assemble_patches(test_patches_shuffled, 2))
		show_image(actual)
		self.assertTrue(np.array_equal(original, actual), "reconstructed image is not the same as the original")
