import time
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import random
from image_assembly import assemble_image, build_graph, jigsaw_kruskals, load_image_from_disk, scramble_image, show_image, assemble_patches


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


def get_test_patches() -> list:
	all_test_patches = load_image_from_disk("TestImages/TestPatches.png")
	patch1 = all_test_patches[:32, :32, :]
	patch2 = all_test_patches[:32, 32:, :]
	patch3 = all_test_patches[32:, :32, :]
	patch4 = all_test_patches[32:, 32:, :]
	return [patch1, patch2, patch3, patch4]


file_name = "TestImages/theo.jpg"
image_shape = (800, 676, 3)


class KruskalsTest(TestCase):
	def test_jigsaw_kruskals(self):
		domain = list()
		averages = list()
		for n in range(10, 30):
			domain.append(n)
			times = list()
			for trial in range(10):
				test_graph = np.array([[[random.randrange(101) for r in range(16)] for b in range(n)] for a in range(n)], dtype=float)
				time_start = time.time()
				reconstruction_matrix = jigsaw_kruskals(test_graph)
				times.append(time.time() - time_start)
				self.assertTrue(verify_reconstruction_matrix(reconstruction_matrix, n), "reconstruction matrix is invalid")
			average = sum(times) / len(times)
			averages.append(average)
			space = "\t"
			if len(str(n)) == 1:
				space += "\t"
			print(f"size: {n}{space}average time: {average}")
		plt.plot(domain, averages)
		plt.show()

	def test_load_image_from_disk(self):
		image = load_image_from_disk(file_name)
		self.assertEqual(len(image.shape), 3, "loaded image is the wrong number of dimensions")
		self.assertTupleEqual(image.shape, image_shape, "loaded image is the wrong shape")
		show_image(image)

	def test_scramble_image(self):
		image = load_image_from_disk(file_name)
		patch_size = 100
		for trial in range(3):
			patches = scramble_image(image, patch_size)
			patch_row_count = image_shape[0] // patch_size
			patch_col_count = image_shape[1] // patch_size
			n = patch_row_count * patch_col_count
			self.assertEqual(len(patches), n, "wrong number of patches")
			show_image(assemble_patches(patches, patch_col_count))

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
		self.assertEqual(0, matrix[2, 3, 13], "identical edges should give a score of 0")

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
