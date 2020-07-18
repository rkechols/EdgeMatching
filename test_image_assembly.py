import time
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import random
from functions import load_image_from_disk
from image_assembly import assemble_image, boring_score, build_graph, combine_patches, jigsaw_kruskals, kl_score, scramble_image, show_image, \
	assemble_patches, verify_reconstruction_matrix


def get_test_patches() -> list:
	"""
	loads the test patches image from disk and breaks it into the 4 pieces
	:return: a list of 4 numpy arrays, each of shape (32, 32, 3)
	"""
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
			random.seed(4)
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
		plt.title("time to run jigsaw_kruskals by input size")
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
		# self.assertLess(matrix[2, 3, 13], matrix[3, 2, 0], "identical interesting edges should give a lower score than identical boring edges")
		self.assertEqual(np.amin(matrix), matrix[2, 3, 13], "identical and interesting edges should give the lowest score")

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

	def test_boring_score(self):
		test_patches = get_test_patches()
		double_plain = combine_patches(test_patches[0], test_patches[1])
		double_plain_score = boring_score(double_plain)
		# show_image(double_plain, f"double_plain: {double_plain_score}")
		hardly_interesting = combine_patches(test_patches[0], test_patches[1], 10)
		hardly_interesting_score = boring_score(hardly_interesting)
		# show_image(hardly_interesting, f"hardly_interesting: {hardly_interesting_score}")
		fairly_interesting = combine_patches(test_patches[2], test_patches[3], 13)
		fairly_interesting_score = boring_score(fairly_interesting)
		# show_image(fairly_interesting, f"fairly_interesting: {fairly_interesting_score}")
		self.assertLess(fairly_interesting_score, hardly_interesting_score, "a combination with fairly detailed edges should get a lower score than one with hardly detailed edges")
		self.assertLess(hardly_interesting_score, double_plain_score, "a combination with hardly detailed edges should get a lower score than one with boring/uniform edges")

	def test_boring_score_empirical(self):
		patch_sizes = [2 + (10 * i) for i in range(11)]
		min_scores = list()
		max_scores = list()
		for patch_size in patch_sizes:
			# hypothetical_min = 85 + (15.038 * math.log(patch_size))
			# hypothetical_max = 255 - (14.235 * math.log(patch_size))
			scores = list()
			for _ in range(10000):
				combo_patch = np.random.randint(0, 256, (patch_size, 2 * patch_size, 3))
				b_score = boring_score(combo_patch)
				# b_score = 100 * (b_score - hypothetical_min) / (hypothetical_max - hypothetical_min)
				scores.append(b_score)
			# find min and max
			min_scores.append(min(scores))
			max_scores.append(max(scores))
			# plot a histogram
			# plt.hist(scores, bins=50)
			# plt.title(f"patch size = {patch_size}")
			# plt.show()
		# plot the relations between patch size and min/max
		plt.plot(patch_sizes, min_scores)
		plt.title(f"min score by patch size")
		plt.show()
		plt.plot(patch_sizes, max_scores)
		plt.title(f"max score by patch size")
		plt.show()
		print("patch sizes:")
		print(patch_sizes)
		print("min scores:")
		print(min_scores)
		print("max scores:")
		print(max_scores)
		absolute_min = min(min_scores)
		absolute_max = max(max_scores)
		print(f"min = {absolute_min}")
		print(f"max = {absolute_max}")
		self.assertTrue(absolute_min >= 0, "a score should never go below 0")
		self.assertTrue(absolute_max < 256, "a score should never go over 255")

	def test_kl_score(self):
		test_patches = get_test_patches()
		red_black1 = combine_patches(test_patches[0], test_patches[1])
		red_black1_score = kl_score(red_black1)
		show_image(red_black1, f"red_black1: {red_black1_score}")
		red_black2 = combine_patches(test_patches[0], test_patches[1], 9)
		red_black2_score = kl_score(red_black2)
		show_image(red_black2, f"red_black2: {red_black2_score}")
		self.assertEqual(red_black1_score, red_black2_score, "kl scores should be rotation invariant")
		red_black3 = combine_patches(test_patches[1], test_patches[0], 9)
		red_black3_score = kl_score(red_black3)
		show_image(red_black3, f"red_black3: {red_black3_score}")
		self.assertEqual(red_black1_score, red_black3_score, "kl scores should be symmetric")
		blue_blue = combine_patches(test_patches[2], test_patches[3])
		blue_blue_score = kl_score(blue_blue)
		show_image(blue_blue, f"blue_blue: {blue_blue_score}")
		self.assertLess(blue_blue_score, red_black1_score, "a kl score for images of similar color makeup should be lower than that of images that are quite different")
