from unittest import TestCase
from functions import *


class FunctionsTest(TestCase):
	def test_load_image_from_disk(self):
		file_name = "TestImages/theo.jpg"
		image_shape = (800, 676, 3)
		image = load_image_from_disk(file_name)
		self.assertEqual(len(image.shape), 3, "loaded image is the wrong number of dimensions")
		self.assertTupleEqual(image.shape, image_shape, "loaded image is the wrong shape")
		show_image(image)

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
