import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from constants import INFINITY
from typing import Union
from kldiv import data_to_probability_distribution, kl_divergence_symmetric


def dissimilarity_score(combo_patch: np.ndarray) -> float:
	"""
	takes a pair of square patches and gives a score indicating how dissimilar the patches are at their touching edges
	:param combo_patch: a numpy array of shape (x, 2x, 3) representing the 2 patches placed next to each other
	:return: a float in range [0.0, 255.0] (inclusive) representing how dissimilar the patches are at their touching edges. high values mean dissimilar; low values mean similar
	"""
	diffs = list()
	middle_seam_index1 = (combo_patch.shape[1] // 2) - 1
	middle_seam_index2 = combo_patch.shape[1] // 2

	left_deep = combo_patch[:, middle_seam_index1 - 1, :]
	left_shallow = combo_patch[:, middle_seam_index1, :]
	right_shallow = combo_patch[:, middle_seam_index2, :]
	right_deep = combo_patch[:, middle_seam_index2 + 1, :]
	# left deep, right shallow
	diff_array = abs(left_deep - right_shallow)
	diffs += list(diff_array.flatten())
	# both shallow
	diff_array = abs(left_shallow - right_shallow)
	diffs += list(diff_array.flatten())
	# left shallow, right deep
	diff_array = abs(left_shallow - right_deep)
	diffs += list(diff_array.flatten())
	# average list at the end
	return sum(diffs) / len(diffs)


def boring_score(combo_patch: np.ndarray) -> float:
	"""
	takes a pair of square patches and gives a score indicating how boring the patches are at their touching edges. pixel value comparisons are only evaluated parallel to the seam
	:param combo_patch: a numpy array of shape (x, 2x, 3) representing the 2 patches placed next to each other
	:return: a float in range [0.0, 255.0] (inclusive) representing how boring the patches are at their touching edges. high values mean boring; low values mean interesting
	"""
	# patch_size = combo_patch.shape[0]
	diffs = list()
	middle_seam_index1 = (combo_patch.shape[1] // 2) - 1
	middle_seam_index2 = combo_patch.shape[1] // 2
	for seam_index in [middle_seam_index1 - 1, middle_seam_index1, middle_seam_index2, middle_seam_index2 + 1]:
		for vertical_shift in range(1, 5):
			diff_array = abs(combo_patch[:(-1 * vertical_shift), seam_index, :] - combo_patch[vertical_shift:, seam_index, :])
			diffs += list(diff_array.flatten())
	# average list at the end
	score = 255 - (sum(diffs) / len(diffs))
	return score


def kl_score(combo_patch: np.ndarray) -> float:
	"""
	takes a pair of square patches and gives a score, based on KL divergence, representing how different their overall color distributions are
	:param combo_patch: a numpy array of shape (x, 2x, 3) representing the 2 patches placed next to each other
	:return: a float representing how different the overall color distributions are. high values mean dissimilar; low values mean similar
	"""
	seam_index = combo_patch.shape[1] // 2
	left = combo_patch[:, :seam_index, :]
	right = combo_patch[:, seam_index:, :]
	distributions_left = [data_to_probability_distribution(left[:, :, c], -0.5, 255.5, 64) for c in range(left.shape[2])]
	distributions_right = [data_to_probability_distribution(right[:, :, c], -0.5, 255.5, 64) for c in range(left.shape[2])]
	kl_values = [kl_divergence_symmetric(d_left, d_right) for d_left, d_right in zip(distributions_left, distributions_right)]
	return sum(kl_values)


def load_image_from_disk(image_file_name: str) -> np.ndarray:
	"""
	takes a path to an image file and loads it from disk
	:param image_file_name: a string indicating which image file to load
	:return: a numpy array of the image with shape (r, c, 3)
	"""
	badly_oriented_image = cv2.imread(image_file_name)
	return np.fliplr(badly_oriented_image.reshape(-1, 3)).reshape(badly_oriented_image.shape)


def show_image(image_rgb: np.ndarray, label: str = None):
	"""
	shows an rgb image
	:param image_rgb: the image to show as a numpy array with shape (r, c, 3)
	:param label: a label to apply to the image
	:return: None
	"""
	plt.style.use("dark_background")
	plt.imshow(image_rgb, vmin=0, vmax=255)
	plt.axis("off")
	if label is not None:
		plt.title(label)
	plt.show()


def assemble_patches(patches: list, num_cols: int) -> Union[None, np.ndarray]:
	"""
	takes a list of square patches and returns a tiled image of the patches with the specified number of columns
	:param patches: a list of numpy arrays each of shape (x, x, 3), being the patches to assemble
	:param num_cols: number of patches for each row to contain. a value of -1 means that all patches are placed on one row
	:return: a numpy array of shape (r, c, 3) that is the tiled image of all the patches
	"""
	n = len(patches)
	if n == 0:
		return None
	patch_size = patches[0].shape[0]  # assumes all are the same size and square
	if num_cols == -1:
		num_rows = 1
		num_cols = n
	else:
		num_rows = n // num_cols
	if n % num_cols != 0:
		num_rows += 1
	# tile the patches
	assembled = np.zeros((num_rows * patch_size, num_cols * patch_size, 3), dtype=patches[0].dtype)
	for i, patch in enumerate(patches):
		row = i // num_cols
		col = i % num_cols
		start_pixel_row = row * patch_size
		start_pixel_col = col * patch_size
		assembled[start_pixel_row:(start_pixel_row + patch_size), start_pixel_col:(start_pixel_col + patch_size), :] = patch
	return assembled


def coord_rot90(row: int, col: int, m: int, n: int, r: int = 1) -> (int, int):
	"""
	takes a row index and column index of an element in a 2D array of specified shape and calculates the ending row and column indices ofter the array has been rotated
	:param row: the row index of the element's starting location
	:param col: the column index of the element's starting location
	:param m: the number of rows in the array being rotated
	:param n: the number of columns in the array being rotated
	:param r: the number of times the array is rotated 90 degrees counter-clockwise (default value is 1)
	:return: a tuple of length 2 giving the row and column indices of the element's ending location in the rotated array
	"""
	r %= 4
	if r == 0:
		return row, col
	if r == 1:
		return n - (col + 1), row
	if r == 2:
		return m - (row + 1), n - (col + 1)
	if r == 3:
		return col, m - (row + 1)


def block_rot90(block: np.ndarray, r: int) -> np.ndarray:
	"""
	takes a subsection of a reconstruction matrix and calculates what it would be after rotation
	:param block: a subsection of a reconstruction matrix as a numpy array of shape (r, c, 2)
	:param r: the number of times `block` is rotated 90 degrees counter-clockwise
	:return: the rotated version of `block` after being rotated
	"""
	r %= 4
	block = np.rot90(block, r)
	block[:, :, 1] += r
	block[:, :, 1] %= 4
	return block


def rotate_random(patch: np.ndarray) -> (np.ndarray, int):
	"""
	takes an image patch and returns it at a random rotation
	:param patch: the image to rotate as a numpy array of shape (r, c, 3)
	:return: the rotated image as a numpy array
	"""
	rotation_index = random.randrange(4)
	new_patch = np.rot90(patch, rotation_index)
	return new_patch, rotation_index


def rotations_from_combination_index(combination_index: int) -> (int, int):
	"""
	takes a combination index and gives two rotation indices
	:param combination_index: an int in range [0,15] inclusive, indicating how the two images should be combined
	:return: a pair of rotation indices, each indicating how far the respective image has been rotated from the other
	"""
	if combination_index >= 16 or combination_index < 0:
		return None, None
	rotation1 = combination_index // 4
	rotation2 = combination_index % 4
	return rotation1, rotation2


def combination_index_from_rotations(rotation1: int, rotation2: int) -> int:
	"""
	takes two rotation indices and gives a combination index
	:param rotation1: an int indicating how far the left/first image is rotated counter-clockwise in 90-degree units
	:param rotation2: an int indicating how far the right/second image is rotated counter-clockwise in 90-degree units
	:return: an int in range [0,15] inclusive, indicating how the two images should be combined
	"""
	return ((4 * rotation1) + rotation2) % 16


def combine_patches(patch1: np.ndarray, patch2: np.ndarray, combination_index: int = 0) -> np.ndarray:
	"""
	takes two image patches and
	:param patch1: the left/first patch to be combined with shape (x, x, 3)
	:param patch2: the right/second patch to be combined with shape (x, x, 3)
	:param combination_index: an int in range [0,15] inclusive, indicating how the two images should be combined
	:return: a numpy array of shape (x, 2x, 3) representing the two patches combined at appropriate rotations
	"""
	r1, r2 = rotations_from_combination_index(combination_index)
	if r1 is None or r2 is None:
		print("invalid combination index; cannot be combined")
		return patch1
	combined = np.empty((patch1.shape[0], patch1.shape[1] + patch2.shape[1], patch1.shape[2]), dtype=patch1.dtype)
	combined[:, :patch1.shape[1]] = np.rot90(patch1, (4 - r1) % 4)
	combined[:, patch1.shape[1]:] = np.rot90(patch2, (4 - r2) % 4)
	return combined


def verify_reconstruction_matrix(matrix: np.ndarray, n: int) -> bool:
	"""
	takes a reconstruction matrix and verifies and a number of patches supposedly contained in it, and verifies that each piece is found and no extras
	:param matrix: the reconstruction matrix to verify, as a numpy array of shape (x, y, 2)
	:param n: the number of patches supposedly contained in the reconstruction matrix
	:return: `True` if the reconstruction matrix contains the needed patch indices on only those, `False` if there are any missing or any extras
	"""
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


def combo_score_mp(coord_and_patches: tuple) -> (tuple, float):
	(i, j, c), patch1, patch2, functions = coord_and_patches
	combined = combine_patches(patch1, patch2, c).astype(int)
	if i == j:
		score = INFINITY
	else:
		score = 0
		for f in functions:
			score += f(combined)
	return (i, j, c), score


class PatchPairGenerator:
	def __init__(self, patches: list, functions: list):
		self.patches = patches
		self.functions = functions

	def __iter__(self):
		for i, patch1 in enumerate(self.patches):
			for j, patch2 in enumerate(self.patches):
				for c in range(16):
					yield (i, j, c), patch1, patch2, self.functions

	def __len__(self):
		n = len(self.patches)
		return n * n * 16


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
