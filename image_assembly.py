import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


# enum-like values to indicate how far the patch has been rotated counter clockwise
ROTATION_0 = 0
ROTATION_90 = 1
ROTATION_180 = 2
ROTATION_270 = 3


def load_image_from_disk(image_file_name: str) -> np.ndarray:
	"""
	takes a path to an image file and loads it from disk
	:param image_file_name: a string indicating which image file to load
	:return: a numpy array of the image with shape (r, c, 3)
	"""
	badly_oriented_image = cv2.imread(image_file_name)
	return np.fliplr(badly_oriented_image.reshape(-1, 3)).reshape(badly_oriented_image.shape)


def show_image(image_rgb: np.ndarray):
	"""
	shows an rgb image
	:param image_rgb: the image to show as a numpy array with shape (r, c, 3)
	:return: None
	"""
	plt.imshow(image_rgb)
	plt.show()


def rotate_random(patch: np.ndarray) -> (np.ndarray, int):
	"""
	takes an image patch and returns it at a random rotation
	:param patch: the image to rotate as a numpy array of shape (r, c, 3)
	:return: the rotated image as a numpy array
	"""
	rotation_index = random.randrange(4)
	new_patch = np.rot90(patch, rotation_index)
	return new_patch, rotation_index


def combination_index_from_rotations(rotation1: int, rotation2: int) -> int:
	"""
	takes two rotation indices and gives a combination index
	:param rotation1: an int indicating how far the left/first image is rotated counter-clockwise in 90-degree units
	:param rotation2: an int indicating how far the right/second image is rotated counter-clockwise in 90-degree units
	:return: an int in range [0,15] inclusive, indicating how the two images should be combined
	"""
	return ((4 * rotation1) + rotation2) % 16


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


def combine_patches(patch1: np.ndarray, patch2: np.ndarray, combination_index: int) -> np.ndarray:
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


def scramble_image(image: np.ndarray, patch_size: int) -> list:
	"""
	takes an rgb image and scrambles it into square patches, each at a random rotation
	:param image: numpy array of shape (r, c, 3) where r is the number of rows and c is the number of columns
	:param patch_size: number indicating how many pixels wide and tall each patch should be
	:return: a list containing the scrambled patches, each of shape (patch_size, patch_size, 3)
	"""
	# TODO
	pass


def build_graph(patches: list) -> np.ndarray:
	"""
	takes a list of scrambled patches and creates an adjacency matrix that gives
	:param patches: list of square numpy arrays, each of the same shape (x, x, 3), length n
	:return: a numpy array of shape (n, n, 16). the value at [i, j, c] indicates how dissimilar patch i and patch j are
	along their shared edge when using combination index c
	"""
	# TODO
	pass


def jigsaw_kruskals(graph: np.ndarray) -> np.ndarray:
	"""
	takes an adjacency matrix and uses kruskal's algorithm to build a reconstruction matrix
	:param graph: a numpy array with shape (n, n, 16) giving the dissimilarity scores for the n patches
	:return: a numpy array of shape (r, c, 2). the value at [i, j, 0] gives the index of the patch that should be located
	in that slot, and [i, j, 1] gives the rotation index of that patch
	"""
	# TODO
	pass


def assemble_image(patches: list, construction_matrix: np.ndarray) -> np.ndarray:
	"""
	takes a list of patches and a reconstruction matrix to assemble the patches
	:param patches: a list of numpy arrays representing the scrambled patches of the original image
	:param construction_matrix: a numpy array of shape (r, c, 2). the value at [i, j, 0] gives the index of the patch that
	should be located in that slot, and [i, j, 1] gives the rotation index of that patch
	:return: the re-assembled image as a numpy array of shape (x, y, 3)
	"""
	# TODO
	pass


if __name__ == "__main__":
	original_image = load_image_from_disk("path/to/image.png")
	show_image(original_image)
	patch_list = scramble_image(original_image, 28)
	adjacency_matrix = build_graph(patch_list)
	reconstruction_matrix = jigsaw_kruskals(adjacency_matrix)
	reconstructed_image = assemble_image(patch_list, reconstruction_matrix)
	show_image(reconstructed_image)
	pass
