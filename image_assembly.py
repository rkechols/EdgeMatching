import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


INFINITY = float("inf")

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


def assemble_patches(patches: list, num_cols: int) -> np.ndarray:
	n = len(patches)
	if n == 0:
		print("no patches to show")
		return
	patch_size = patches[0].shape[0]  # assumes all are the same size and square
	num_rows = n // num_cols
	if n % num_cols != 0:
		num_rows += 1
	# tile the patches
	assembled = np.zeros((num_rows * patch_size, num_cols * patch_size, 3), dtype=int)
	for i, patch in enumerate(patches):
		row = i // num_cols
		col = i % num_cols
		start_pixel_row = row * patch_size
		start_pixel_col = col * patch_size
		assembled[start_pixel_row:(start_pixel_row + patch_size), start_pixel_col:(start_pixel_col + patch_size), :] = patch
	return assembled


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


def combine_blocks(first_block: np.ndarray, second_block: np.ndarray, a: int, b: int, r: int) -> np.ndarray:
	# get the rotation values of each block
	r1, r2 = rotations_from_combination_index(r)
	first_block = np.rot90(first_block, (4 - r1) % 4)
	first_block[:, :, 1] = (first_block[:, :, 1] - r1) % 4
	second_block = np.rot90(second_block, (4 - r2) % 4)
	second_block[:, :, 1] = (second_block[:, :, 1] - r2) % 4
	# check that the two values are indeed in their blocks, and where
	a_locs = np.where(first_block[:, :, 0] == a)
	a_locs = list(zip(a_locs[0], a_locs[1]))
	if len(a_locs) != 1:
		raise RuntimeError("'a' could not be found (or was found multiple times) in the first block when combining blocks")
	a_loc = a_locs[0]
	b_locs = np.where(second_block[:, :, 0] == b)
	b_locs = list(zip(b_locs[0], b_locs[1]))
	if len(b_locs) != 1:
		raise RuntimeError("'b' could not be found (or was found multiple times) in the second block when combining blocks")
	b_loc = b_locs[0]
	# figure out how wide and tall the combined piece is going to be, and what the shift is
	height = max(a_loc[0] + (second_block.shape[0] - b_loc[0]), first_block.shape[0], second_block.shape[0])
	width = max(a_loc[1] + (second_block.shape[1] - b_loc[1]) + 1, first_block.shape[1], second_block.shape[1])
	row_shift = a_loc[0] - b_loc[0]
	col_shift = (a_loc[1] - b_loc[1]) + 1
	# combine the blocks, if we can
	combined_block = np.empty((height, width, 2), dtype=first_block.dtype)
	combined_block[:, :, :] = -1
	combined_block[:first_block.shape[0], :first_block.shape[1], :] = first_block
	for row in range(second_block.shape[0]):
		for col in range(second_block.shape[1]):
			if second_block[row, col, 0] == -1:
				continue
			row_combined = row + row_shift
			col_combined = col + col_shift
			try:
				if combined_block[row_combined, col_combined, 0] == -1:
					combined_block[row_combined, col_combined] = second_block[row, col]
				else:  # found a conflict
					return None
			except IndexError:
				continue
	return combined_block


def jigsaw_kruskals(graph: np.ndarray) -> np.ndarray:
	"""
	takes an adjacency matrix and uses kruskal's algorithm to build a reconstruction matrix
	:param graph: a numpy array with shape (n, n, 16) giving the dissimilarity scores for the n patches
	:return: a numpy array of shape (r, c, 2). the value at [i, j, 0] gives the index of the patch that should be located
	in that slot, and [i, j, 1] gives the rotation index of that patch
	"""
	graph = copy.copy(graph)
	n = graph.shape[0]
	assert graph.shape[1] == n
	for x in range(n):
		graph[x, x] = INFINITY
	sections = list()
	for i in range(n):
		sections.append(({i}, np.array([[[i, 0]]])))
	# join chunks until we only have one chunk
	while len(sections) > 1:
		# find the edge of minimum weight
		min_edges = np.where(graph == np.amin(graph))
		min_edges = list(zip(min_edges[0], min_edges[1], min_edges[2]))
		for a, b, r in min_edges:
			# find the two blocks
			first_block_index = -1
			second_block_index = -1
			for i, (s, _) in enumerate(sections):
				if a in s:
					first_block_index = i
					if second_block_index != -1:
						break
				if b in s:
					second_block_index = i
					if first_block_index != -1:
						break
			if first_block_index == -1:
				raise RuntimeError("error combining sets! ('a' was not found)")
			if second_block_index == -1:
				raise RuntimeError("error combining sets! ('b' was not found)")
			first_set, first_block = sections[first_block_index]
			second_set, second_block = sections[second_block_index]
			# combine the two blocks, if we can
			combined_block = combine_blocks(first_block, second_block, a, b, r)
			if combined_block is None:
				graph[a, b, r] = INFINITY
				continue
			# adjust our data for next iteration
			if second_block_index < first_block_index:
				first_block_index, second_block_index = second_block_index, first_block_index
			del sections[second_block_index]
			if len(sections) != 1:  # we can save ourselves the time of blocking out values if we're already done
				for i in first_set:
					for j in second_set:
						graph[i, j] = INFINITY
						graph[j, i] = INFINITY
			combined_set = first_set | second_set
			sections[first_block_index] = (combined_set, combined_block)
			break
	return sections[0][1]


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
	original_image = load_image_from_disk("TestImages/Strange.png")
	show_image(original_image)
	patch_list = scramble_image(original_image, 28)
	adjacency_matrix = build_graph(patch_list)
	reconstruction_matrix = jigsaw_kruskals(adjacency_matrix)
	reconstructed_image = assemble_image(patch_list, reconstruction_matrix)
	show_image(reconstructed_image)
