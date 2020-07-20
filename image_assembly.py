import datetime
# import math
from typing import Union
import multiprocessing as mp
import numpy as np
import random
from functions import assemble_patches, load_image_from_disk, show_image
from kldiv import data_to_probability_distribution, kl_divergence_symmetric
# from PatchPairBoolNet import PatchPairBoolNet
from tqdm import tqdm
# import torch


INFINITY = float("inf")

# enum-like values to indicate how far the patch has been rotated counter clockwise
ROTATION_0 = 0
ROTATION_90 = 1
ROTATION_180 = 2
ROTATION_270 = 3

# enum-like values for piece values in reconstruction matrices
YES_PIECE = 1
EXPANSION_SPACE = 0
NO_PIECE = -1

# globals
# hypothetical_min = 0
# hypothetical_max = 255


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


def scramble_image(image: np.ndarray, patch_size: int, seed: int = None) -> list:
	"""
	takes an rgb image and scrambles it into square patches, each at a random rotation
	:param image: numpy array of shape (r, c, 3) where r is the number of rows and c is the number of columns
	:param patch_size: number indicating how many pixels wide and tall each patch should be
	:param seed: a seed to use for the image scrambling. If none, then the scramble will be truly random
	:return: a list containing the scrambled patches, each of shape (patch_size, patch_size, 3)
	"""
	if seed is not None:
		random.seed(seed)
	# get the image (we actually have it: np.ndarray)
	# Break image into patches
	vertical_patches = image.shape[0] // patch_size
	horizontal_patches = image.shape[1] // patch_size

	patched_array = list()

	for i in range(vertical_patches):
		# starting pixel
		vert_pixel_location = patch_size * i
		for j in range(horizontal_patches):
			# starting pixel
			hor_pix_location = patch_size * j
			patch = image[vert_pixel_location:vert_pixel_location+patch_size, hor_pix_location:hor_pix_location+patch_size, :]
			patched_array.append(patch)

	# Scramble the patches (AKA put them in a random order)
	random.shuffle(patched_array)
	# for i in range(patched_array):
	# Doing it without .shuffle command: pick a random index between 0 and i, swap what is in spot 0 and spot i, do this a couple times
	# Also give them a random rotation!
	rotated_patched_array = list()
	for patch in patched_array:
		rotated_patched_array.append(np.rot90(patch, random.randrange(4)))
	return rotated_patched_array


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
	patch_size = combo_patch.shape[0]
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


def build_graph(patches: list) -> np.ndarray:
	"""
	takes a list of scrambled patches and creates an adjacency matrix that gives scores to each possible pairing of patches
	:param patches: list of square numpy arrays, each of the same shape (x, x, 3); the list's length is referred to as n
	:return: a numpy array of shape (n, n, 16). the value at [i, j, c] indicates how costly pairing patch i and patch j is when using combination index c.
	low values mean they are a good pairing; high values mean they are a bad pairing
	"""
	functions = [boring_score, dissimilarity_score]
	n = len(patches)
	score_matrix = np.empty((n, n, 16), dtype=float)
	with mp.Pool() as pool:
		gen = PatchPairGenerator(patches, functions)
		score_count = len(gen)
		result_generator = pool.imap_unordered(combo_score_mp, gen)
		with tqdm(total=score_count) as progress_bar:
			for (i, j, c), score in result_generator:
				score_matrix[i, j, c] = score
				progress_bar.update()
	return score_matrix


# def build_graph_from_nn(patches: list, nn_path: str) -> np.ndarray:
# 	"""
# 	takes a list of scrambled patches and creates an adjacency matrix that gives scores to each possible pairing of patches. uses a Neural Network to assign scores
# 	:param patches: list of square numpy arrays, each of the same shape (x, x, 3); the list's length is referred to as n
# 	:param nn_path: the path to the NN's saved location on disk
# 	:return: a numpy array of shape (n, n, 16). the value at [i, j, c] indicates how costly pairing patch i and patch j is when using combination index c.
# 	low values mean they are a good pairing; high values mean they are a bad pairing
# 	"""
# 	# get the nn set up
# 	net = PatchPairBoolNet()
# 	net.load_state_dict(torch.load(nn_path))
# 	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 	net.to(device)
# 	# get all the scores using the net
# 	n = len(patches)
# 	pairing_scores = np.empty((n, n, 16), dtype=float)
# 	# "for each" loop that gives you the index
# 	for i, patch1 in enumerate(patches):
# 		for j, patch2 in enumerate(patches):
# 			for c in range(16):
# 				if i == j:
# 					pairing_scores[i, j, c] = INFINITY
# 					continue
# 				combined = combine_patches(patch1, patch2, c)
# 				score = net.forward_numpy(combined, device)
# 				pairing_scores[i, j, c] = score
# 		if (i + 1) % 5 == 0:
# 			print(f"completed {i + 1} of {n} patches' scores ({(100 * (i + 1)) // n}")
# 	return pairing_scores


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


def combine_blocks(first_block: np.ndarray, second_block: np.ndarray, a: int, b: int, r: int) -> Union[None, np.ndarray]:
	"""
	takes a two subsections of a reconstruction matrix, two patch indices, and a combination index. returns the combined subsection of a reconstruction matrix (if possible)
	:param first_block: a subsection of a reconstruction matrix as a numpy array of shape (m, n, 2)
	:param second_block: a subsection of a reconstruction matrix as a numpy array of shape (r, c, 2)
	:param a: the value to be found in `first_block` as the attaching point
	:param b: the value to be found in `second_block` as the attaching point
	:param r: the combination index (a value in range [0, 15]) indicating how the two attaching points should be rotated as the are placed next to each other
	:return: the combination of `first_block` and `second_block` as specified as a numpy array of shape (r, c, 2). if the combination is not possible, `None` is returned
	"""
	first_block = np.copy(first_block)
	second_block = np.copy(second_block)
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
	# find how far each base piece has been rotated already and reverse it
	a_rotate = (4 - first_block[a_loc][1]) % 4
	b_rotate = (4 - second_block[b_loc][1]) % 4
	# get the rotation values of each block
	r1, r2 = rotations_from_combination_index(r)
	r1_reversed = (4 - r1) % 4
	r2_reversed = (4 - r2) % 4
	# when the blocks are rotated, a_loc and b_loc will have changed
	a_loc = coord_rot90(a_loc[0], a_loc[1], first_block.shape[0], first_block.shape[1], a_rotate + r1_reversed)
	b_loc = coord_rot90(b_loc[0], b_loc[1], second_block.shape[0], second_block.shape[1], b_rotate + r2_reversed)
	first_block = block_rot90(first_block, a_rotate + r1_reversed)
	second_block = block_rot90(second_block, b_rotate + r2_reversed)
	# figure out how wide and tall the combined piece is going to be, and what the shift is
	row_shift = a_loc[0] - b_loc[0]
	if row_shift >= 0:
		first_row_shift = 0
		second_row_shift = row_shift
		height = max(first_block.shape[0], second_row_shift + second_block.shape[0])
	else:
		first_row_shift = -row_shift
		second_row_shift = 0
		height = max(first_row_shift + first_block.shape[0], second_block.shape[0])
	col_shift = (a_loc[1] - b_loc[1]) + 1
	if col_shift >= 0:
		first_col_shift = 0
		second_col_shift = col_shift
		width = max(first_block.shape[1], second_col_shift + second_block.shape[1])
	else:
		first_col_shift = -col_shift
		second_col_shift = 0
		width = max(first_col_shift + first_block.shape[1], second_block.shape[1])
	# combine the blocks, if we can
	combined_block = np.empty((height, width, 2), dtype=first_block.dtype)
	combined_block[:, :, :] = NO_PIECE
	combined_block[first_row_shift:(first_block.shape[0] + first_row_shift), first_col_shift:(first_block.shape[1] + first_col_shift), :] = first_block
	for row in range(second_block.shape[0]):
		for col in range(second_block.shape[1]):
			if second_block[row, col, 0] == NO_PIECE:
				continue
			row_combined = row + second_row_shift
			col_combined = col + second_col_shift
			try:
				if combined_block[row_combined, col_combined, 0] == NO_PIECE:
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
			if a == b:
				continue
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


def find_expansion_spaces(construction_matrix: np.ndarray) -> list:
	"""
	takes a construction matrix as used in `jigsaw_prims` and finds placed marked for expansion
	:param construction_matrix: a 2-dimensional numpy array containing values of `YES_PIECE`, `NO_PIECE`, or `EXPANSION_SPACE`
	:return: a list of tuples indicating elements in the array with value of `EXPANSION_SPACE`
	"""
	unzipped_coordinates = np.where(construction_matrix == EXPANSION_SPACE)
	return list(zip(unzipped_coordinates[0], unzipped_coordinates[1]))


def prims_placement_score(construction_matrix: np.ndarray, assembled_image: np.ndarray, patch_to_place: np.ndarray, row: int, col: int) -> float:
	"""
	takes a construction matrix as used in `jigsaw_prims`, the partially-assembled rgb image that it represents, a patch to place, and the place to put it;
	gives a score to the hypothetical placement of the patch as specified
	:param construction_matrix: a 2-dimensional numpy array containing values of `YES_PIECE`, `NO_PIECE`, or `EXPANSION_SPACE`
	:param assembled_image: the partially-assembled rgb image corresponding to `construction_matrix` as a numpy array of shape (r, c, 3)
	:param patch_to_place: the square patch to be placed in `assembled_image`, already rotated as it needs to be placed, as a numpy array of shape (x, x, 3)
	:param row: the index of the row in `construction_matrix` where the patch will be placed
	:param col: the index of the column in `construction_matrix` where the patch will be placed
	:return: a score indicating how costly placing the patch is as specified. low values mean it is a good placement; high values mean it is a bad placement
	"""
	patch_size = patch_to_place.shape[0]
	neighbors = list()
	for row_shift in [-1, 1]:
		neighbor_row = row + row_shift
		if 0 <= neighbor_row < construction_matrix.shape[0]:
			if construction_matrix[neighbor_row, col] == YES_PIECE:
				neighbors.append((neighbor_row, col))
	for col_shift in [-1, 1]:
		neighbor_col = col + col_shift
		if 0 <= neighbor_col < construction_matrix.shape[1]:
			if construction_matrix[row, neighbor_col] == YES_PIECE:
				neighbors.append((row, neighbor_col))
	if len(neighbors) == 0:
		raise RuntimeError("prims_placement_score could not find a neighbor")
	neighbor_scores = list()
	for neighbor_row, neighbor_col in neighbors:
		# pull the neighbor patch out of the image
		neighbor_patch = assembled_image[(patch_size * neighbor_row):(patch_size * (neighbor_row + 1)), (patch_size * neighbor_col):(patch_size * (neighbor_col + 1)), :]
		# combine them horizontally so we can give them to the scoring function
		if neighbor_row < row:  # neighbor is above
			combined = combine_patches(np.rot90(neighbor_patch), np.rot90(patch_to_place))
		elif neighbor_row > row:  # neighbor is below
			combined = combine_patches(np.rot90(patch_to_place), np.rot90(neighbor_patch))
		elif neighbor_col < col:  # neighbor is left
			combined = combine_patches(neighbor_patch, patch_to_place)
		elif neighbor_col > col:  # neighbor is right
			combined = combine_patches(patch_to_place, neighbor_patch)
		else:
			raise RuntimeError(f"prims_placement_score picked a weird neighbor...? original = ({row},{col}), neighbor = ({neighbor_row},{neighbor_col}")
		# get the boring score, then normalize it
		b_score = boring_score(combined)
		# b_score = 100 * (b_score - hypothetical_min) / (hypothetical_max - hypothetical_min)
		d_score = dissimilarity_score(combined)
		k_score = kl_score(combined)
		score = sum(s * w for s, w in zip([b_score, d_score, k_score], [0.62, 1.05, 1.0]))
		# show_image(combined, str(score))
		neighbor_scores.append(score)
	# penalize for having blank neighbors to keep it square-ish
	neighbor_scores += ([20 * max(neighbor_scores)] * (4 - len(neighbor_scores)))
	# average the score from all of the neighbors
	to_return = sum(neighbor_scores) / len(neighbor_scores)
	# show what we've done for debugging
	# hypothetical_placement = np.copy(assembled_image)
	# hypothetical_placement[(patch_size * row):(patch_size * (row + 1)), (patch_size * col):(patch_size * (col + 1)), :] = patch_to_place
	# show_image(hypothetical_placement, str(to_return))
	return to_return


def jigsaw_prims(patches: list) -> np.ndarray:
	"""
	takes a list of square patches and uses prim's algorithm to assemble the patches
	:param patches: a list of numpy arrays representing the scrambled patches of the original image
	:return: the re-assembled image as a numpy array of shape (x, y, 3)
	"""
	n = len(patches)
	patches_available = list(range(n))
	# figure out the patch size
	patch_size = patches[0].shape[0]
	# a matrix to store scores already calculated
	scores_matrix = np.empty((3, 3, n, 4), dtype=float)
	scores_matrix[:, :, :, :] = INFINITY
	# a matrix to store which slots have pieces
	construction_matrix = np.array([[NO_PIECE, EXPANSION_SPACE, NO_PIECE], [EXPANSION_SPACE, YES_PIECE, EXPANSION_SPACE], [NO_PIECE, EXPANSION_SPACE, NO_PIECE]])
	# an array holding the actual pixel values (starts at all black)
	assembled_image = np.zeros((patch_size * 3, patch_size * 3, 3), dtype=int)
	# pull the start patch out and place it in the assembled image
	assembled_image[patch_size:(2 * patch_size), patch_size:(2 * patch_size), :] = patches[0]
	show_image(assembled_image)
	patches_available.remove(0)
	# while the list of remaining patches isn't empty, pull out the next best option and place it
	while len(patches_available) > 0:
		# to place the next best option:
		# for each empty spot adjacent to a placed patch, try putting each unused patch in each possible orientation
		# for each of those options, give it a score accounting for all of its neighbors
		# whichever of all the options has the best score is the one that gets placed
		# -----
		# find places where we could put a patch
		expansion_spaces = find_expansion_spaces(construction_matrix)
		if len(expansion_spaces) == 0:
			raise RuntimeError(f"jigsaw_prims could not find any expansion spaces, but there are more patches to place")
		# make sure they all have scores
		for row, col in expansion_spaces:
			for patch_index in patches_available:
				for r in range(4):
					if scores_matrix[row, col, patch_index, r] == INFINITY:
						patch_to_place = np.rot90(patches[patch_index], r)
						# if row == 0 and col == 1 and patch_index == 3 and r == 3:
						# 	show_image(patch_to_place)
						# 	show_image(assembled_image)
						# 	print(construction_matrix)
						score = prims_placement_score(construction_matrix, assembled_image, patch_to_place, row, col)
						scores_matrix[row, col, patch_index, r] = score
		# find the placement of the best score
		min_scores = np.where(scores_matrix == np.amin(scores_matrix))
		min_scores = list(zip(min_scores[0], min_scores[1], min_scores[2], min_scores[3]))
		# place the best patch
		for row, col, patch_index, r in min_scores:
			# make sure we picked a piece that hasn't been used yet
			if patch_index not in patches_available:
				continue  # shouldn't need this, but just in case
			patches_available.remove(patch_index)
			# mark the space as taken; update its neighbors as available for placement
			construction_matrix[row, col] = YES_PIECE
			# actually place the patch
			rotated_patch = np.rot90(patches[patch_index], r)
			assembled_image[(row * patch_size):((row + 1) * patch_size), (col * patch_size):((col + 1) * patch_size), :] = rotated_patch
			show_image(assembled_image, str(scores_matrix[row, col, patch_index, r]))
			# set the place's scores and the patch's scores to infinity to mark them as taken/used
			scores_matrix[row, col, :, :] = INFINITY
			scores_matrix[:, :, patch_index, :] = INFINITY
			# check if we've placed a piece at the edge of the available canvas; expand if we did
			if row == 0 or row == construction_matrix.shape[0] - 1:
				new_scores_row = np.empty((1, scores_matrix.shape[1], scores_matrix.shape[2], scores_matrix.shape[3]), dtype=float)
				new_scores_row[:, :, :, :] = INFINITY
				new_construction_row = np.empty((1, construction_matrix.shape[1]), dtype=int)
				new_construction_row[:, :] = NO_PIECE
				new_image_row = np.zeros((patch_size, assembled_image.shape[1], assembled_image.shape[2]), dtype=int)
				if row == 0:  # we placed one on the top row
					scores_matrix = np.concatenate((new_scores_row, scores_matrix), axis=0)
					construction_matrix = np.concatenate((new_construction_row, construction_matrix), axis=0)
					assembled_image = np.concatenate((new_image_row, assembled_image), axis=0)
					row += 1
				else:  # we placed one on the bottom row
					scores_matrix = np.concatenate((scores_matrix, new_scores_row), axis=0)
					construction_matrix = np.concatenate((construction_matrix, new_construction_row), axis=0)
					assembled_image = np.concatenate((assembled_image, new_image_row), axis=0)
			if col == 0 or col == construction_matrix.shape[1] - 1:
				new_scores_col = np.empty((scores_matrix.shape[0], 1, scores_matrix.shape[2], scores_matrix.shape[3]), dtype=float)
				new_scores_col[:, :, :, :] = INFINITY
				new_construction_col = np.empty((construction_matrix.shape[0], 1), dtype=int)
				new_construction_col[:, :] = NO_PIECE
				new_image_col = np.zeros((assembled_image.shape[0], patch_size, assembled_image.shape[2]), dtype=int)
				if col == 0:  # we placed one on the left column
					scores_matrix = np.concatenate((new_scores_col, scores_matrix), axis=1)
					construction_matrix = np.concatenate((new_construction_col, construction_matrix), axis=1)
					assembled_image = np.concatenate((new_image_col, assembled_image), axis=1)
					col += 1
				else:  # we placed one on the bottom row
					scores_matrix = np.concatenate((scores_matrix, new_scores_col), axis=1)
					construction_matrix = np.concatenate((construction_matrix, new_construction_col), axis=1)
					assembled_image = np.concatenate((assembled_image, new_image_col), axis=1)
			# check and update the neighbors of the newly placed patch
			# also reset the adjacent places' scores to infinity so they'll be recalculated accounting for the new piece
			for row_shift in [-1, 1]:
				if construction_matrix[row + row_shift, col] == NO_PIECE:
					construction_matrix[row + row_shift, col] = EXPANSION_SPACE
				if construction_matrix[row + row_shift, col] == EXPANSION_SPACE:
					scores_matrix[row + row_shift, col, :, :] = INFINITY
			for col_shift in [-1, 1]:
				if construction_matrix[row, col + col_shift] == NO_PIECE:
					construction_matrix[row, col + col_shift] = EXPANSION_SPACE
				if construction_matrix[row, col + col_shift] == EXPANSION_SPACE:
					scores_matrix[row, col + col_shift, :, :] = INFINITY
			break
	# trim edges
	h, w, _ = assembled_image.shape
	assembled_image = assembled_image[patch_size:(h - patch_size), patch_size:(w - patch_size), :]
	return assembled_image


def assemble_image(patches: list, construction_matrix: np.ndarray) -> Union[None, np.ndarray]:
	"""
	takes a list of patches and a reconstruction matrix to assemble the patches
	:param patches: a list of numpy arrays representing the scrambled patches of the original image
	:param construction_matrix: a numpy array of shape (r, c, 2). the value at [i, j, 0] gives the index of the patch that
	should be located in that slot, and [i, j, 1] gives the rotation index of that patch
	:return: the re-assembled image as a numpy array of shape (x, y, 3)
	"""
	# tells us how many pixels tall/wide each patch is
	patch_size = patches[0].shape[0]
	# num of rows/cols (of patches)
	rows = construction_matrix.shape[0]
	columns = construction_matrix.shape[1]
	# putting it together, it's an int because we are working with an image
	# starts at zeros because 0 is black
	assembled = np.zeros((rows * patch_size, columns * patch_size, 3), dtype=int)
	# iterate across construction_matrix... (it tells us what piece to pull and where to rotate)
	for i in range(construction_matrix.shape[0]):
		for j in range(construction_matrix.shape[1]):
			# index of the patch
			patch_index = construction_matrix[i, j, 0]
			# if it's a -1, that means that no piece goes there and we leave it as black
			if patch_index == -1:
				continue
			# rotation index of the patch
			rotation_command = construction_matrix[i, j, 1]
			# get the actual patch
			patch_in_question = patches[patch_index]
			# rotate it
			rotated_patch = np.rot90(patch_in_question, rotation_command)
			# put it in its place
			assembled[(i * patch_size):(patch_size * (i + 1)), (j * patch_size):(patch_size * (j + 1)), :] = rotated_patch
	return assembled


def compare_images(image1: np.ndarray, image2: np.ndarray):
	"""
	takes two rgb images and displays a red/green image of where the two input images have the same pixel values; green pixels mean identical, red pixels mean they differ
	:param image1: the first image for comparison as a numpy array of shape (m, n, 3)
	:param image2: the second image for comparison as a numpy array of shape (r, c, 3)
	:return: None
	"""
	red_pixel = np.array([205, 0, 0])
	green_pixel = np.array([0, 205, 40])
	full_height = max(image1.shape[0], image2.shape[0])
	inner_height = min(image1.shape[0], image2.shape[0])
	full_width = max(image1.shape[1], image2.shape[1])
	inner_width = min(image1.shape[1], image2.shape[1])
	to_show = np.empty((full_height, full_width, 3), dtype=int)
	for i in range(to_show.shape[0]):
		for j in range(to_show.shape[1]):
			if i >= inner_height or j >= inner_width:
				to_show[i, j, :] = red_pixel
				continue
			identical = True
			for c in range(3):
				if image1[i, j, c] != image2[i, j, c]:
					identical = False
					break
			if identical:
				to_show[i, j, :] = green_pixel
			else:
				to_show[i, j, :] = red_pixel
	show_image(to_show)


if __name__ == "__main__":
	original_image = load_image_from_disk("TestImages/Giraffe.jpg")
	show_image(original_image)
	# ps = original_image.shape[1] // 2
	ps = 28
	patch_list = scramble_image(original_image, ps, 4)
	show_image(assemble_patches(patch_list, original_image.shape[1] // ps))
	# hypothetical_min = 85 + (15.038 * math.log(ps))
	# hypothetical_max = 255 - (14.235 * math.log(ps))
	print(f"algorithm start time: {datetime.datetime.now()}")
	reconstructed_image = jigsaw_prims(patch_list)
	print(f"algorithm end time: {datetime.datetime.now()}")
	if reconstructed_image is not None:
		show_image(reconstructed_image)
		for rotation in range(4):
			compare_images(original_image, np.rot90(reconstructed_image, rotation))
	else:
		print("reconstructed_image is None")
