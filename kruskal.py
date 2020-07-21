from typing import Union
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from constants import *
from functions import block_rot90, boring_score, combine_patches, coord_rot90, dissimilarity_score, rotations_from_combination_index, verify_reconstruction_matrix


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


def kruskals_reconstruction(graph: np.ndarray) -> np.ndarray:
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


def jigsaw_kruskals(patches: list):
	adjacency_matrix = build_graph(patches)
	reconstruction_matrix = kruskals_reconstruction(adjacency_matrix)
	valid = verify_reconstruction_matrix(reconstruction_matrix, len(patches))
	print(f"reconstruction_matrix valid: {valid}")
	if valid:
		reconstructed_image = assemble_image(patches, reconstruction_matrix)
		return reconstructed_image
	else:
		return None
