# based on the paper by Genady Paikin and Ayellet Tal, found at http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7299116


import multiprocessing as mp
import numpy as np
from constants import INFINITY
from tqdm import tqdm


def predict_3rd_pixel(col1: np.ndarray, col2: np.ndarray) -> np.ndarray:
	"""
	takes two columns (or rows) of pixels and predicts what the next column (or row) of pixels would be
	:param col1: the first list of pixels as a numpy array of shape (x, 3)
	:param col2: the second list of pixels as a numpy array of shape (x, 3)
	:return: the predicted third list of pixels as a numpy array of shape (x, 3)
	"""
	diff_col = col2 - col1
	expected = col2 + diff_col
	# restrict to [0, 255]
	coords = np.where(expected < 0)
	for row, channel in zip(*coords):
		expected[row, channel] = 0
	coords = np.where(expected > 255)
	for row, channel in zip(*coords):
		expected[row, channel] = 255
	return expected


def norm_l1(col1: np.ndarray, col2: np.ndarray) -> int:
	"""
	takes two columns (or rows) of pixels and gives the L_1 norm of the their difference
	:param col1: the first list of pixels as a numpy array of shape (x, 3)
	:param col2: the second list of pixels as a numpy array of shape (x, 3)
	:return: the L_1 norm
	"""
	diff_col = abs(col1 - col2)
	return sum(diff_col.flatten())


def dissimilarity_score_pt(patch1: np.ndarray, patch2: np.ndarray) -> int:
	"""
	takes two patches to be placed next to each other and gives an asymmetric dissimilarity score
	:param patch1: the patch to be placed on the left, as a numpy array of shape (x, y, 3)
	:param patch2: the patch to be placed on the right, as a numpy array of shape (x, z, 3)
	:return: the asymmetric dissimilarity score of `patch1` with `patch2`
	"""
	expected_col = predict_3rd_pixel(patch1[:, -2, :], patch1[:, -1, :])
	return norm_l1(expected_col, patch2[:, 0, :])


def dissimilarity_score_pt_mp(coord_and_patches: tuple) -> (tuple, float):
	# for use with the multiprocessing library
	(patch1_index, patch1_r, patch2_index, patch2_r), patch1, patch2 = coord_and_patches
	if patch1_index == patch2_index:
		score = INFINITY
	else:
		score = dissimilarity_score_pt(patch1, patch2)
	return (patch1_index, patch1_r, patch2_index, patch2_r), score


# generates input for dissimilarity_score_pt_mp()
class DissimilarityScorePtMpGenerator:
	def __init__(self, patches: list):
		self.patches = patches

	def __iter__(self):
		for patch1_index, patch1 in enumerate(self.patches):
			for r1 in range(4):
				patch1_rotated = np.rot90(patch1, r1)
				for patch2_index, patch2 in enumerate(self.patches):
					for r2 in range(4):
						patch2_rotated = np.rot90(patch2, r2)
						yield (patch1_index, r1, patch2_index, r2), patch1_rotated, patch2_rotated

	def __len__(self):
		n = len(self.patches)
		return n * 4 * n * 4


def get_dissimilarity_scores(patches: list) -> np.ndarray:
	"""
	takes a list of scrambled patches and creates a matrix that gives dissimilarity scores to each possible pairing of patches
	:param patches: list of square numpy arrays, each of the same shape (x, x, 3); the list's length is referred to as n
	:return: a numpy array of shape (n, 4, n, 4). the value at [i, r1, j, r2] indicates how costly pairing patch i and patch j is when they are rotated r1 and r2 times, respectively
	low values mean they are a good pairing; high values mean they are a bad pairing
	"""
	n = len(patches)
	score_matrix = np.empty((n, 4, n, 4), dtype=float)
	with mp.Pool() as pool:
		gen = DissimilarityScorePtMpGenerator(patches)
		result_generator = pool.imap_unordered(dissimilarity_score_pt_mp, gen)
		with tqdm(total=len(gen)) as progress_bar:
			for (patch1_index, r1, patch2_index, r2), score in result_generator:
				score_matrix[patch1_index, r1, patch2_index, r2] = score
				progress_bar.update()
	return score_matrix


def compatibility_score(dissimilarity_scores: np.ndarray, patch1_index: int, patch1_r: int, patch2_index: int, patch2_r: int) -> float:
	"""
	takes a matrix of all dissimilarity scores plus a particular combination of patches and gives the asymmetric compatibility score of that combination
	:param dissimilarity_scores: the matrix of all dissimilarity scores as a numpy array of shape (n, 4, n, 4), where n is the total number of patches
	:param patch1_index: the index referring to the first/left patch
	:param patch1_r: an int in range [0, 3] indicating how far the first/left patch has been rotated
	:param patch2_index: the index referring to the second/right patch
	:param patch2_r: an int in range [0, 3] indicating how far the second/right patch has been rotated
	:return: the asymmetric compatibility score of the two pieces
	"""
	d_score = dissimilarity_scores[patch1_index, patch1_r, patch2_index, patch2_r]
	relevant_slice = dissimilarity_scores[patch1_index, patch1_r, :, :]
	next_best_d_score = np.amin(relevant_slice[relevant_slice > d_score])  # if there is no second-best, this will raise a ValueError
	return 1.0 - (d_score / next_best_d_score)


def compatibility_score_mp(coord_and_dissimilarities: tuple) -> (tuple, float):
	# for use with the multiprocessing library
	(patch1_index, patch1_r, patch2_index, patch2_r), dissimilarity_scores = coord_and_dissimilarities
	if patch1_index == patch2_index:
		score = INFINITY
	else:
		score = compatibility_score(dissimilarity_scores, patch1_index, patch1_r, patch2_index, patch2_r)
	return (patch1_index, patch1_r, patch2_index, patch2_r), score


# generates input for compatibility_score_mp()
class CompatibilityScoreMpGenerator:
	def __init__(self, dissimilarity_scores: np.ndarray):
		self.n = dissimilarity_scores.shape[0]
		self.d = dissimilarity_scores

	def __iter__(self):
		for patch1_index in range(self.n):
			for r1 in range(4):
				for patch2_index in range(self.n):
					for r2 in range(4):
						yield (patch1_index, r1, patch2_index, r2), self.d

	def __len__(self):
		return self.n * 4 * self.n * 4


def get_compatibility_scores(dissimilarity_scores: np.ndarray) -> np.ndarray:
	"""
	takes a matrix of all dissimilarity scores and gives a matrix of compatibility scores
	:param dissimilarity_scores: the matrix of all dissimilarity scores as a numpy array of shape (n, 4, n, 4), where n is the total number of patches
	:return: a numpy array of shape (n, 4, n, 4). the value at [i, r1, j, r2] is a compatibility score for patch i and patch j when they are rotated r1 and r2 times, respectively.
	low values mean they are a bad pairing; high values mean they are a good pairing
	"""
	n = dissimilarity_scores.shape[0]
	score_matrix = np.empty((n, 4, n, 4), dtype=float)
	with mp.Pool() as pool:
		gen = CompatibilityScoreMpGenerator(dissimilarity_scores)
		result_generator = pool.imap_unordered(compatibility_score_mp, gen)
		with tqdm(total=len(gen)) as progress_bar:
			for (patch1_index, r1, patch2_index, r2), score in result_generator:
				score_matrix[patch1_index, r1, patch2_index, r2] = score
				progress_bar.update()
	return score_matrix


def get_best_buddies(compatibility_scores: np.ndarray) -> np.ndarray:
	"""
	takes a matrix of compatibility scores and gives a matrix of best buddies
	:param compatibility_scores: the matrix of all compatibility scores as a numpy array of shape (n, 4, n, 4), where n is the total number of patches
	:return: a matrix indicating best buddies for each edge of each patch, as a numpy array of shape (n, 4). the value at [i, r1] is a tuple indicating the best buddy for patch i
	when rotated r1 times. the tuple is of form (j, r2), indicating that patch j is the best buddy when rotated r2 times. if there is a value of `None` in place of a tuple, then
	patch i has no best buddy
	"""
	n = compatibility_scores.shape[0]
	buddy_matrix = np.empty((n, 4), dtype=tuple)
	# TODO: actually figure out best buddies
	buddy_matrix[:, :] = None  # this is trash
	return buddy_matrix


def pick_first_piece(buddy_matrix: np.ndarray, compatibility_scores: np.ndarray) -> int:
	"""
	takes info about best buddies and compatibility scores and selects the first piece to be placed
	:param buddy_matrix: the matrix indicating the best buddy of each piece, if any, as a numpy of shape (n, 4) containing tuples of form (piece_index, rotation_index)
	:param compatibility_scores: the matrix of all compatibility scores as a numpy array of shape (n, 4, n, 4), where n is the total number of patches
	:return: the index of the patch that is selected as our first piece to place
	"""
	# TODO
	pass


def jigsaw_pt(patches: list):
	"""
	takes a list of square patches and uses Paikin and Tal's algorithm to assemble the patches
	:param patches: a list of numpy arrays representing the scrambled patches of the original image
	:return: the re-assembled image as a numpy array of shape (r, c, 3)
	"""
	# TODO: actually implement the algorithm; these function calls are just examples
	dissimilarity_scores = get_dissimilarity_scores(patches)
	compatibility_scores = get_compatibility_scores(dissimilarity_scores)
	buddy_matrix = get_best_buddies(compatibility_scores)
	first_piece = pick_first_piece(buddy_matrix, compatibility_scores)
	return None
