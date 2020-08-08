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


def predict_3rd_pixel_mp(coord_last_columns: (np.ndarray, np.ndarray)) -> ((int, int), np.ndarray):
	coord, col1, col2 = coord_last_columns
	return coord, predict_3rd_pixel(col1, col2)


# generates input for predict_3rd_pixel_mp()
class Predict3rdPixelMpGenerator:
	def __init__(self, patches: list):
		self.patches = patches

	def __iter__(self):
		for patch_index, patch in enumerate(self.patches):
			for r in range(4):
				patch_rotated = np.rot90(patch, r)
				yield (patch_index, r), patch_rotated[:, -2, :], patch_rotated[:, -1, :]

	def __len__(self):
		n = len(self.patches)
		return n * 4


def get_3rd_pixel_predictions(patches: list) -> np.ndarray:
	"""
	TODO
	:param patches:
	:return:
	"""
	n = len(patches)
	prediction_matrix = np.empty((n, 4), dtype=np.ndarray)
	with mp.Pool() as pool:
		gen = Predict3rdPixelMpGenerator(patches)
		result_generator = pool.imap_unordered(predict_3rd_pixel_mp, gen)
		with tqdm(total=len(gen)) as progress_bar:
			for (patch_index, r), predicted_column in result_generator:
				prediction_matrix[patch_index, r] = predicted_column
				progress_bar.update()
	return prediction_matrix


def norm_l1(col1: np.ndarray, col2: np.ndarray) -> int:
	"""
	takes two columns (or rows) of pixels and gives the L_1 norm of the their difference
	:param col1: the first list of pixels as a numpy array of shape (x, 3)
	:param col2: the second list of pixels as a numpy array of shape (x, 3)
	:return: the L_1 norm
	"""
	diff_col = abs(col1 - col2)
	return sum(diff_col.flatten())


def norm_l1_mp(coord_and_columns: tuple) -> (tuple, float):
	# for use with the multiprocessing library
	(patch1_index, patch1_r, patch2_index, patch2_r), predicted, actual = coord_and_columns
	if patch1_index == patch2_index:
		score = INFINITY
	else:
		score = norm_l1(predicted, actual)
	return (patch1_index, patch1_r, patch2_index, patch2_r), score


# generates input for dissimilarity_score_pt_mp()
class DissimilarityScorePtMpGenerator:
	def __init__(self, patches: list, predictions_matrix: np.ndarray):
		self.patches = patches
		self.predictions = predictions_matrix

	def __iter__(self):
		for patch1_index, patch1 in enumerate(self.patches):
			for r1 in range(4):
				predicted_column = self.predictions[patch1_index, r1]
				for patch2_index, patch2 in enumerate(self.patches):
					for r2 in range(4):
						actual_column = patch2[:, 0, :]
						yield (patch1_index, r1, patch2_index, r2), predicted_column, actual_column

	def __len__(self):
		n = len(self.patches)
		return n * 4 * n * 4


def get_dissimilarity_scores(patches: list, predictions_matrix: np.ndarray) -> np.ndarray:
	"""
	takes a list of scrambled patches and creates a matrix that gives dissimilarity scores to each possible pairing of patches
	:param patches: list of square numpy arrays, each of the same shape (x, x, 3); the list's length is referred to as n
	:param predictions_matrix: TODO
	:return: a numpy array of shape (n, 4, n, 4). the value at [i, r1, j, r2] indicates how costly pairing patch i and patch j is when they are rotated r1 and r2 times, respectively
	low values mean they are a good pairing; high values mean they are a bad pairing
	"""
	n = len(patches)
	score_matrix = np.empty((n, 4, n, 4), dtype=float)
	with mp.Pool() as pool:
		gen = DissimilarityScorePtMpGenerator(patches, predictions_matrix)
		result_generator = pool.imap_unordered(norm_l1_mp, gen)
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
	next_best_d_score = np.amin(relevant_slice[relevant_slice != d_score])  # if there is no second-best, this will raise a ValueError
	if next_best_d_score == INFINITY:
		return 0.0
	return 1.0 - (d_score / next_best_d_score)


def compatibility_score_mp(coord_and_dissimilarities: tuple) -> (tuple, float):
	# for use with the multiprocessing library
	(patch1_index, patch1_r, patch2_index, patch2_r), dissimilarity_scores = coord_and_dissimilarities
	if patch1_index == patch2_index:
		score = -INFINITY
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
	# look at each piece in each rotation, find its best neighbor
	# piece we are looking at
	for i in range(n):
		# rotation index
		for r1 in range(compatibility_scores.shape[1]):
			r1_inverse = (r1 + 2) % 4
			# pull out the block of scores that are relevant
			region = compatibility_scores[i, r1, :, :]
			# look for the largest value
			largest_coordinates = np.where(region == np.amax(region))
			largest_coordinates = list(zip(largest_coordinates[0], largest_coordinates[1]))
			found_buddy = False
			for j, r2 in largest_coordinates:
				# see if it's "mutual"
				r2_inverse = (r2 + 2) % 4
				region_inverse = compatibility_scores[j, r2_inverse, :, :]
				largest_coordinates_inverse = np.where(region_inverse == np.amax(region_inverse))
				largest_coordinates_inverse = list(zip(largest_coordinates_inverse[0], largest_coordinates_inverse[1]))
				if (i, r1_inverse) in largest_coordinates_inverse:
					buddy_matrix[i, r1] = (j, r2)
					found_buddy = True
					break
				# if it is, add it to buddy_matrix
			if not found_buddy:  # none of the values tied for best are a best buddy
				buddy_matrix[i, r1] = None
	return buddy_matrix


def pick_first_piece(buddy_matrix: np.ndarray, compatibility_scores: np.ndarray) -> int:
	"""
	takes info about best buddies and compatibility scores and selects the first piece to be placed
	:param buddy_matrix: the matrix indicating the best buddy of each piece, if any, as a numpy of shape (n, 4) containing tuples of form (piece_index, rotation_index)
	:param compatibility_scores: the matrix of all compatibility scores as a numpy array of shape (n, 4, n, 4), where n is the total number of patches
	:return: the index of the patch that is selected as our first piece to place
	"""
	candidates = list()
	n = buddy_matrix.shape[0]
	for i in range(n):
		buddy_count = sum([buddy_matrix[i, r] is not None for r in range(buddy_matrix.shape[1])])
		if buddy_count != 4:
			continue
		# check if its buddies each have 4 buddies
		passes = True
		for r1 in range(buddy_matrix.shape[1]):
			buddy_index, _ = buddy_matrix[i, r1]
			buddy_count2 = sum([buddy_matrix[buddy_index, r] is not None for r in range(buddy_matrix.shape[1])])
			if buddy_count2 != 4:
				passes = False
				break
		if passes:
			candidates.append(i)
	if len(candidates) == 0:
		# TODO: ...?
		pass
	if len(candidates) == 1:
		return candidates[0]
	# pick the piece that has the best sum of mutual compatibility scores with its 4 best buddies
	best_score = -1.0
	best_index = -1
	for i in candidates:
		# sum its mutual compatibility scores with its 4 best buddies
		mutual_compatibility_sum = 0.0
		for r1 in range(buddy_matrix.shape[1]):
			j, r2 = buddy_matrix[i, r1]
			compatibility1 = compatibility_scores[i, r1, j, r2]
			compatibility2 = compatibility_scores[j, (r2 + 2) % 4, i, (r1 + 2) % 4]
			mutual_compatibility_sum += (compatibility1 + compatibility2) / 2.0
		# check if this score is better than any others we've seen
		if mutual_compatibility_sum > best_score:
			best_score = mutual_compatibility_sum
			best_index = i
	return best_index


def jigsaw_pt(patches: list):
	"""
	takes a list of square patches and uses Paikin and Tal's algorithm to assemble the patches
	:param patches: a list of numpy arrays representing the scrambled patches of the original image
	:return: the re-assembled image as a numpy array of shape (r, c, 3)
	"""
	# TODO: actually implement the algorithm; these function calls are just examples
	print("computing 3rd pixel predictions...")
	predictions_matrix = get_3rd_pixel_predictions(patches)
	print("computing dissimilarity scores...")
	dissimilarity_scores = get_dissimilarity_scores(patches, predictions_matrix)
	print("computing initial compatibility scores...")
	compatibility_scores = get_compatibility_scores(dissimilarity_scores)
	print("finding initial best buddies...")
	buddy_matrix = get_best_buddies(compatibility_scores)
	print("selecting first piece...")
	first_piece = pick_first_piece(buddy_matrix, compatibility_scores)
	print(f"first piece selected: {first_piece}")
	return None
