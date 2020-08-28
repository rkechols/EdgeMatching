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


def predict_3rd_pixel_mp(coord_last_columns: (tuple, np.ndarray, np.ndarray)) -> (tuple, np.ndarray):
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


def norm_l1_mp(coord_and_columns: (tuple, np.ndarray, np.ndarray)) -> (tuple, float):
	# for use with the multiprocessing library
	(patch1_index, patch1_r, patch2_index, patch2_r), predicted, actual, = coord_and_columns
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
						patch2_rotated = np.rot90(patch2, r2)
						actual_column = patch2_rotated[:, 0, :]
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


def get_best_neighbors(dissimilarity_scores: np.ndarray) -> np.ndarray:
	"""
	TODO
	:param dissimilarity_scores:
	:return:
	"""
	n = dissimilarity_scores.shape[0]
	best_neighbors = np.empty((n, 4), dtype=tuple)
	for i in range(n):
		for r1 in range(dissimilarity_scores.shape[1]):
			relevant_section = dissimilarity_scores[i, r1, :, :]
			best_dissimilarity = np.where(relevant_section == np.amin(relevant_section))
			best_neighbors[i, r1] = list(zip(*best_dissimilarity))[0]  # just take the first if there's a tie
	return best_neighbors


def compatibility_score(dissimilarity_scores: np.ndarray, best_neighbors: np.ndarray, patch_index: int, r: int) -> np.ndarray:
	"""
	TODO
	:param dissimilarity_scores:
	:param best_neighbors:
	:param patch_index:
	:param r:
	:return:
	"""
	relevant_slice = dissimilarity_scores[patch_index, r, :, :]
	scores_to_return = np.empty_like(relevant_slice)
	best_d_j, best_d_r = best_neighbors[patch_index, r]
	best_d_score = dissimilarity_scores[patch_index, r, best_d_j, best_d_r]
	next_best_d_score = np.amin(relevant_slice[relevant_slice != best_d_score])  # if there is no second-best, this will raise a ValueError
	for patch_index2 in range(scores_to_return.shape[0]):
		for r2 in range(scores_to_return.shape[1]):
			if next_best_d_score != 0:
				scores_to_return[patch_index2, r2] = 1 - (relevant_slice[patch_index2, r2] / next_best_d_score)
			else:
				scores_to_return[patch_index2, r2] = 0.0
	return scores_to_return


def compatibility_score_mp(coord_and_dissimilarities: (tuple, np.ndarray, np.ndarray)) -> (tuple, np.ndarray):
	# for use with the multiprocessing library
	(patch_index, r), dissimilarity_scores, best_neighbors = coord_and_dissimilarities
	compatibility_scores = compatibility_score(dissimilarity_scores, best_neighbors, patch_index, r)
	return (patch_index, r), compatibility_scores


# generates input for compatibility_score_mp()
class CompatibilityScoreMpGenerator:
	def __init__(self, dissimilarity_scores: np.ndarray, best_neighbors: np.ndarray):
		self.n = dissimilarity_scores.shape[0]
		self.d = dissimilarity_scores
		self.b = best_neighbors

	def __iter__(self):
		for patch_index in range(self.n):
			for r in range(4):
				yield (patch_index, r), self.d, self.b

	def __len__(self):
		return self.n * 4


def get_compatibility_scores(dissimilarity_scores: np.ndarray, best_neighbors: np.ndarray) -> np.ndarray:
	"""
	takes a matrix of all dissimilarity scores and gives a matrix of compatibility scores
	:param dissimilarity_scores: the matrix of all dissimilarity scores as a numpy array of shape (n, 4, n, 4), where n is the total number of patches
	:param best_neighbors: TODO
	:return: TODO
	low values mean they are a bad pairing; high values mean they are a good pairing
	"""
	n = dissimilarity_scores.shape[0]
	score_matrix = np.empty((n, 4, n, 4), dtype=float)
	with mp.Pool() as pool:
		gen = CompatibilityScoreMpGenerator(dissimilarity_scores, best_neighbors)
		result_generator = pool.imap_unordered(compatibility_score_mp, gen)
		with tqdm(total=len(gen)) as progress_bar:
			for (patch1_index, r1), scores_section in result_generator:
				score_matrix[patch1_index, r1, :, :] = scores_section
				progress_bar.update()
	return score_matrix


def get_best_buddies(compatibility_scores: np.ndarray) -> np.ndarray:
	"""
	takes a matrix of compatibility scores and gives a matrix of best buddies
	:param compatibility_scores: TODO
	:return: a matrix indicating best buddies for each edge of each patch, as a numpy array of shape (n, 4). the value at [i, r1] is a tuple indicating the best buddy for patch i
	when rotated r1 times. the tuple is of form (j, r2), indicating that patch j is the best buddy when rotated r2 times. if there is a value of `None` in place of a tuple, then
	patch i has no best buddy
	"""
	n = compatibility_scores.shape[0]
	buddy_matrix = np.empty((n, 4), dtype=tuple)
	# look at each piece in each rotation, see if its best neighbor is mutual
	for i in range(n):
		for r1 in range(compatibility_scores.shape[1]):
			r1_inverse = (r1 + 2) % 4
			relevant_section = compatibility_scores[i, r1, :, :]
			# find the best compatibility score
			best_compatibility = np.where(relevant_section == np.amax(relevant_section))
			best_compatibility = list(zip(*best_compatibility))
			found_buddy = False
			for j, r2 in best_compatibility:  # there might be ties; try all of them
				# see if it's "mutual"; if it is, add it to buddy_matrix
				r2_inverse = (r2 + 2) % 4
				relevant_section_back = compatibility_scores[j, r2_inverse, :, :]
				best_compatibility_back = np.where(relevant_section_back == np.amax(relevant_section_back))
				best_compatibility_back = list(zip(*best_compatibility_back))
				for back_i, back_r1 in best_compatibility_back:
					if i == back_i and r1_inverse == back_r1:
						buddy_matrix[i, r1] = (j, r2)
						found_buddy = True
						break
				if found_buddy:
					break
			if not found_buddy:  # this one has no best buddy :(
				buddy_matrix[i, r1] = None
	return buddy_matrix


def pick_first_piece(buddy_matrix: np.ndarray, compatibility_scores: np.ndarray, best_neighbors: np.ndarray) -> int:
	"""
	takes info about best buddies and compatibility scores and selects the first piece to be placed
	:param buddy_matrix: the matrix indicating the best buddy of each piece, if any, as a numpy of shape (n, 4) containing tuples of form (piece_index, rotation_index)
	:param compatibility_scores: the matrix of all compatibility scores as a numpy array of shape (n, 4, n, 4), where n is the total number of patches
	:param best_neighbors: TODO
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
	buddy_counts = [sum([buddy_matrix[buddy_index, r] is not None for r in range(buddy_matrix.shape[1])]) for buddy_index in range(buddy_matrix.shape[0])]
	highest_buddy_count = np.amax(buddy_counts)
	if len(candidates) == 0:
		print("NO PIECE HAS ENOUGH BEST BUDDIES 2 LAYERS DEEP")
		x = np.argwhere(buddy_counts == highest_buddy_count)
		candidates = x.flatten().tolist()
	if len(candidates) == 1:
		return candidates[0]
	if highest_buddy_count != 0:  # pick the piece that has the best sum of mutual compatibility scores with its best buddies
		matrix_to_use = buddy_matrix
	else:  # pick the piece that has the best sum of mutual compatibility scores with its best-scoring neighbors
		matrix_to_use = best_neighbors
	best_score = -INFINITY
	best_index = -1
	for i in candidates:
		# sum its mutual compatibility scores
		mutual_compatibility_sum = 0.0
		for r1 in range(matrix_to_use.shape[1]):
			if matrix_to_use[i, r1] is None:
				continue
			j, r2 = matrix_to_use[i, r1]
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
	print("computing 3rd pixel predictions...")
	predictions_matrix = get_3rd_pixel_predictions(patches)
	print("computing dissimilarity scores...")
	dissimilarity_scores = get_dissimilarity_scores(patches, predictions_matrix)
	print("computing best neighbors...")
	best_neighbors = get_best_neighbors(dissimilarity_scores)
	print("computing initial compatibility scores...")
	compatibility_scores = get_compatibility_scores(dissimilarity_scores, best_neighbors)
	print("finding initial best buddies...")
	buddy_matrix = get_best_buddies(compatibility_scores)
	print("selecting first piece...")
	first_piece = pick_first_piece(buddy_matrix, compatibility_scores, best_neighbors)
	print(f"first piece selected: {first_piece}")
	# TODO: actually implement the body of the algorithm
	return None
