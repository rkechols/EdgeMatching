# based on the paper by Genady Paikin and Ayellet Tal, found at http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7299116


import multiprocessing as mp
from typing import List, Tuple
import numpy as np
from constants import INFINITY, NO_PIECE, EXPANSION_SPACE, YES_PIECE
from tqdm import tqdm
from functions import rgb_to_lab


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


def predict_3rd_pixel_mp(coord_last_columns: (Tuple[int, int], np.ndarray, np.ndarray)) -> (Tuple[int, int], np.ndarray):
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


def get_3rd_pixel_predictions(patches: List[np.ndarray]) -> np.ndarray:
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


def norm_l1_mp(coord_and_columns: (Tuple[int, int, int], np.ndarray, np.ndarray)) -> (Tuple[int, int, int], float):
	# for use with the multiprocessing library
	t, predicted, actual, = coord_and_columns
	if t[0] == t[2]:  # same patch
		score = INFINITY
	else:
		score = norm_l1(predicted, actual)
	return t, score


# generates input for dissimilarity_score_pt_mp()
class DissimilarityScorePtMpGenerator:
	def __init__(self, patches: List[np.ndarray], predictions_matrix: np.ndarray, rotations_shuffled: bool = True):
		self.patches = patches
		self.predictions = predictions_matrix
		self.rotations_shuffled = rotations_shuffled

	def __iter__(self):
		for patch1_index, patch1 in enumerate(self.patches):
			for r1 in range(4):
				predicted_column = self.predictions[patch1_index, r1]
				for patch2_index, patch2 in enumerate(self.patches):
					if self.rotations_shuffled:
						for r2 in range(4):
							patch2_rotated = np.rot90(patch2, r2)
							actual_column = patch2_rotated[:, 0, :]
							yield (patch1_index, r1, patch2_index, r2), predicted_column, actual_column
					else:
						yield (patch1_index, r1, patch2_index), predicted_column, np.rot90(patch2, r1)[:, 0, :]

	def __len__(self):
		n = len(self.patches)
		if self.rotations_shuffled:
			return n * 4 * n * 4
		else:
			return n * 4 * n


def get_dissimilarity_scores(patches: List[np.ndarray], predictions_matrix: np.ndarray, rotations_shuffled: bool = True) -> np.ndarray:
	"""
	takes a list of scrambled patches and creates a matrix that gives dissimilarity scores to each possible pairing of patches
	:param patches: list of square numpy arrays, each of the same shape (x, x, 3); the list's length is referred to as n
	:param predictions_matrix: TODO
	:param rotations_shuffled: indicates if the patches have been rotated randomly (vs all being rotated correctly to start with)
	:return: a numpy array of shape (n, 4, n, 4). the value at [i, r1, j, r2] indicates how costly pairing patch i and patch j is when they are rotated r1 and r2 times, respectively
	low values mean they are a good pairing; high values mean they are a bad pairing
	"""
	n = len(patches)
	if rotations_shuffled:
		score_matrix = np.empty((n, 4, n, 4), dtype=float)
		with mp.Pool() as pool:
			gen = DissimilarityScorePtMpGenerator(patches, predictions_matrix, rotations_shuffled)
			result_generator = pool.imap_unordered(norm_l1_mp, gen)
			with tqdm(total=len(gen)) as progress_bar:
				for (patch1_index, r1, patch2_index, r2), score in result_generator:
					score_matrix[patch1_index, r1, patch2_index, r2] = score
					progress_bar.update()
	else:
		score_matrix = np.empty((n, 4, n), dtype=float)
		with mp.Pool() as pool:
			gen = DissimilarityScorePtMpGenerator(patches, predictions_matrix, rotations_shuffled)
			result_generator = pool.imap_unordered(norm_l1_mp, gen)
			with tqdm(total=len(gen)) as progress_bar:
				for (patch1_index, r1, patch2_index), score in result_generator:
					score_matrix[patch1_index, r1, patch2_index] = score
					progress_bar.update()
	return score_matrix


def get_best_neighbors(dissimilarity_scores: np.ndarray, rotations_shuffled: bool = True) -> np.ndarray:
	"""
	TODO
	:param dissimilarity_scores:
	:param rotations_shuffled: indicates if the patches have been rotated randomly (vs all being rotated correctly to start with)
	:return:
	"""
	n = dissimilarity_scores.shape[0]
	best_neighbors = np.empty((n, 4), dtype=tuple)
	for i in range(n):
		for r1 in range(dissimilarity_scores.shape[1]):
			if rotations_shuffled:
				relevant_section = dissimilarity_scores[i, r1, :, :]
				best_dissimilarity = np.where(relevant_section == np.amin(relevant_section))
				best_neighbors[i, r1] = list(zip(*best_dissimilarity))[0]  # just take the first if there's a tie
			else:
				relevant_section = dissimilarity_scores[i, r1, :]
				best_dissimilarity = np.where(relevant_section == np.amin(relevant_section))
				best_neighbors[i, r1] = best_dissimilarity[0][0]  # just take the first if there's a tie
	return best_neighbors


def compatibility_score(dissimilarity_scores: np.ndarray, best_neighbors: np.ndarray, patch_index: int, r: int, rotations_shuffled: bool = True) -> np.ndarray:
	"""
	TODO
	:param dissimilarity_scores:
	:param best_neighbors:
	:param patch_index:
	:param r:
	:param rotations_shuffled: indicates if the patches have been rotated randomly (vs all being rotated correctly to start with)
	:return:
	"""
	if rotations_shuffled:
		relevant_slice = dissimilarity_scores[patch_index, r, :, :]
		scores_to_return = np.empty_like(relevant_slice)
		best_d_j, best_d_r = best_neighbors[patch_index, r]
		best_d_score = dissimilarity_scores[patch_index, r, best_d_j, best_d_r]
		next_best_d_score = np.amin(
			relevant_slice[relevant_slice != best_d_score])  # if there is no second-best, this will raise a ValueError
		for patch_index2 in range(scores_to_return.shape[0]):
			for r2 in range(scores_to_return.shape[1]):
				if next_best_d_score != 0:
					scores_to_return[patch_index2, r2] = 1 - (relevant_slice[patch_index2, r2] / next_best_d_score)
				else:
					scores_to_return[patch_index2, r2] = 0.0
	else:
		relevant_slice = dissimilarity_scores[patch_index, r, :]
		scores_to_return = np.empty_like(relevant_slice)
		best_d_j = best_neighbors[patch_index, r]
		best_d_score = dissimilarity_scores[patch_index, r, best_d_j]
		next_best_d_score = np.amin(
			relevant_slice[relevant_slice != best_d_score])  # if there is no second-best, this will raise a ValueError
		for patch_index2 in range(scores_to_return.shape[0]):
			if next_best_d_score != 0:
				scores_to_return[patch_index2] = 1 - (relevant_slice[patch_index2] / next_best_d_score)
			else:
				scores_to_return[patch_index2] = 0.0
	return scores_to_return


def compatibility_score_mp(coord_and_dissimilarities: (tuple, np.ndarray, np.ndarray, bool)) -> (tuple, np.ndarray):
	# for use with the multiprocessing library
	(patch_index, r), dissimilarity_scores, best_neighbors, rotations_shuffled = coord_and_dissimilarities
	compatibility_scores = compatibility_score(dissimilarity_scores, best_neighbors, patch_index, r, rotations_shuffled)
	return (patch_index, r), compatibility_scores


# generates input for compatibility_score_mp()
class CompatibilityScoreMpGenerator:
	def __init__(self, dissimilarity_scores: np.ndarray, best_neighbors: np.ndarray, rotations_shuffled: bool = True):
		self.n = dissimilarity_scores.shape[0]
		self.d = dissimilarity_scores
		self.b = best_neighbors
		self.rotations_shuffled = rotations_shuffled

	def __iter__(self):
		for patch_index in range(self.n):
			for r in range(4):
				yield (patch_index, r), self.d, self.b, self.rotations_shuffled

	def __len__(self):
		return self.n * 4


def get_compatibility_scores(dissimilarity_scores: np.ndarray, best_neighbors: np.ndarray, rotations_shuffled: bool = True) -> np.ndarray:
	"""
	takes a matrix of all dissimilarity scores and gives a matrix of compatibility scores
	:param dissimilarity_scores: the matrix of all dissimilarity scores as a numpy array of shape (n, 4, n[, 4]), where n is the total number of patches
	:param best_neighbors: TODO
	:param rotations_shuffled: indicates if the patches have been rotated randomly (vs all being rotated correctly to start with)
	:return: TODO
	low values mean they are a bad pairing; high values mean they are a good pairing
	"""
	n = dissimilarity_scores.shape[0]
	if rotations_shuffled:
		score_matrix = np.empty((n, 4, n, 4), dtype=float)
		with mp.Pool() as pool:
			gen = CompatibilityScoreMpGenerator(dissimilarity_scores, best_neighbors, rotations_shuffled)
			result_generator = pool.imap_unordered(compatibility_score_mp, gen)
			with tqdm(total=len(gen)) as progress_bar:
				for (patch1_index, r1), scores_section in result_generator:
					score_matrix[patch1_index, r1, :, :] = scores_section
					progress_bar.update()
	else:
		score_matrix = np.empty((n, 4, n), dtype=float)
		with mp.Pool() as pool:
			gen = CompatibilityScoreMpGenerator(dissimilarity_scores, best_neighbors, rotations_shuffled)
			result_generator = pool.imap_unordered(compatibility_score_mp, gen)
			with tqdm(total=len(gen)) as progress_bar:
				for (patch1_index, r1), scores_section in result_generator:
					score_matrix[patch1_index, r1, :] = scores_section
					progress_bar.update()
	return score_matrix


def get_best_buddies(compatibility_scores: np.ndarray, rotations_shuffled: bool = True) -> np.ndarray:
	"""
	takes a matrix of compatibility scores and gives a matrix of best buddies
	:param compatibility_scores: TODO
	:param rotations_shuffled: indicates if the patches have been rotated randomly (vs all being rotated correctly to start with)
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
			if rotations_shuffled:
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
			else:  # no rotations
				relevant_section = compatibility_scores[i, r1, :]
				# find the best compatibility score
				best_compatibility = np.where(relevant_section == np.amax(relevant_section))
				best_compatibility = best_compatibility[0]
				found_buddy = False
				for j in best_compatibility:  # there might be ties; try all of them
					# see if it's "mutual"; if it is, add it to buddy_matrix
					relevant_section_back = compatibility_scores[j, r1_inverse, :]
					best_compatibility_back = np.where(relevant_section_back == np.amax(relevant_section_back))
					best_compatibility_back = best_compatibility_back[0]
					for back_i in best_compatibility_back:
						if i == back_i:
							buddy_matrix[i, r1] = j
							found_buddy = True
							break
					if found_buddy:
						break
				if not found_buddy:  # this one has no best buddy :(
					buddy_matrix[i, r1] = None
	return buddy_matrix


def pick_first_piece(buddy_matrix: np.ndarray, compatibility_scores: np.ndarray, best_neighbors: np.ndarray, rotations_shuffled: bool = True) -> int:
	"""
	takes info about best buddies and compatibility scores and selects the first piece to be placed
	:param buddy_matrix: the matrix indicating the best buddy of each piece, if any, as a numpy of shape (n, 4) containing tuples of form (piece_index, rotation_index)
	:param compatibility_scores: the matrix of all compatibility scores as a numpy array of shape (n, 4, n[, 4)], where n is the total number of patches
	:param best_neighbors: TODO
	:param rotations_shuffled: indicates if the patches have been rotated randomly (vs all being rotated correctly to start with)
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
			if rotations_shuffled:
				buddy_index, _ = buddy_matrix[i, r1]
			else:
				buddy_index = buddy_matrix[i, r1]
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
			if rotations_shuffled:
				j, r2 = matrix_to_use[i, r1]
				compatibility1 = compatibility_scores[i, r1, j, r2]
				compatibility2 = compatibility_scores[j, (r2 + 2) % 4, i, (r1 + 2) % 4]
				mutual_compatibility_sum += (compatibility1 + compatibility2) / 2.0
			else:
				j = matrix_to_use[i, r1]
				compatibility1 = compatibility_scores[i, r1, j]
				compatibility2 = compatibility_scores[j, (r1 + 2) % 4, i]
				mutual_compatibility_sum += (compatibility1 + compatibility2) / 2.0
		# check if this score is better than any others we've seen
		if mutual_compatibility_sum > best_score:
			best_score = mutual_compatibility_sum
			best_index = i
	return best_index


class PoolCandidate:
	def __init__(self, score: float, index: int, row: int, col: int):
		self.score = score
		self.index = index
		self.row = row
		self.col = col

	def __lt__(self, other):
		if not isinstance(other, PoolCandidate):
			raise ValueError("class PoolCandidate cannot be compared to an object of any other type")
		return self.score > other.score  # reversed for a MAX heap


def solve_puzzle(patches: List[np.ndarray], first_piece: int, dissimilarity_scores: np.ndarray, compatibility_scores: np.ndarray, buddy_matrix: np.ndarray, rotations_shuffled: bool) -> np.ndarray:
	"""
	runs the main loop of greedy placement
	:param patches: list of numpy arrays that are the actual patches (puzzle pieces)
	:param first_piece: index/id of the patch that is to be placed first
	:param dissimilarity_scores: array of dissimilarity scores between each pair of patches at a given orientation
	:param compatibility_scores: array of compatibility/confidence scores between each pair of patches at a given orientation
	:param buddy_matrix: array of which patches are best buddies (mutual best neighbors) at a given orientation
	:param rotations_shuffled: indicates if patches have been rotated as part of the scramble
	:return: reconstruction matrix, shape (row, col, 2) where at a given location index 0 gives the patch index/id, and index 1 gives its rotation
	"""
	# need to add first piece to the puzzle
	# need to correctly make the new puzzle (the one we're going to add pieces to one piece at a time) with the right dimensions etc.
	# need to make a potential pool which adds all the best buddies of the last piece placed
	# then edge with best compatibility score is added to the puzzle
	# construction_matrix needs to then be updated and reconstruction_matrix may need to have size updated
	# if the pool is empty, we have to re score best buddy matrix and exclude pieces that have already been placed
	# continue the process until all pieces have been placed
	print(patches)

	preference_pool = list()  # max heap of PoolCandidates
	pieces_placed = set()

	best_first_piece = buddy_matrix[first_piece]
	# todo addBuddies of first piece
	pieces_remaining = buddy_matrix.shape[0]
	construction_matrix = np.array([[NO_PIECE, EXPANSION_SPACE, NO_PIECE],
									[EXPANSION_SPACE, YES_PIECE, EXPANSION_SPACE],
									[NO_PIECE, EXPANSION_SPACE, NO_PIECE]])
	reconstruction_matrix = np.zeros((3, 3, 2), dtype=int)
	reconstruction_matrix[1][1] = [first_piece, 0]  # Add the first piece, 0 refers to the rotation
	pieces_remaining -= 1

	while pieces_remaining > 0:
		if len(preference_pool) > 0:
			# todo: get next piece out of pool, find if canAddPiece
			piece_index = 0
			row, col = 0, 0
			# place piece found
			reconstruction_matrix[row, col] = [piece_index, 0]
			adjust_matrices(row, col, reconstruction_matrix, construction_matrix, preference_pool)
			pieces_placed.add(piece_index)
			pieces_remaining -= 1

		# todo: put buddies into the pool buddies3
		else:  # pool is empty
			# recalculate the compatibility function
			# find the best neighbors (not best buddies)
			best_neighbors = get_best_neighbors(dissimilarity_scores, rotations_shuffled)
			compatibility_scores = get_compatibility_scores(dissimilarity_scores, best_neighbors, rotations_shuffled)
			buddy_matrix = get_best_buddies(compatibility_scores, rotations_shuffled)
	# do something here??

	# todo: trim matrices at end?
	return reconstruction_matrix


def add_buddies_to_pool(placed_piece: int, check_mutuality: bool, check_cycles: bool, preference_pool: List[PoolCandidate]):
	# adds buddies of a piece into the pool

	numBestNeighborRelationshipsOfNext = 0  # how many best neighbor relationships there are (out of 8) if we place nextA in this location

	w1 = 0.5
	w2 = 0.5

	# todo: Need to use the actual indices for the construction matrix and best buddies

	if (construction_matrix[nextRow - 1 + originIndex][nextCol + originIndex] == YES_PIECE and
			bestBuddies[construction_matrix[nextRow - 1 + originIndex][nextCol + originIndex] - 1][
				1] == next):  # theres a piece above and is best neighbor up
		numBestNeighborRelationshipsOfNext += 1

	if (construction_matrix[nextRow + 1 + originIndex][nextCol + originIndex] == YES_PIECE and bestBuddies[next][1] ==
			construction_matrix[nextRow + 1 + originIndex][nextCol + originIndex] - 1):
		numBestNeighborRelationshipsOfNext += 1

	if (construction_matrix[nextRow + originIndex][nextCol - 1 + originIndex] == YES_PIECE and
			bestBuddies[construction_matrix[nextRow + originIndex][nextCol - 1 + originIndex] - 1][3] == next):
		numBestNeighborRelationshipsOfNext += 1

	if (construction_matrix[nextRow + originIndex][nextCol + 1 + originIndex] == YES_PIECE and bestBuddies[next][3] ==
			construction_matrix[nextRow + originIndex][nextCol + 1 + originIndex] - 1):
		numBestNeighborRelationshipsOfNext += 1

	if (construction_matrix[nextRow + 1 + originIndex][nextCol + originIndex] == YES_PIECE and
			bestBuddies[construction_matrix[nextRow + 1 + originIndex][nextCol + originIndex] - 1][0] == next):
		numBestNeighborRelationshipsOfNext += 1

	if (construction_matrix[nextRow - 1 + originIndex][nextCol + originIndex] == YES_PIECE and bestBuddies[next][0] ==
			construction_matrix[nextRow - 1 + originIndex][
				nextCol + originIndex] - 1):  # there is piece above and is best neighbor down
		numBestNeighborRelationshipsOfNext += 1

	if (construction_matrix[nextRow + originIndex][nextCol + 1 + originIndex] == YES_PIECE and
			bestBuddies[construction_matrix[nextRow + originIndex][nextCol + 1 + originIndex] - 1][2] == next):
		numBestNeighborRelationshipsOfNext += 1

	if (construction_matrix[nextRow + originIndex][nextCol - 1 + originIndex] == YES_PIECE and bestBuddies[next][2] ==
			construction_matrix[nextRow + originIndex][nextCol - 1 + originIndex] - 1):
		numBestNeighborRelationshipsOfNext += 1

	sumMutualCompatibility = 0  # aggregates a compatibility of potential neighbors
	numPlacedNeighborsOfNext = 0  # counts how many neighbors it would have if placed
	if (construction_matrix[nextRow - 1 + originIndex][nextCol + originIndex] == YES_PIECE):  # if there is a piece above nextA
		sumMutualCompatibility += w1 * confMatrix[1][construction_matrix[nextRow - 1 + originIndex][nextCol + originIndex] - 1][
		next] + w2 * confMatrix[0][next][construction_matrix[nextRow - 1 + originIndex][nextCol + originIndex] - 1]
		# w1 * confidence score FROM piece above TO nextA oriented down (1) # w2 * confidence score TO piece above FROM nextA oriented up (0)
		numPlacedNeighborsOfNext += 1

	if (construction_matrix[nextRow + 1 + originIndex][nextCol + originIndex] == YES_PIECE):
		sumMutualCompatibility += w1 * confMatrix[0][construction_matrix[nextRow + 1 + originIndex][nextCol + originIndex] - 1][
			next] + w2 * confMatrix[1][next][construction_matrix[nextRow + 1 + originIndex][nextCol + originIndex] - 1]
		numPlacedNeighborsOfNext += 1

	if (construction_matrix[nextRow + originIndex][nextCol - 1 + originIndex] == YES_PIECE):
		sumMutualCompatibility += w1 * confMatrix[3][construction_matrix[nextRow + originIndex][nextCol - 1 + originIndex] - 1][
			next] + w2 * confMatrix[2][next][construction_matrix[nextRow + originIndex][nextCol - 1 + originIndex] - 1]
		numPlacedNeighborsOfNext += 1

	if (construction_matrix[nextRow + originIndex][nextCol + 1 + originIndex] == YES_PIECE):
		sumMutualCompatibility += w1 * confMatrix[2][construction_matrix[nextRow + originIndex][nextCol + 1 + originIndex] - 1][
			next] + w2 * confMatrix[3][next][construction_matrix[nextRow + originIndex][nextCol + 1 + originIndex] - 1]
		numPlacedNeighborsOfNext += 1

	cycle_bonus = 0.0

	if (numPlacedNeighborsOfNext == 1 and
			(bestBuddies[next][dir[0]] != -1
			 and bestBuddies[bestBuddies[next][dir[0]]][dir[1]] != -1
			 and bestBuddies[bestBuddies[bestBuddies[next][dir[0]]][dir[1]]][dir[2]] == placed_piece)):
		cycle_bonus += 0.25

	if (clockwiseOfNext != null and numPlacedNeighborsOfNext == 1 and  # if next has ONE placed clockwise neighbor
			(bestBuddies[next][dir[0]] != -1 and  # and has a best neighbor in that direction
			 bestBuddies[bestBuddies[next][dir[0]]][dir[1]] == clockwiseOfNext)):  # ????
		cycle_bonus += 0.25
		print("???")

	if (numPlacedNeighborsOfNext == 1 and
			(bestBuddies[next][dir[2]] != -1
			 and bestBuddies[bestBuddies[next][dir[2]]][dir[1]] != -1
			 and bestBuddies[bestBuddies[bestBuddies[next][dir[2]]][dir[1]]][dir[0]] == placed_piece)):
		cycle_bonus += 0.25

	if (counterClockwiseOfNext != null and numPlacedNeighborsOfNext == 1 and
			(bestBuddies[next][dir[2]] != -1 and
			 bestBuddies[bestBuddies[next][dir[2]]][dir[1]] == counterClockwiseOfNext)):
		cycle_bonus += 0.25

	bestNeighborBonus = max((numBestNeighborRelationshipsOfNext - 2),
								 0)  # if 0,1,or 2 neighbors result is 0, else if 3-8 neighbors result is 1-6
	neighborCountBonus = max((numPlacedNeighborsOfNext - 2),
								  0)  # if 0,1,or 2 neighbors result is 0, else if 3 or 4 neighbors result is 1 or 2
	nextA = new
	nextA[0] = next
	nextA[1] = nextCol
	nextA[2] = nextRow
	if (construction_matrix[nextRow + originIndex][nextCol + originIndex] == EXPANSION_SPACE):  # if spot is available
		# calculate the score if we were to place it there
		placementScore = sumMutualCompatibility / numPlacedNeighborsOfNext

		if (check_cycles):
			placementScore += cycle_bonus
			placementScore += bestNeighborBonus * 0.2

		elif not check_mutuality:  # *does* get a bonus for having more neighbors
			placementScore += numPlacedNeighborsOfNext * 0.5


		# put it in the pool
		placementScoreToPiece.put(placementScore, nextA)

	pass


def adjust_matrices(row, col, reconstruction_matrix, construction_matrix, pool):
	construction_matrix[row][col] = YES_PIECE
	# resize reconstruction and construction matrices
	if row == 0 or row == construction_matrix.shape[0] - 1:
		new_scores_row = np.zeros((1, reconstruction_matrix.shape[1], 2), dtype=int)
		new_construction_row = np.empty((1, construction_matrix.shape[1]), dtype=int)
		new_construction_row[:, :] = NO_PIECE
		if row == 0:  # we placed one on the top row
			for piece in pool:
				piece.row += 1
			reconstruction_matrix = np.concatenate((new_scores_row, reconstruction_matrix), axis=0)
			construction_matrix = np.concatenate((new_construction_row, construction_matrix), axis=0)
			row += 1
		else:  # we placed one on the bottom row
			reconstruction_matrix = np.concatenate((reconstruction_matrix, new_scores_row), axis=0)
			construction_matrix = np.concatenate((construction_matrix, new_construction_row), axis=0)
	if col == 0 or col == construction_matrix.shape[1] - 1:
		new_scores_col = np.zeros((reconstruction_matrix.shape[0], 1, 2), dtype=int)
		new_construction_col = np.empty((construction_matrix.shape[0], 1), dtype=int)
		new_construction_col[:, :] = NO_PIECE
		if col == 0:  # we placed one on the left column
			for piece in pool:
				piece.col += 1
			reconstruction_matrix = np.concatenate((new_scores_col, reconstruction_matrix), axis=1)
			construction_matrix = np.concatenate((new_construction_col, construction_matrix), axis=1)
			col += 1
		else:  # we placed one on the bottom row
			reconstruction_matrix = np.concatenate((reconstruction_matrix, new_scores_col), axis=1)
			construction_matrix = np.concatenate((construction_matrix, new_construction_col), axis=1)
	# check and update the neighbors of the newly placed patch
	for row_shift in [-1, 1]:
		if construction_matrix[row + row_shift, col] == NO_PIECE:
			construction_matrix[row + row_shift, col] = EXPANSION_SPACE
	for col_shift in [-1, 1]:
		if construction_matrix[row, col + col_shift] == NO_PIECE:
			construction_matrix[row, col + col_shift] = EXPANSION_SPACE


def jigsaw_pt(patches: List[np.ndarray], rotations_shuffled: bool = True):
	"""
	takes a list of square patches and uses Paikin and Tal's algorithm to assemble the patches
	:param patches: a list of numpy arrays representing the scrambled patches of the original image (RGB)
	:param rotations_shuffled: indicates if the patches have been rotated randomly (vs all being rotated correctly to start with)
	:return: the re-assembled image as a numpy array of shape (r, c, 3)
	"""
	patches_lab = [rgb_to_lab(p) for p in patches]
	print("computing 3rd pixel predictions...")
	predictions_matrix = get_3rd_pixel_predictions(patches_lab)
	print("computing dissimilarity scores...")
	dissimilarity_scores = get_dissimilarity_scores(patches_lab, predictions_matrix, rotations_shuffled)
	print("computing best neighbors...")
	best_neighbors = get_best_neighbors(dissimilarity_scores, rotations_shuffled)
	print("computing initial compatibility scores...")
	compatibility_scores = get_compatibility_scores(dissimilarity_scores, best_neighbors, rotations_shuffled)
	print("finding initial best buddies...")
	buddy_matrix = get_best_buddies(compatibility_scores, rotations_shuffled)
	total_slots = buddy_matrix.shape[0] * buddy_matrix.shape[1]
	buddy_count = 0
	for i in range(buddy_matrix.shape[0]):
		for j in range(buddy_matrix.shape[1]):
			if buddy_matrix[i, j] is not None:
				buddy_count += 1
	print(f"best buddies count: {buddy_count} / {total_slots}")
	print("selecting first piece...")
	first_piece = pick_first_piece(buddy_matrix, compatibility_scores, best_neighbors, rotations_shuffled)
	print(f"first piece selected: {first_piece}")
	# TODO: actually implement the body of the algorithm (added solve_puzzle function for this below)
	reconstruction_matrix = solve_puzzle(patches_lab, first_piece, dissimilarity_scores, compatibility_scores, buddy_matrix, rotations_shuffled)
	return reconstruction_matrix
