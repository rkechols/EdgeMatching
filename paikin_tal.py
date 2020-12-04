# based on the paper by Genady Paikin and Ayellet Tal, found at http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7299116

import heapq
import multiprocessing as mp
from typing import List, Set, Tuple
import numpy as np
from constants import INFINITY, NO_PIECE, EXPANSION_SPACE, YES_PIECE, ROTATION_0, ROTATION_180, ROTATION_90, ROTATION_270
from tqdm import tqdm
from functions import rgb_to_lab, rgb2lab


def predict_3rd_pixel(col1: np.ndarray, col2: np.ndarray, use_lab_color: bool) -> np.ndarray:
	"""
	takes two columns (or rows) of pixels and predicts what the next column (or row) of pixels would be
	:param col1: the first list of pixels as a numpy array of shape (x, 3)
	:param col2: the second list of pixels as a numpy array of shape (x, 3)
	:param use_lab_color: indicates if LAB color space is being used (if `False`, then RGB is being used)
	:return: the predicted third list of pixels as a numpy array of shape (x, 3)
	"""
	diff_col = col2 - col1
	expected = col2 + diff_col
	if not use_lab_color:  # RGB needs to be clipped; restrict to [0, 255]
		coords = np.where(expected < 0)
		for row, channel in zip(*coords):
			expected[row, channel] = 0
		coords = np.where(expected > 255)
		for row, channel in zip(*coords):
			expected[row, channel] = 255
	return expected


def predict_3rd_pixel_mp(coord_last_columns: (Tuple[int, int], np.ndarray, np.ndarray, bool)) -> (Tuple[int, int], np.ndarray):
	coord, col1, col2, use_lab_color = coord_last_columns
	return coord, predict_3rd_pixel(col1, col2, use_lab_color)


# generates input for predict_3rd_pixel_mp()
class Predict3rdPixelMpGenerator:
	def __init__(self, patches: List[np.ndarray], use_lab_color: bool):
		self.patches = patches
		self.use_lab_color = use_lab_color

	def __iter__(self):
		for patch_index, patch in enumerate(self.patches):
			for r in range(4):
				patch_rotated = np.rot90(patch, r)
				yield (patch_index, r), patch_rotated[:, -2, :], patch_rotated[:, -1, :], self.use_lab_color

	def __len__(self):
		n = len(self.patches)
		return n * 4


def norm_l1(col1: np.ndarray, col2: np.ndarray) -> float:
	"""
	takes two columns (or rows) of pixels and gives the L_1 norm of the their difference
	:param col1: the first list of pixels as a numpy array of shape (x, 3)
	:param col2: the second list of pixels as a numpy array of shape (x, 3)
	:return: the L_1 norm
	"""
	col1 = col1.astype(int)
	col2 = col2.astype(int)
	diff_col = abs(col1 - col2)
	return float(sum(diff_col.flatten()))


def norm_l2(col1: np.ndarray, col2: np.ndarray) -> float:
	"""
	TODO
	:param col1:
	:param col2:
	:return:
	"""
	diffs = (col1 - col2) ** 2
	return float(diffs.sum())


def norm_l1_mp(coord_and_columns: (Tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray)) -> (Tuple[int, int, int], float):
	# for use with the multiprocessing library
	t, predicted, actual, color1, color2 = coord_and_columns
	if t[0] == t[2]:  # same patch
		score = INFINITY
	else:
		color_distance = norm_l2(color1, color2)
		if color_distance > 300:
			score = INFINITY
		else:
			score = norm_l1(predicted, actual)
	return t, score


# generates input for dissimilarity_score_pt_mp()
class DissimilarityScorePtMpGenerator:
	def __init__(self, patches: List[np.ndarray], predictions_matrix: np.ndarray, average_color_matrix: np.ndarray, rotations_shuffled: bool = True):
		self.patches = patches
		self.predictions = predictions_matrix
		self.colors = average_color_matrix
		self.rotations_shuffled = rotations_shuffled

	def __iter__(self):
		for patch1_index, patch1 in enumerate(self.patches):
			for r1 in range(4):
				predicted_column = self.predictions[patch1_index, r1]
				color1 = self.colors[patch1_index, r1]
				for patch2_index, patch2 in enumerate(self.patches):
					if self.rotations_shuffled:
						for r2 in range(4):
							patch2_rotated = np.rot90(patch2, r2)
							actual_column = patch2_rotated[:, 0, :]
							color2 = self.colors[patch2_index, r2]
							yield (patch1_index, r1, patch2_index, r2), predicted_column, actual_column, color1, color2
					else:
						r2 = (r1 + 2) % 4
						color2 = self.colors[patch2_index, r2]
						yield (patch1_index, r1, patch2_index), predicted_column, np.rot90(patch2, r1)[:, 0, :], color1, color2

	def __len__(self):
		n = len(self.patches)
		if self.rotations_shuffled:
			return n * 4 * n * 4
		else:
			return n * 4 * n


def get_best_neighbors(scores_matrix: np.ndarray, rotations_shuffled: bool = True) -> np.ndarray:
	"""
	TODO
	:param scores_matrix:
	:param rotations_shuffled: indicates if the patches have been rotated randomly (vs all being rotated correctly to start with)
	:return:
	"""
	n = scores_matrix.shape[0]
	if rotations_shuffled:
		d_type = tuple
	else:
		d_type = int
	best_neighbors = np.empty((n, 4), dtype=d_type)
	for i in range(n):
		for r1 in range(scores_matrix.shape[1]):
			if rotations_shuffled:
				relevant_section = scores_matrix[i, r1, :, :]
				best_dissimilarity = np.where(relevant_section == np.amin(relevant_section))
				best_neighbors[i, r1] = list(zip(*best_dissimilarity))[0]  # just take the first if there's a tie
			else:
				relevant_section = scores_matrix[i, r1, :]
				best_dissimilarity = np.where(relevant_section == np.amin(relevant_section))
				best_neighbors[i, r1] = best_dissimilarity[0][0]  # just take the first if there's a tie
	return best_neighbors


def calc_compatibility_score_row(dissimilarity_scores: np.ndarray, best_neighbors_dissimilarity: np.ndarray, patch_index: int, r: int, rotations_shuffled: bool = True) -> np.ndarray:
	"""
	TODO
	:param dissimilarity_scores:
	:param best_neighbors_dissimilarity:
	:param patch_index:
	:param r:
	:param rotations_shuffled: indicates if the patches have been rotated randomly (vs all being rotated correctly to start with)
	:return:
	"""
	if rotations_shuffled:
		relevant_slice = dissimilarity_scores[patch_index, r, :, :]
		scores_to_return = np.empty_like(relevant_slice, dtype=np.float)
		best_d_j, best_d_r = best_neighbors_dissimilarity[patch_index, r]
		best_d_score = dissimilarity_scores[patch_index, r, best_d_j, best_d_r]
		next_best_d_score = np.amin(
			relevant_slice[relevant_slice != best_d_score])  # if there is no second-best, this will raise a ValueError
		for patch_index2 in range(scores_to_return.shape[0]):
			for r2 in range(scores_to_return.shape[1]):
				if next_best_d_score != 0:
					if relevant_slice[patch_index2, r2] == INFINITY and next_best_d_score == INFINITY:
						scores_to_return[patch_index2] = 0.0
					else:
						scores_to_return[patch_index2, r2] = 1.0 - (relevant_slice[patch_index2, r2] / next_best_d_score)
				else:
					scores_to_return[patch_index2, r2] = 0.001
	else:
		relevant_slice = np.copy(dissimilarity_scores[patch_index, r, :])
		scores_to_return = np.empty_like(relevant_slice)
		best_d_j = best_neighbors_dissimilarity[patch_index, r]
		best_d_score = dissimilarity_scores[patch_index, r, best_d_j]
		relevant_slice[best_d_j] = INFINITY
		next_best_d_score = np.amin(relevant_slice)  # if there is no second-best, this will raise a ValueError
		relevant_slice[best_d_j] = best_d_score
		for patch_index2 in range(scores_to_return.shape[0]):
			if next_best_d_score != 0:
				if relevant_slice[patch_index2] == INFINITY and next_best_d_score == INFINITY:
					scores_to_return[patch_index2] = 0.0
				else:
					scores_to_return[patch_index2] = 1.0 - (relevant_slice[patch_index2] / next_best_d_score)
			else:
				scores_to_return[patch_index2] = 0.001
	return scores_to_return


def compatibility_score_mp(coord_and_dissimilarities: (Tuple[int, int], np.ndarray, np.ndarray, bool)) -> (Tuple[int, int], np.ndarray):
	# for use with the multiprocessing library
	(patch_index, r), dissimilarity_scores, best_neighbors_dissimilarity, rotations_shuffled = coord_and_dissimilarities
	compatibility_scores = calc_compatibility_score_row(dissimilarity_scores, best_neighbors_dissimilarity, patch_index, r, rotations_shuffled)
	return (patch_index, r), compatibility_scores


# generates input for compatibility_score_mp()
class CompatibilityScoreMpGenerator:
	def __init__(self, dissimilarity_scores: np.ndarray, best_neighbors_dissimilarity: np.ndarray, rotations_shuffled: bool = True):
		self.n = dissimilarity_scores.shape[0]
		self.d = dissimilarity_scores
		self.b = best_neighbors_dissimilarity
		self.rotations_shuffled = rotations_shuffled

	def __iter__(self):
		for patch_index in range(self.n):
			for r in range(4):
				yield (patch_index, r), self.d, self.b, self.rotations_shuffled

	def __len__(self):
		return self.n * 4


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


class PTSolver:
	def __init__(self, patches: List[np.ndarray], rotations_shuffled: bool = True, use_lab_color: bool = True):
		self.n = len(patches)
		self.patches = patches
		self.rotations_shuffled = rotations_shuffled
		self.use_lab_color = use_lab_color
		if self.use_lab_color:
			#self.patches = [rgb_to_lab(p) for p in self.patches]
			for i in range(len(self.patches)):
				for j in range(self.patches[i].shape[0]):
					for k in range(self.patches[i].shape[1]):
						R=self.patches[i][j][k][0]
						G=self.patches[i][j][k][1]
						B=self.patches[i][j][k][2]
						lab = rgb2lab(R,G,B)
						self.patches[i][j][k][0] = lab[0]
						self.patches[i][j][k][1] = lab[1]
						self.patches[i][j][k][2] = lab[2]



		self.prediction_matrix = np.empty((self.n, 4), dtype=np.ndarray)
		self.average_edge_colors = np.empty((self.n, 4, 3), dtype=np.float)  # 4 edges, 3 channels
		if self.rotations_shuffled:
			self.dissimilarity_scores = np.empty((self.n, 4, self.n, 4), dtype=np.float)
			self.compatibility_scores = np.empty((self.n, 4, self.n, 4), dtype=np.float)
		else:
			self.dissimilarity_scores = np.empty((self.n, 4, self.n), dtype=np.float)
			self.compatibility_scores = np.empty((self.n, 4, self.n), dtype=np.float)
		self.buddy_matrix = np.empty((self.n, 4), dtype=tuple)
		self.preference_pool: List[PoolCandidate] = list()  # max heap of PoolCandidates
		self.patches_placed: Set[int] = set()
		self.reconstruction_matrix = np.zeros((3, 3, 2), dtype=np.int)
		self.construction_matrix = np.array([
			[NO_PIECE, EXPANSION_SPACE, NO_PIECE],
			[EXPANSION_SPACE, YES_PIECE, EXPANSION_SPACE],
			[NO_PIECE, EXPANSION_SPACE, NO_PIECE]
		])

	def get_3rd_pixel_predictions(self):
		with mp.Pool() as pool:
			gen = Predict3rdPixelMpGenerator(self.patches, self.use_lab_color)
			result_generator = pool.imap_unordered(predict_3rd_pixel_mp, gen)
			with tqdm(total=len(gen)) as progress_bar:
				for (patch_index, r), predicted_column in result_generator:
					self.prediction_matrix[patch_index, r] = predicted_column
					progress_bar.update()

	def get_average_edge_colors(self):
		for i, patch in enumerate(self.patches):
			for r in range(4):
				pixels_row = np.rot90(patch, r)[:, -1]
				average_color = np.average(pixels_row, axis=0)
				self.average_edge_colors[i, r, :] = average_color

	def get_dissimilarity_scores(self):
		if self.rotations_shuffled:
			with mp.Pool() as pool:
				gen = DissimilarityScorePtMpGenerator(self.patches, self.prediction_matrix, self.average_edge_colors, self.rotations_shuffled)
				result_generator = pool.imap_unordered(norm_l1_mp, gen)
				with tqdm(total=len(gen)) as progress_bar:
					for (patch1_index, r1, patch2_index, r2), score in result_generator:
						self.dissimilarity_scores[patch1_index, r1, patch2_index, r2] = score
						progress_bar.update()
		else:
			with mp.Pool() as pool:
				gen = DissimilarityScorePtMpGenerator(self.patches, self.prediction_matrix, self.average_edge_colors, self.rotations_shuffled)
				result_generator = pool.imap_unordered(norm_l1_mp, gen)
				with tqdm(total=len(gen)) as progress_bar:
					for (patch1_index, r1, patch2_index), score in result_generator:
						self.dissimilarity_scores[patch1_index, r1, patch2_index] = score
						progress_bar.update()

	def get_compatibility_scores(self, best_neighbors_dissimilarity: np.ndarray):
		if self.rotations_shuffled:
			with mp.Pool() as pool:
				gen = CompatibilityScoreMpGenerator(self.dissimilarity_scores, best_neighbors_dissimilarity, self.rotations_shuffled)
				result_generator = pool.imap_unordered(compatibility_score_mp, gen)
				with tqdm(total=len(gen)) as progress_bar:
					for (patch1_index, r1), scores_section in result_generator:
						self.compatibility_scores[patch1_index, r1, :, :] = scores_section
						progress_bar.update()
		else:
			with mp.Pool() as pool:
				gen = CompatibilityScoreMpGenerator(self.dissimilarity_scores, best_neighbors_dissimilarity, self.rotations_shuffled)
				result_generator = pool.imap_unordered(compatibility_score_mp, gen)
				with tqdm(total=len(gen)) as progress_bar:
					for (patch1_index, r1), scores_section in result_generator:
						self.compatibility_scores[patch1_index, r1, :] = scores_section
						progress_bar.update()

	def get_best_buddies(self):
		# look at each piece in each rotation, see if its best neighbor is mutual
		for i in range(self.n):
			for r1 in range(self.compatibility_scores.shape[1]):
				r1_inverse = (r1 + 2) % 4
				if self.rotations_shuffled:
					relevant_section = self.compatibility_scores[i, r1, :, :]
					# find the best compatibility score
					best_compatibility = np.where(relevant_section == np.amax(relevant_section))
					best_compatibility = list(zip(*best_compatibility))
					found_buddy = False
					for j, r2 in best_compatibility:  # there might be ties; try all of them
						# see if it's "mutual"; if it is, add it to buddy_matrix
						r2_inverse = (r2 + 2) % 4
						relevant_section_back = self.compatibility_scores[j, r2_inverse, :, :]
						best_compatibility_back = np.where(relevant_section_back == np.amax(relevant_section_back))
						best_compatibility_back = list(zip(*best_compatibility_back))
						for back_i, back_r1 in best_compatibility_back:
							if i == back_i and r1_inverse == back_r1:
								self.buddy_matrix[i, r1] = (j, r2)
								found_buddy = True
								break
						if found_buddy:
							break
					if not found_buddy:  # this one has no best buddy :(
						self.buddy_matrix[i, r1] = None
				else:  # no rotations
					relevant_section = self.compatibility_scores[i, r1, :]
					# find the best compatibility score
					best_compatibility = np.where(relevant_section == np.amax(relevant_section))
					best_compatibility = best_compatibility[0]
					found_buddy = False
					for j in best_compatibility:  # there might be ties; try all of them
						# see if it's "mutual"; if it is, add it to buddy_matrix
						relevant_section_back = self.compatibility_scores[j, r1_inverse, :]
						best_compatibility_back = np.where(relevant_section_back == np.amax(relevant_section_back))
						best_compatibility_back = best_compatibility_back[0]
						for back_i in best_compatibility_back:
							if i == back_i:
								self.buddy_matrix[i, r1] = j
								found_buddy = True
								break
						if found_buddy:
							break
					if not found_buddy:  # this one has no best buddy :(
						self.buddy_matrix[i, r1] = None

	def pick_first_piece(self, best_neighbors_compatibility: np.ndarray) -> int:
		candidates = list()
		for i in range(self.n):
			buddy_count = sum([self.buddy_matrix[i, r] is not None for r in range(self.buddy_matrix.shape[1])])
			if buddy_count != 4:
				continue
			# check if its buddies each have 4 buddies
			passes = True
			for r1 in range(self.buddy_matrix.shape[1]):
				if self.rotations_shuffled:
					buddy_index, _ = self.buddy_matrix[i, r1]
				else:
					buddy_index = self.buddy_matrix[i, r1]
				buddy_count2 = sum([self.buddy_matrix[buddy_index, r] is not None for r in range(self.buddy_matrix.shape[1])])
				if buddy_count2 != 4:
					passes = False
					break
			if passes:
				candidates.append(i)
		buddy_counts = [sum([self.buddy_matrix[buddy_index, r] is not None for r in range(self.buddy_matrix.shape[1])]) for buddy_index in range(self.buddy_matrix.shape[0])]
		highest_buddy_count = np.amax(buddy_counts)
		if len(candidates) == 0:
			print("NO PIECE HAS ENOUGH BEST BUDDIES 2 LAYERS DEEP")
			x = np.argwhere(buddy_counts == highest_buddy_count)
			candidates = x.flatten().tolist()
		if len(candidates) == 1:
			return candidates[0]
		if highest_buddy_count != 0:  # pick the piece that has the best sum of mutual compatibility scores with its best buddies
			matrix_to_use = self.buddy_matrix
		else:  # pick the piece that has the best sum of mutual compatibility scores with its best-scoring neighbors
			matrix_to_use = best_neighbors_compatibility
		best_score = -INFINITY
		best_index = -1
		for i in candidates:
			# sum its mutual compatibility scores
			mutual_compatibility_sum = 0.0
			for r1 in range(matrix_to_use.shape[1]):
				if matrix_to_use[i, r1] is None:
					continue
				if self.rotations_shuffled:
					j, r2 = matrix_to_use[i, r1]
					compatibility1 = self.compatibility_scores[i, r1, j, r2]
					compatibility2 = self.compatibility_scores[j, (r2 + 2) % 4, i, (r1 + 2) % 4]
					mutual_compatibility_sum += (compatibility1 + compatibility2) / 2.0
				else:
					j = matrix_to_use[i, r1]
					compatibility1 = self.compatibility_scores[i, r1, j]
					compatibility2 = self.compatibility_scores[j, (r1 + 2) % 4, i]
					mutual_compatibility_sum += (compatibility1 + compatibility2) / 2.0
			# check if this score is better than any others we've seen
			if mutual_compatibility_sum > best_score:
				best_score = mutual_compatibility_sum
				best_index = i
		return best_index

	def add_buddies_to_pool(self, placed_piece: int, placed_row: int, placed_col: int, check_mutuality: bool, check_cycles: bool, best_neighbors: np.ndarray):
		# adds buddies of a piece into the preference_pool
		for r in range(4):
			neighbor = best_neighbors[placed_piece, r]
			if neighbor == -1:
				continue  # TODO: will best_neighbors ever have empty spots?

			op = (r + 2) % 4
			dir_a = np.zeros(3)

			# clockwise_of_next = neighbor 90 degrees clockwise
			# counter_clockwise_of_next = neighbor 90 degrees counterclockwise

			if (neighbor not in self.patches_placed) and (not check_mutuality or placed_piece == best_neighbors[neighbor, op]):
				if r == ROTATION_0:  # RIGHT
					neighbor_col = placed_col + 1
					neighbor_row = placed_row
					dir_a[0] = 1
					dir_a[1] = 2
					dir_a[2] = 0
					clockwise_of_next = self.reconstruction_matrix[neighbor_row + 1, neighbor_col]
					counter_clockwise_of_next = self.reconstruction_matrix[neighbor_row - 1, neighbor_col]

				elif r == ROTATION_90:  # UP
					neighbor_col = placed_col  # col
					neighbor_row = placed_row - 1  # row
					dir_a[0] = 3  # right
					dir_a[1] = 1  # down
					dir_a[2] = 2  # left
					clockwise_of_next = self.reconstruction_matrix[neighbor_row, neighbor_col + 1]  # right neighbor
					counter_clockwise_of_next = self.reconstruction_matrix[neighbor_row, neighbor_col - 1]  # left neighbor

				elif r == ROTATION_180:  # LEFT
					neighbor_col = placed_col - 1
					neighbor_row = placed_row
					dir_a[0] = 0
					dir_a[1] = 3
					dir_a[2] = 1
					clockwise_of_next = self.reconstruction_matrix[neighbor_row - 1, neighbor_col]
					counter_clockwise_of_next = self.reconstruction_matrix[neighbor_row + 1, neighbor_col]

				elif r == ROTATION_270:  # DOWN
					neighbor_col = placed_col
					neighbor_row = placed_row + 1
					dir_a[0] = 2  # left
					dir_a[1] = 0  # up
					dir_a[2] = 3  # right
					clockwise_of_next = self.reconstruction_matrix[neighbor_row, neighbor_col - 1]
					counter_clockwise_of_next = self.reconstruction_matrix[neighbor_row, neighbor_col + 1]

				else:
					# default will never happen, but you gotta put *something* for safe coding
					print("BAD ORDINAL VALUE")
					neighbor_col = 0
					neighbor_row = 0
					clockwise_of_next = None
					counter_clockwise_of_next = None

				# # do we need this??
				# if ((useSize and (
				# 		neighbor_col - minX + 1 > puzzleParts.nw | | maxX - neighbor_col - originIndex + 1 > puzzleParts.nw | |
				# 		neighbor_row - minY + 1 > puzzleParts.nh | | maxY - neighbor_row - originIndex + 1 > puzzleParts.nh))):
				# 	continue

				num_best_neighbor_relationships_of_next = 0  # how many best neighbor relationships there are (out of 8) if we place nextA in this location

				w1 = 0.5
				w2 = 0.5

				# TODO: Need to use the actual indices for the construction matrix and best buddies
				# TODO: These need extra checks to make sure they don't go out of bounds on self.construction_matrix

				if self.construction_matrix[neighbor_row - 1, neighbor_col] == YES_PIECE and best_neighbors[self.construction_matrix[neighbor_row - 1, neighbor_col] - 1][1] == neighbor:
					# there's a piece above and is best neighbor up
					num_best_neighbor_relationships_of_next += 1

				if self.construction_matrix[neighbor_row + 1, neighbor_col] == YES_PIECE and best_neighbors[neighbor, 1] == self.construction_matrix[neighbor_row + 1, neighbor_col] - 1:
					num_best_neighbor_relationships_of_next += 1

				if self.construction_matrix[neighbor_row, neighbor_col - 1] == YES_PIECE and best_neighbors[self.construction_matrix[neighbor_row, neighbor_col - 1] - 1][3] == neighbor:
					num_best_neighbor_relationships_of_next += 1

				if self.construction_matrix[neighbor_row, neighbor_col + 1] == YES_PIECE and best_neighbors[neighbor, 3] == self.construction_matrix[neighbor_row, neighbor_col + 1] - 1:
					num_best_neighbor_relationships_of_next += 1

				if self.construction_matrix[neighbor_row + 1, neighbor_col] == YES_PIECE and best_neighbors[self.construction_matrix[neighbor_row + 1, neighbor_col] - 1][0] == neighbor:
					num_best_neighbor_relationships_of_next += 1

				if self.construction_matrix[neighbor_row - 1, neighbor_col] == YES_PIECE and best_neighbors[neighbor, 0] == self.construction_matrix[neighbor_row - 1, neighbor_col] - 1:
					# there is piece above and is best neighbor down
					num_best_neighbor_relationships_of_next += 1

				if self.construction_matrix[neighbor_row, neighbor_col + 1] == YES_PIECE and best_neighbors[self.construction_matrix[neighbor_row, neighbor_col + 1] - 1][2] == neighbor:
					num_best_neighbor_relationships_of_next += 1

				if self.construction_matrix[neighbor_row, neighbor_col - 1] == YES_PIECE and best_neighbors[neighbor, 2] == self.construction_matrix[neighbor_row, neighbor_col - 1] - 1:
					num_best_neighbor_relationships_of_next += 1

				sum_mutual_compatibility = 0  # aggregates a compatibility of potential neighbors
				num_placed_neighbors_of_next = 0  # counts how many neighbors it would have if placed
				if self.construction_matrix[neighbor_row - 1, neighbor_col] == YES_PIECE:  # if there is a piece above nextA
					sum_mutual_compatibility += (w1 * self.compatibility_scores[1, self.construction_matrix[neighbor_row - 1, neighbor_col] - 1][neighbor]) + (w2 * self.compatibility_scores[0, neighbor, self.construction_matrix[neighbor_row - 1, neighbor_col] - 1])
					# w1 * confidence score FROM piece above TO nextA oriented down (1) # w2 * confidence score TO piece above FROM nextA oriented up (0)
					num_placed_neighbors_of_next += 1

				if self.construction_matrix[neighbor_row + 1, neighbor_col] == YES_PIECE:
					sum_mutual_compatibility += (w1 * self.compatibility_scores[0, self.construction_matrix[neighbor_row + 1, neighbor_col] - 1][neighbor]) + (w2 * self.compatibility_scores[1, neighbor, self.construction_matrix[neighbor_row + 1, neighbor_col] - 1])
					num_placed_neighbors_of_next += 1

				if self.construction_matrix[neighbor_row, neighbor_col - 1] == YES_PIECE:
					sum_mutual_compatibility += (w1 * self.compatibility_scores[3, self.construction_matrix[neighbor_row, neighbor_col - 1] - 1][neighbor]) + (w2 * self.compatibility_scores[2, neighbor, self.construction_matrix[neighbor_row, neighbor_col - 1] - 1])
					num_placed_neighbors_of_next += 1

				if self.construction_matrix[neighbor_row, neighbor_col + 1] == YES_PIECE:
					sum_mutual_compatibility += (w1 * self.compatibility_scores[2, self.construction_matrix[neighbor_row, neighbor_col + 1] - 1][neighbor]) + (w2 * self.compatibility_scores[3, neighbor, self.construction_matrix[neighbor_row, neighbor_col + 1] - 1])
					num_placed_neighbors_of_next += 1

				cycle_bonus = 0.0

				if (num_placed_neighbors_of_next == 1 and
						(best_neighbors[neighbor, dir_a[0]] != -1
						 and best_neighbors[best_neighbors[neighbor, dir_a[0]], dir_a[1]] != -1
						 and best_neighbors[best_neighbors[best_neighbors[neighbor, dir_a[0]], dir_a[1]], dir_a[2]] == placed_piece)):
					cycle_bonus += 0.25

				if (clockwise_of_next is not None and num_placed_neighbors_of_next == 1 and  # if next has ONE placed clockwise neighbor
						(best_neighbors[neighbor, dir_a[0]] != -1 and  # and has a best neighbor in that direction
						 best_neighbors[best_neighbors[neighbor, dir_a[0]], dir_a[1]] == clockwise_of_next)):  # ????
					cycle_bonus += 0.25
					print("???")

				if (num_placed_neighbors_of_next == 1 and
						(best_neighbors[neighbor, dir_a[2]] != -1
						 and best_neighbors[best_neighbors[neighbor, dir_a[2]], dir_a[1]] != -1
						 and best_neighbors[best_neighbors[best_neighbors[neighbor, dir_a[2]], dir_a[1]], dir_a[0]] == placed_piece)):
					cycle_bonus += 0.25

				if (counter_clockwise_of_next is not None and num_placed_neighbors_of_next == 1 and
						(best_neighbors[neighbor, dir_a[2]] != -1 and
						 best_neighbors[best_neighbors[neighbor, dir_a[2]], dir_a[1]] == counter_clockwise_of_next)):
					cycle_bonus += 0.25

				best_neighbor_bonus = max((num_best_neighbor_relationships_of_next - 2), 0)  # if 0,1,or 2 neighbors result is 0, else if 3-8 neighbors result is 1-6
				neighbor_count_bonus = max((num_placed_neighbors_of_next - 2), 0)  # if 0,1,or 2 neighbors result is 0, else if 3 or 4 neighbors result is 1 or 2
				# TODO: neighbor_count_bonus is currently unused

				if self.construction_matrix[neighbor_row, neighbor_col] != YES_PIECE:  # if spot is available
					# calculate the score if we were to place it there
					placement_score = sum_mutual_compatibility / num_placed_neighbors_of_next

					if check_cycles:
						placement_score += cycle_bonus
						placement_score += best_neighbor_bonus * 0.2
					elif not check_mutuality:  # *does* get a bonus for having more neighbors
						placement_score += num_placed_neighbors_of_next * 0.5

					# put it in the preference_pool
					heapq.heappush(self.preference_pool, PoolCandidate(placement_score, neighbor, neighbor_row, neighbor_col))

	def block_dissimilarity_scores(self, row: int, col: int, placed_patch: int):
		if self.construction_matrix[row, col + 1] == YES_PIECE:
			self.dissimilarity_scores[placed_patch, ROTATION_0, :] = INFINITY
			self.dissimilarity_scores[:, ROTATION_180, placed_patch] = INFINITY
		if self.construction_matrix[row, col - 1] == YES_PIECE:
			self.dissimilarity_scores[placed_patch, ROTATION_180, :] = INFINITY
			self.dissimilarity_scores[:, ROTATION_0, placed_patch] = INFINITY
		if self.construction_matrix[row + 1, col] == YES_PIECE:
			self.dissimilarity_scores[placed_patch, ROTATION_90, :] = INFINITY
			self.dissimilarity_scores[:, ROTATION_270, placed_patch] = INFINITY
		if self.construction_matrix[row - 1, col] == YES_PIECE:
			self.dissimilarity_scores[placed_patch, ROTATION_270, :] = INFINITY
			self.dissimilarity_scores[:, ROTATION_90, placed_patch] = INFINITY

		for i in self.patches_placed:
			self.dissimilarity_scores[i, :, placed_patch] = INFINITY
			self.dissimilarity_scores[placed_patch, :, i] = INFINITY

	def add_candidates(self, best_neighbors: np.ndarray):
		for row in range(self.construction_matrix.shape[0]):
			for col in range(self.construction_matrix.shape[1]):
				if self.construction_matrix[row, col] == YES_PIECE:
					if self.construction_matrix[row + 1, col] == EXPANSION_SPACE or self.construction_matrix[row - 1, col] == EXPANSION_SPACE \
							or self.construction_matrix[row, col + 1] == EXPANSION_SPACE or self.construction_matrix[row, col - 1] == EXPANSION_SPACE:
						self.add_buddies_to_pool(self.reconstruction_matrix[row, col, 0], row, col, False, False, best_neighbors)  # checkMutuality false for now

	def adjust_matrices(self, row: int, col: int):
		self.construction_matrix[row, col] = YES_PIECE
		# resize reconstruction and construction matrices
		if row == 0 or row == self.construction_matrix.shape[0] - 1:
			new_scores_row = np.zeros((1, self.reconstruction_matrix.shape[1], 2), dtype=int)
			new_construction_row = np.empty((1, self.construction_matrix.shape[1]), dtype=int)
			new_construction_row[:, :] = NO_PIECE
			if row == 0:  # we placed one on the top row
				for piece in self.preference_pool:
					piece.row += 1
				self.reconstruction_matrix = np.concatenate((new_scores_row, self.reconstruction_matrix), axis=0)
				self.construction_matrix = np.concatenate((new_construction_row, self.construction_matrix), axis=0)
				row += 1
			else:  # we placed one on the bottom row
				self.reconstruction_matrix = np.concatenate((self.reconstruction_matrix, new_scores_row), axis=0)
				self.construction_matrix = np.concatenate((self.construction_matrix, new_construction_row), axis=0)
		if col == 0 or col == self.construction_matrix.shape[1] - 1:
			new_scores_col = np.zeros((self.reconstruction_matrix.shape[0], 1, 2), dtype=int)
			new_construction_col = np.empty((self.construction_matrix.shape[0], 1), dtype=int)
			new_construction_col[:, :] = NO_PIECE
			if col == 0:  # we placed one on the left column
				for piece in self.preference_pool:
					piece.col += 1
				self.reconstruction_matrix = np.concatenate((new_scores_col, self.reconstruction_matrix), axis=1)
				self.construction_matrix = np.concatenate((new_construction_col, self.construction_matrix), axis=1)
				col += 1
			else:  # we placed one on the bottom row
				self.reconstruction_matrix = np.concatenate((self.reconstruction_matrix, new_scores_col), axis=1)
				self.construction_matrix = np.concatenate((self.construction_matrix, new_construction_col), axis=1)
		# check and update the neighbors of the newly placed patch
		for row_shift in [-1, 1]:
			if self.construction_matrix[row + row_shift, col] == NO_PIECE:
				self.construction_matrix[row + row_shift, col] = EXPANSION_SPACE
		for col_shift in [-1, 1]:
			if self.construction_matrix[row, col + col_shift] == NO_PIECE:
				self.construction_matrix[row, col + col_shift] = EXPANSION_SPACE

	def greedy_placement_loop(self, first_piece: int, best_neighbors: np.ndarray):
		# need to add first piece to the puzzle
		# need to correctly make the new puzzle (the one we're going to add pieces to one piece at a time) with the right dimensions etc.
		# need to make a potential preference_pool which adds all the best buddies of the last piece placed
		# then edge with best compatibility score is added to the puzzle
		# construction_matrix needs to then be updated and reconstruction_matrix may need to have size updated
		# if the preference_pool is empty, we have to re score best buddy matrix and exclude pieces that have already been placed
		# continue the process until all pieces have been placed
		pieces_remaining = self.n
		with tqdm(total=self.n) as progress:
			self.reconstruction_matrix[1, 1] = [first_piece, 0]  # Add the first piece, 0 refers to the rotation
			self.block_dissimilarity_scores(1, 1, first_piece)
			self.patches_placed.add(first_piece)
			pieces_remaining -= 1
			self.add_buddies_to_pool(first_piece, 1, 1, True, True, best_neighbors)
			progress.update()

			while pieces_remaining > 0:
				if len(self.preference_pool) > 0:
					# get next piece from the preference pool
					next_piece = heapq.heappop(self.preference_pool)

					if next_piece.index in self.patches_placed or self.construction_matrix[next_piece.row, next_piece.col] == YES_PIECE:
						# this piece is already placed; skip it
						continue

					can_add_piece = False
					if next_piece.row > 0 and self.construction_matrix[next_piece.row - 1, next_piece.col] == YES_PIECE:
						neighbor_up = self.reconstruction_matrix[next_piece.row - 1, next_piece.col, 0]
						if best_neighbors[neighbor_up, ROTATION_270] == next_piece.index and best_neighbors[next_piece.index, ROTATION_90] == neighbor_up:
							can_add_piece = True
					if next_piece.row < self.construction_matrix.shape[0] - 1 and self.construction_matrix[next_piece.row + 1, next_piece.col] == YES_PIECE:
						neighbor_down = self.reconstruction_matrix[next_piece.row + 1, next_piece.col, 0]
						if best_neighbors[neighbor_down, ROTATION_90] == next_piece.index and best_neighbors[next_piece.index, ROTATION_270] == neighbor_down:
							can_add_piece = True
					if next_piece.col > 0 and self.construction_matrix[next_piece.row, next_piece.col - 1] == YES_PIECE:
						neighbor_left = self.reconstruction_matrix[next_piece.row, next_piece.col - 1, 0]
						if best_neighbors[neighbor_left, ROTATION_0] == next_piece.index and best_neighbors[next_piece.index, ROTATION_180] == neighbor_left:
							can_add_piece = True
					if next_piece.col < self.construction_matrix.shape[1] - 1 and self.construction_matrix[next_piece.row, next_piece.col + 1] == YES_PIECE:
						neighbor_right = self.reconstruction_matrix[next_piece.row, next_piece.col + 1, 0]
						if best_neighbors[neighbor_right, ROTATION_180] == next_piece.index and best_neighbors[next_piece.index, ROTATION_0] == neighbor_right:
							can_add_piece = True

					if not can_add_piece:
						# can't do this piece in this spot
						continue  # we already removed the piece using `heapq.heappop`

					# place the piece
					self.reconstruction_matrix[next_piece.row, next_piece.col] = [next_piece.index, 0]
					self.adjust_matrices(next_piece.row, next_piece.col)
					self.block_dissimilarity_scores(next_piece.row, next_piece.col, next_piece.index)
					self.patches_placed.add(next_piece.index)
					pieces_remaining -= 1
					# we already removed the piece using `heapq.heappop`
					self.add_buddies_to_pool(next_piece.index, next_piece.row, next_piece.col, True, True, best_neighbors)
					if pieces_remaining < self.n / 2:
						self.add_buddies_to_pool(next_piece.index, next_piece.row, next_piece.col, False, False, best_neighbors)
					progress.update()

				else:  # preference_pool is empty
					best_neighbors_dissimilarity = get_best_neighbors(self.dissimilarity_scores, self.rotations_shuffled)
					self.get_compatibility_scores(best_neighbors_dissimilarity)
					best_neighbors = get_best_neighbors(self.compatibility_scores, self.rotations_shuffled)
					# print("finding initial best buddies...")
					# buddy_matrix = get_best_buddies(compatibility_scores, rotations_shuffled)
					self.add_candidates(best_neighbors)

		return self.reconstruction_matrix[1:-1, 1:-1, :]  # trim off padding edges

	def count_best_buddies(self):
		total_slots = self.buddy_matrix.shape[0] * self.buddy_matrix.shape[1]
		buddy_count = 0
		for i in range(self.buddy_matrix.shape[0]):
			for j in range(self.buddy_matrix.shape[1]):
				if self.buddy_matrix[i, j] is not None:
					buddy_count += 1
		print(f"best buddies count: {buddy_count} / {total_slots}")

	def solve(self) -> np.ndarray:
		print("computing 3rd pixel predictions...")
		self.get_3rd_pixel_predictions()
		print("computing average border colors...")
		self.get_average_edge_colors()
		print("computing dissimilarity scores...")
		self.get_dissimilarity_scores()
		print("computing best neighbors by dissimilarity...")
		best_neighbors_dissimilarity = get_best_neighbors(self.dissimilarity_scores, self.rotations_shuffled)
		print("computing initial compatibility scores...")
		self.get_compatibility_scores(best_neighbors_dissimilarity)
		print("computing best neighbors by compatibility...")
		best_neighbors_compatibility = get_best_neighbors(self.compatibility_scores, self.rotations_shuffled)
		print("finding initial best buddies...")
		self.get_best_buddies()
		self.count_best_buddies()
		print("selecting first piece...")
		first_piece = self.pick_first_piece(best_neighbors_compatibility)
		print(f"first piece selected: {first_piece}")
		print("running main placement loop...")
		self.greedy_placement_loop(first_piece, best_neighbors_compatibility)
		return self.reconstruction_matrix
