import numpy as np
from constants import *
from functions import boring_score, combine_patches, dissimilarity_score, kl_score, show_image


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
		score = sum(s * w for s, w in zip([b_score, d_score, k_score], [0.61, 1.05, 1.1]))
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
