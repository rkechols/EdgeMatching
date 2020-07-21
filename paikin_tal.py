# based on the paper by Genady Paikin and Ayellet Tal, found at http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7299116


import numpy as np


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


def compatibility_score(dissimilarity_matrix: np.ndarray, patch1_index: int, patch1_r: int, patch2_index: int, patch2_r: int) -> float:
	"""
	takes a matrix of all dissimilarity scores plus a particular combination of patches and gives the asymmetric compatibility score of that combination
	:param dissimilarity_matrix: the matrix of all dissimilarity scores as a numpy array of shape (n, 4, n, 4), where n is the total number of patches
	:param patch1_index: the index referring to the first/left patch
	:param patch1_r: an int in range [0, 3] indicating how far the first/left patch has been rotated
	:param patch2_index: the index referring to the second/right patch
	:param patch2_r: an int in range [0, 3] indicating how far the second/right patch has been rotated
	:return: the asymmetric compatibility score of the two pieces
	"""
	d_score = dissimilarity_matrix[patch1_index, patch1_r, patch2_index, patch2_r]
	relevant_slice = dissimilarity_matrix[patch1_index, patch1_r, :, :]
	second_best_d_score = np.amin(relevant_slice[relevant_slice != np.amin(relevant_slice)])
	return 1.0 - (d_score / second_best_d_score)


def compatibility_score_symmetric(dissimilarity_matrix: np.ndarray, patch1_index: int, patch1_r: int, patch2_index: int, patch2_r: int) -> float:
	"""
	takes a matrix of all dissimilarity scores plus a particular combination of patches and gives the symmetric compatibility (mutual compatibility) score of that combination
	:param dissimilarity_matrix: the matrix of all dissimilarity scores as a numpy array of shape (n, 4, n, 4), where n is the total number of patches
	:param patch1_index: the index referring to the first/left patch
	:param patch1_r: an int in range [0, 3] indicating how far the first/left patch has been rotated
	:param patch2_index: the index referring to the second/right patch
	:param patch2_r: an int in range [0, 3] indicating how far the second/right patch has been rotated
	:return: the symmetric compatibility (mutual compatibility) score of the two pieces
	"""
	c1 = compatibility_score(dissimilarity_matrix, patch1_index, patch1_r, patch2_index, patch2_r)
	c2 = compatibility_score(dissimilarity_matrix, patch2_index, patch2_r, patch1_index, patch1_r)
	return (c1 + c2) / 2.0


def jigsaw_pt(patches: list):
	pass
