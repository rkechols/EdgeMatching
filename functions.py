import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Union


def load_image_from_disk(image_file_name: str) -> np.ndarray:
	"""
	takes a path to an image file and loads it from disk
	:param image_file_name: a string indicating which image file to load
	:return: a numpy array of the image with shape (r, c, 3)
	"""
	badly_oriented_image = cv2.imread(image_file_name)
	return np.fliplr(badly_oriented_image.reshape(-1, 3)).reshape(badly_oriented_image.shape)


def show_image(image_rgb: np.ndarray, label: str = None):
	"""
	shows an rgb image
	:param image_rgb: the image to show as a numpy array with shape (r, c, 3)
	:param label: a label to apply to the image
	:return: None
	"""
	plt.style.use("dark_background")
	plt.imshow(image_rgb, vmin=0, vmax=255)
	plt.axis("off")
	if label is not None:
		plt.title(label)
	plt.show()


def assemble_patches(patches: list, num_cols: int) -> Union[None, np.ndarray]:
	"""
	takes a list of square patches and returns a tiled image of the patches with the specified number of columns
	:param patches: a list of numpy arrays each of shape (x, x, 3), being the patches to assemble
	:param num_cols: number of patches for each row to contain. a value of -1 means that all patches are placed on one row
	:return: a numpy array of shape (r, c, 3) that is the tiled image of all the patches
	"""
	n = len(patches)
	if n == 0:
		return None
	patch_size = patches[0].shape[0]  # assumes all are the same size and square
	if num_cols == -1:
		num_rows = 1
		num_cols = n
	else:
		num_rows = n // num_cols
	if n % num_cols != 0:
		num_rows += 1
	# tile the patches
	assembled = np.zeros((num_rows * patch_size, num_cols * patch_size, 3), dtype=patches[0].dtype)
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
