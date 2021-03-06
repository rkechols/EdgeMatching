import copy
import datetime
import numpy as np
import random
from functions import assemble_patches, load_image_from_disk, show_image
from kruskal import jigsaw_kruskals, assemble_image_kruskal
from accuracy import verify_accuracy
from prim import jigsaw_prims


# globals
# hypothetical_min = 0
# hypothetical_max = 255

def patch_image(image: np.ndarray, patch_size: int) -> list: #todo return
	vertical_patches = image.shape[0] // patch_size
	horizontal_patches = image.shape[1] // patch_size

	patched_array = list()

	for i in range(vertical_patches):
		# starting pixel
		vert_pixel_location = patch_size * i
		for j in range(horizontal_patches):
			# starting pixel
			hor_pix_location = patch_size * j
			patch = image[vert_pixel_location:vert_pixel_location + patch_size, hor_pix_location:hor_pix_location + patch_size, :]
			patched_array.append(patch)
	return patched_array, (vertical_patches, horizontal_patches)


def scramble_image(patches: list, seed: int = None) -> (list, dict):
	"""
	takes an rgb image and scrambles it into square patches, each at a random rotation
	:param patches: TODO
	:param seed: a seed to use for the image scrambling. If none, then the scramble will be truly random
	:return: a list containing the scrambled patches, each of shape (patch_size, patch_size, 3) TODO
	"""
	n = len(patches)
	if seed is not None:
		random.seed(seed)
	# make a list of the original indices, and a list of the shuffled indices so we know what piece went where
	all_indices = list(range(n))
	all_indices_shuffled = copy.copy(all_indices)
	random.shuffle(all_indices_shuffled)
	rotations = [random.randrange(4) for _ in range(n)]
	shuffle_dict = dict()
	patches_scrambled = list()
	for original, new, r in zip(all_indices, all_indices_shuffled, rotations):
		shuffle_dict[new] = (original, (4 - r) % 4)
		patches_scrambled.append(np.rot90(patches[new], r))
	return patches_scrambled, shuffle_dict


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
	original_image = load_image_from_disk("TestImages/Giraffe.jpg ")
	show_image(original_image, "original")
	# ps = original_image.shape[1] // 2
	ps = 28
	original_patched, dimensions = patch_image(original_image, ps)
	patch_list, shuffle_dictionary = scramble_image(original_patched, 4)
	show_image(assemble_patches(patch_list, original_image.shape[1] // ps), "scrambled")
	# hypothetical_min = 85 + (15.038 * math.log(ps))
	# hypothetical_max = 255 - (14.235 * math.log(ps))
	print(f"algorithm start time: {datetime.datetime.now()}")
	reconstruction_matrix = jigsaw_kruskals(patch_list)
	reconstructed_image = assemble_image_kruskal(patch_list, reconstruction_matrix)
	# reconstructed_image = jigsaw_kruskals(patch_list)
	# reconstructed_image = jigsaw_prims(patch_list)
	print(f"algorithm end time: {datetime.datetime.now()}")
	if reconstructed_image is not None:
		show_image(reconstructed_image, "final answer")
		accuracy, location_accuracy, relative_accuracy = verify_accuracy(reconstruction_matrix, shuffle_dictionary, dimensions)
		show_image(reconstructed_image, "final answer")
		print("Absolute accuracy: " + str(accuracy * 100) + "%")
		print("with " + str(location_accuracy * 100) + "% in the correct position")
		print("Relative accuracy: " + str(relative_accuracy * 100) + "%")
	else:
		print("reconstructed_image is None")
