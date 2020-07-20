from unittest import TestCase
from functions import load_image_from_disk
from image_assembly import scramble_image, show_image, assemble_patches


class ImageAssemblyTest(TestCase):
	def test_scramble_image(self):
		file_name = "TestImages/theo.jpg"
		image_shape = (800, 676, 3)
		image = load_image_from_disk(file_name)
		patch_size = 100
		for trial in range(3):
			patches = scramble_image(image, patch_size)
			patch_row_count = image_shape[0] // patch_size
			patch_col_count = image_shape[1] // patch_size
			n = patch_row_count * patch_col_count
			self.assertEqual(len(patches), n, "wrong number of patches")
			show_image(assemble_patches(patches, patch_col_count))
