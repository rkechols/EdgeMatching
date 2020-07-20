from unittest import TestCase
from functions import load_image_from_disk
from kldiv import data_to_probability_distribution, kl_divergence_symmetric


class KLDivergenceTest(TestCase):
	def test_kl_divergence_symmetric(self):
		image_giraffe = load_image_from_disk("TestImages/Giraffe.jpg")
		image_theo = load_image_from_disk("TestImages/theo.jpg")
		image_strange = load_image_from_disk("TestImages/Strange.png")
		distributions_giraffe = [data_to_probability_distribution(image_giraffe[:, :, c], -0.5, 255.5, 64, True) for c in range(image_giraffe.shape[2])]
		distributions_theo = [data_to_probability_distribution(image_theo[:, :, c], -0.5, 255.5, 64, True) for c in range(image_theo.shape[2])]
		distributions_strange = [data_to_probability_distribution(image_strange[:, :, c], -0.5, 255.5, 64, True) for c in range(image_strange.shape[2])]
		print(f"giraffe v theo: {[kl_divergence_symmetric(distributions_giraffe[i], distributions_theo[i]) for i in range(3)]}")
		print(f"theo v strange: {[kl_divergence_symmetric(distributions_theo[i], distributions_strange[i]) for i in range(3)]}")
		print(f"strange v giraffe: {[kl_divergence_symmetric(distributions_strange[i], distributions_giraffe[i]) for i in range(3)]}")
