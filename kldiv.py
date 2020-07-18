import numpy as np
from matplotlib import pyplot as plt
from functions import load_image_from_disk


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
	"""
	calculates the Kullback–Leibler divergence (relative entropy) of two probability distributions. note that this is asymmetric
	:param p: the first probability distribution
	:param q: the second probability distribution
	:return: the KL divergence value
	"""
	return sum(p[i] * np.log(p[i] / q[i]) for i in range(len(p)) if p[i] != 0 and q[i] != 0)


def kl_divergence_symmetric(p: np.ndarray, q: np.ndarray) -> float:
	"""
	calculates the symmetric Kullback–Leibler divergence (relative entropy) of two probability distributions by calculating the two asymmetric values and averaging them
	:param p: the first probability distribution
	:param q: the second probability distribution
	:return: the symmetric KL divergence value
	"""
	left = _kl_divergence(p, q)
	right = _kl_divergence(q, p)
	return (left + right) / 2


def data_to_probability_distribution(data: np.ndarray, low: float, high: float, bucket_count: int, show_plot: bool = False) -> np.ndarray:
	"""
	takes an array of values and creates a probability distribution with the given parameters
	:param data: a numpy array of values to be analyzed
	:param low: the lowest value of the probability distribution. values below this will be ignored
	:param high: the highest value of the probability distribution. values above this will be ignored
	:param bucket_count: the number of buckets of equal size that the data will be divided into
	:param show_plot: if `True`, the resulting probability distribution will be plotted and shown
	:return: the resulting probability distribution as a 1-dimensional numpy array with `bucket_count` values
	"""
	flat_data = list(data.flatten())
	width = high - low
	bucket_width = width / bucket_count
	# list of values that divide the buckets
	dividers = [low + (i * bucket_width) for i in range(bucket_count + 1)]
	# the actual distribution array, normalized so the sum is 1.0
	distribution = np.histogram(flat_data, dividers)[0]
	distribution = distribution / len(flat_data)
	if show_plot:
		bucket_centers = [(dividers[i] + dividers[i + 1]) / 2 for i in range(bucket_count)]
		plt.plot(bucket_centers, distribution)
		plt.show()
	return distribution


if __name__ == "__main__":
	image_giraffe = load_image_from_disk("TestImages/Giraffe.jpg")
	image_theo = load_image_from_disk("TestImages/theo.jpg")
	image_strange = load_image_from_disk("TestImages/Strange.png")
	distributions_giraffe = [data_to_probability_distribution(image_giraffe[:, :, c], -0.5, 255.5, 64, True) for c in range(image_giraffe.shape[2])]
	distributions_theo = [data_to_probability_distribution(image_theo[:, :, c], -0.5, 255.5, 64, True) for c in range(image_theo.shape[2])]
	distributions_strange = [data_to_probability_distribution(image_strange[:, :, c], -0.5, 255.5, 64, True) for c in range(image_strange.shape[2])]
	print(f"giraffe v theo: {[kl_divergence_symmetric(distributions_giraffe[i], distributions_theo[i]) for i in range(3)]}")
	print(f"theo v strange: {[kl_divergence_symmetric(distributions_theo[i], distributions_strange[i]) for i in range(3)]}")
	print(f"strange v giraffe: {[kl_divergence_symmetric(distributions_strange[i], distributions_giraffe[i]) for i in range(3)]}")
