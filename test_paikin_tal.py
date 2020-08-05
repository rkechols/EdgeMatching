import numpy as np
from unittest import TestCase
from paikin_tal import get_best_buddies


class PaikinTalTest(TestCase):
	def test_get_best_buddies(self):
		compatibility_scores = np.array(
			[
				[
					[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
					[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
					[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
					[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
				],
				[
					[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
					[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
					[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
					[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
				],
				[
					[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
					[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
					[[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
					[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
				]
			]
			, dtype=float
		)
		buddies = get_best_buddies(compatibility_scores)
		# right shape
		self.assertTupleEqual((3, 4), buddies.shape, "resulting matrix was the wrong shape")
		# make sure there are no false values
		for patch_index1 in range(buddies.shape[0]):
			for r1 in range(buddies.shape[1]):
				t = buddies[patch_index1, r1]
				if t is not None:
					if patch_index1 == 0 and r1 == 0:
						self.assertTupleEqual(t, (1, 3), f"best buddy listed for {(patch_index1, r1)} was incorrect")
					elif patch_index1 == 1 and r1 == 3:
						self.assertTupleEqual(t, (0, 0), f"best buddy listed for {(patch_index1, r1)} was incorrect")
					else:
						self.fail(f"found a non-None tuple (buddy pairing) that should be none: {(patch_index1, r1)} with {t}")
				else:
					if (patch_index1 == 0 and r1 == 0) or (patch_index1 == 1 and r1 == 3):
						self.fail(f"{(patch_index1, r1)} was listed as having no best buddy, but should have a buddy")
