import random
import numpy as np
from unittest import TestCase
from constants import INFINITY
from paikin_tal import get_best_buddies, pick_first_piece


class PaikinTalTest(TestCase):
	# def test_get_best_buddies(self):  # TODO: update this test for the current format of compatibility_scores
	# 	for _ in range(1000):  # multiple trials since randomness is involved
	# 		n = 3
	# 		compatibility_scores = np.empty((n, 4), dtype=tuple)
	# 		for i in range(n):
	# 			for r in range(4):
	# 				other_patch_index = i
	# 				while other_patch_index == i:
	# 					other_patch_index = random.randrange(n)
	# 				other_r = random.randrange(4)
	# 				score = random.uniform(0.0, 0.95)
	# 				compatibility_scores[i, r] = (other_patch_index, other_r, score)
	# 		compatibility_scores[0, 0] = (1, 3, 1.0)
	# 		compatibility_scores[1, 1] = (0, 2, 1.0)
	# 		compatibility_scores[2, 2] = (1, 0, 1.0)
	# 		buddies = get_best_buddies(compatibility_scores)
	# 		# right shape
	# 		self.assertTupleEqual((3, 4), buddies.shape, "resulting matrix was the wrong shape")
	# 		# make sure there are no false values
	# 		not_none_count = 0
	# 		for patch_index1 in range(buddies.shape[0]):
	# 			for r1 in range(buddies.shape[1]):
	# 				t = buddies[patch_index1, r1]
	# 				if t is not None:
	# 					not_none_count += 1
	# 					if patch_index1 == 0 and r1 == 0:
	# 						self.assertTupleEqual(t, (1, 3), f"best buddy listed for {(patch_index1, r1)} was incorrect")
	# 					elif patch_index1 == 1 and r1 == 1:
	# 						self.assertTupleEqual(t, (0, 2), f"best buddy listed for {(patch_index1, r1)} was incorrect")
	# 					else:
	# 						# verify it's symmetric
	# 						self.assertEqual((patch_index1, (r1 + 2) % 4), buddies[t[0], (t[1] + 2) % 4], f"found an asymmetric buddy pairing:\n{buddies}")
	# 				else:
	# 					if (patch_index1 == 0 and r1 == 0) or (patch_index1 == 1 and r1 == 1):
	# 						self.fail(f"{(patch_index1, r1)} was listed as having no best buddy, but should have a buddy:\n{buddies}")
	# 		self.assertTrue(not_none_count % 2 == 0, "number of patches with best buddies should be even, since the relation is symmetric but irreflexive")

	def test_pick_first_piece_no_competition(self):
		# for a visualization of this test case, see the following: https://1drv.ms/u/s!AsgHxnBnyNbihXfX9GsLzyytvuxm
		buddy_matrix = np.array(
			[
				[(10, 1), (2, 2), None, None],      # 0
				[(10, 3), None, None, (5, 1)],      # 1
				[(0, 3), (6, 1), (7, 3), (8, 3)],   # 2
				[None, None, (9, 3), None],         # 3
				[None, (5, 2), None, None],         # 4
				[(4, 3), (12, 2), (6, 3), (1, 1)],  # 5
				[(10, 0), (5, 0), (9, 1), (2, 3)],  # 6
				[None, (2, 0), (9, 0), None],       # 7
				[None, (2, 1), None, None],         # 8
				[(12, 1), (3, 0), (7, 0), (6, 0)],  # 9
				[(11, 1), (1, 2), (6, 2), (0, 2)],  # 10
				[None, None, None, (10, 2)],        # 11
				[(5, 3), (14, 0), None, (9, 2)],    # 12
				[None, None, None, (14, 1)],        # 13
				[(12, 3), None, None, (13, 1)]      # 14
			]
		)
		n = buddy_matrix.shape[0]
		compatibility_scores = np.zeros((n, 4, n, 4), dtype=float)
		answer = pick_first_piece(buddy_matrix, compatibility_scores)
		self.assertEqual(answer, 6)

	def test_pick_first_piece_by_mutual_compatibility(self):
		pass
