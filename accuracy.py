import numpy as np
from functions import block_rot90


def verify_accuracy(original, reconstructed, shuffle_dictionary): # TODO, don't actually need original patches?
    num_rows = reconstructed.shape[0]
    num_columns = reconstructed.shape[1]
    original_mapped = np.zeros((num_rows, num_columns, 2))
    for i in range(num_rows):
        for j in range(num_columns):
            index = (i * num_columns) + j
            location_rotation = shuffle_dictionary[index]
            location = location_rotation[0]
            rotation = location_rotation[1]
            original_mapped[location // num_columns][location % num_columns] = [index, rotation]

    max_accuracy = 0
    location_accuracy = 0
    best_rotation = 0
    for rotation in range(4):
        rotated_orig = block_rot90(original_mapped, rotation)
        accuracy = absolute_accuracy(rotated_orig, reconstructed)
        # print("possible accuracy: " + str(accuracy))
        # accuracy_po = absolute_accuracy_placement_only(rotated_orig, reconstructed)
        # print("placement only accuracy: " + str(accuracy_po))
        # accuracy_bo = absolute_accuracy_placement_bonus(rotated_orig, reconstructed)
        # print("placement bonus accuracy: " + str(accuracy_bo))
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_rotation = rotation
            location_accuracy = absolute_accuracy_placement_only(rotated_orig, reconstructed)
        print("END OF ROTATION " + str(rotation))
    print("best rotation: " + str(best_rotation))
    return max_accuracy, location_accuracy


def absolute_accuracy(correct, reconstructed):
    # what percent of the squares have the right piece with right rotation

    score = 0
    num_squares = correct.shape[0] * correct.shape[1]
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]):
            if np.all(correct[i][j] == reconstructed[i][j]):
                score += 1
            else:
                print("incorrect: " + str(reconstructed[i][j]) + ", should be: " + str(correct[i][j]))
    return score / num_squares


def absolute_accuracy_placement_only(correct, reconstructed):
    score = 0
    num_squares = correct.shape[0] * correct.shape[1]
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]):
            if correct[i][j][0] == reconstructed[i][j][0]:
                score += 1
    return score / num_squares


def absolute_accuracy_placement_bonus(correct, reconstructed):
    score = 0
    num_squares = correct.shape[0] * correct.shape[1]
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]):
            if correct[i][j][0] == reconstructed[i][j][0]:
                score += 0.5
                if correct[i][j][1] == reconstructed[i][j][1]: #inside if?
                    score += 0.5
    return score / num_squares


def relative_accuracy():
    # percentage of correct edge pairings
    return 0
