import numpy as np
from functions import block_rot90, coord_rot90


def verify_accuracy(original, reconstructed, shuffle_dictionary): # TODO, don't actually need original patches?
    num_rows = reconstructed.shape[0]
    num_columns = reconstructed.shape[1]
    original_mapped = np.zeros((num_rows, num_columns, 2))
    for i in range(num_rows):
        for j in range(num_columns):
            index = (i * num_columns) + j
            original_mapped[i][j] = shuffle_dictionary[index]

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
            relative_accuracy(rotated_orig, reconstructed)
        # print("END OF ROTATION " + str(rotation))
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
            # else:
                # print("incorrect: " + str(reconstructed[i][j]) + ", should be: " + str(correct[i][j]))
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
# TODO edge cases such as weirdly shaped reconstructed stuff


def relative_accuracy(correct, reconstructed):
    # percentage of correct edge pairings
    score = 0
    # find each square with index of correct, look at it's 4 (9) neighbors, rotate those 9 squares
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]): # for each square in correct
            index_correct = correct[i][j][0]
            row_recon, col_recon = find_index_match(index_correct, reconstructed) # find index in reconstruction matrix that matches this index

            rotated = reconstructed #name?
            rotation_difference = reconstructed[row_recon][col_recon][1] - correct[i][j][1] % 4
            if rotation_difference != 0:
                rotated = block_rot90(reconstructed, rotation_difference)
                row_recon, col_recon = coord_rot90(row_recon, col_recon, reconstructed.shape[0],
                                                   reconstructed.shape[1], rotation_difference)

            correct_edges = [(-1, 0)] * 4
            if i > 0:
                correct_edges[0] = correct[i-1][j]
            if j < correct.shape[0] - 1:
                correct_edges[1] = correct[i][j+1]
            if i < correct.shape[1] - 1:
                correct_edges[2] = correct[i+1][j]
            if j > 0:
                correct_edges[3] = correct[i][j-1]

            reconstructed_edges = [(-1, 0)] * 4
            if row_recon > 0:
                reconstructed_edges[0] = rotated[row_recon - 1][col_recon]
            if col_recon < rotated.shape[0] - 1:
                reconstructed_edges[1] = rotated[row_recon][col_recon + 1]
            if row_recon < rotated.shape[1] - 1:
                reconstructed_edges[2] = rotated[row_recon + 1][col_recon]
            if col_recon > 0:
                reconstructed_edges[3] = rotated[row_recon][col_recon - 1]

            score += np.sum(correct_edges == reconstructed_edges)

    print("score" + str(score/ (16 * 4))) #todo hard code fix
    return score / 16


def find_index_match(index_to_find, matrix):
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row][col][0] == index_to_find:
                return row, col
    return "not found"