import numpy as np
from functions import block_rot90, coord_rot90


def verify_accuracy(reconstructed, shuffle_dictionary, dimensions):
    num_rows = dimensions[0]
    num_columns = dimensions[1]
    correct_reconstructed = np.zeros((num_rows, num_columns, 2), dtype=int)
    for i in range(num_rows):
        for j in range(num_columns):
            index = (i * num_columns) + j
            correct_reconstructed[i][j] = shuffle_dictionary[index]

    max_accuracy_abs = 0
    location_accuracy = 0
    best_rotation = 0

    for rotation in range(4):
        rotated_orig = block_rot90(correct_reconstructed.copy(), rotation)
        abs_accuracy = absolute_accuracy(rotated_orig, reconstructed)
        # accuracy_bon = absolute_accuracy_placement_bonus(rotated_orig, reconstructed)
        # print("placement bonus accuracy: " + str(accuracy_bon))
        if abs_accuracy > max_accuracy_abs:
            max_accuracy_abs = abs_accuracy
            best_rotation = rotation
            location_accuracy = absolute_accuracy_placement_only(rotated_orig, reconstructed)
    print("best rotation absolute: " + str(best_rotation))

    rel_accuracy = relative_accuracy(correct_reconstructed, reconstructed)

    return max_accuracy_abs, location_accuracy, rel_accuracy


def absolute_accuracy(correct, reconstructed):
    # what percent of the squares have the right piece with right rotation
    num_rows = min(correct.shape[0], reconstructed.shape[0])
    num_cols = min(correct.shape[1], reconstructed.shape[1])
    score = 0
    num_squares = correct.shape[0] * correct.shape[1]
    for i in range(num_rows):
        for j in range(num_cols):
            if np.all(correct[i][j] == reconstructed[i][j]):
                score += 1
            # else:
            #     print("incorrect: " + str(reconstructed[i][j]) + ", should be: " + str(correct[i][j]))
    return score / num_squares


def absolute_accuracy_placement_only(correct, reconstructed):
    score = 0
    num_rows = min(correct.shape[0], reconstructed.shape[0])
    num_cols = min(correct.shape[1], reconstructed.shape[1])
    num_squares = correct.shape[0] * correct.shape[1]
    for i in range(num_rows):
        for j in range(num_cols):
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


def relative_accuracy(correct, reconstructed):
    '''
    Calculates accuracy based on pieces relative to each other.
    If a piece is not rotated the same way, it rotates the matrix before comparing its neighbors
    :param correct: Correct reconstruction matrix
    :param reconstructed: Reconstruction matrix to evaluate its accuracy
    :return: percentage of total edges (number of squares * 4) that are next to the correct neighbor
                and in the correct rotation relative to each other
    '''
    score = 0
    incorrect = 0
    num_rows = correct.shape[0]
    num_cols = correct.shape[1]
    num_edges_total = correct.shape[0] * correct.shape[1] * 4

    for i in range(num_rows):
        for j in range(num_cols): # for each square in correct
            index_correct = correct[i][j][0]
            row_recon, col_recon = find_index_match(index_correct, reconstructed) # find coordinates in reconstruction matrix that matches this value

            rotated_reconstructed = reconstructed.copy()
            rotation_difference = (correct[i][j][1] - rotated_reconstructed[row_recon][col_recon][1]) % 4
            if rotation_difference != 0:
                rotated_reconstructed = block_rot90(rotated_reconstructed, rotation_difference)
                row_recon, col_recon = coord_rot90(row_recon, col_recon, reconstructed.shape[0],
                                                   reconstructed.shape[1], rotation_difference)
            assert(np.all(rotated_reconstructed[row_recon][col_recon] == correct[i][j]))

            correct_edges = [[-1, -1]] * 4
            if i > 0:
                correct_edges[0] = correct[i-1][j]
            if j < correct.shape[1] - 1:
                correct_edges[1] = correct[i][j+1]
            if i < correct.shape[0] - 1:
                correct_edges[2] = correct[i+1][j]
            if j > 0:
                correct_edges[3] = correct[i][j-1]

            reconstructed_edges = [[-1, -1]] * 4
            if row_recon > 0:
                reconstructed_edges[0] = rotated_reconstructed[row_recon - 1][col_recon]
            if col_recon < rotated_reconstructed.shape[1] - 1:
                reconstructed_edges[1] = rotated_reconstructed[row_recon][col_recon + 1]
            if row_recon < rotated_reconstructed.shape[0] - 1:
                reconstructed_edges[2] = rotated_reconstructed[row_recon + 1][col_recon]
            if col_recon > 0:
                reconstructed_edges[3] = rotated_reconstructed[row_recon][col_recon - 1]

            for edge in range(len(correct_edges)):
                if np.all(correct_edges[edge] == reconstructed_edges[edge]) or \
                        (correct_edges[edge][0] == -1 and reconstructed_edges[edge][0] == -1): #rotation doesn't matter if it's a blank piece
                    score += 1
                # else:
                #     incorrect += 1
    # print("incorrect edges: " + str(incorrect))
    # print("rel: " + str(score/num_edges_total))
    return score / num_edges_total  #todo return most frequent rotation


def find_index_match(index_to_find, matrix):
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row][col][0] == index_to_find:
                return row, col
    return "not found"