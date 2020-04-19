def print_mat(mat):
    print()
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            print(mat[i][j], end=' ')
        print()
    print()


def merge_peeled_and_outer_layer(rotated_peeled_matrix, rotated_outer_layer):
    mat_size = len(rotated_peeled_matrix) + 2
    merged = np.zeros((mat_size, mat_size))
    merged[0] = rotated_outer_layer[:mat_size]
    #     print(rotated_outer_layer, mat_size)
    #     print_mat(merged)
    #     TODO
    #     print(rotated_outer_layer[ 3 * mat_size - 3 : 2 * mat_size - 3: -1], 3 * mat_size - 2, 2 * mat_size - 2)
    merged[-1] = rotated_outer_layer[3 * mat_size - 3: 2 * mat_size - 3: -1]
    #     print_mat(merged)
    for i in range(1, mat_size - 1):
        merged[i][-1] = rotated_outer_layer[mat_size - 1 + i]
        merged[i][0] = rotated_outer_layer[-i]
    #     print_mat(merged)
    for i in range(len(rotated_peeled_matrix)):
        for j in range(len(rotated_peeled_matrix[i])):
            merged[i + 1][j + 1] = rotated_peeled_matrix[i][j]
    #     print_mat(merged)

    return merged


def rotate_one_layer_one_click(matrix_outer_layer, clockwise=True):
    start, end = matrix_outer_layer[:1], matrix_outer_layer[-1:]
    mid = matrix_outer_layer[1:-1]
    if clockwise:
        return end + start + mid
    return mid + end + start


def get_outer_layer_of_matrix(matrix):
    if len(matrix) == 0:
        return []
    outer_layer = []
    outer_layer.extend(matrix[0][:])
    for row in range(1, len(matrix) - 1):
        outer_layer.append(matrix[row][- 1])
    outer_layer.extend(matrix[-1][::-1])
    for row in range(len(matrix) - 2, 0, -1):
        outer_layer.append(matrix[row][0])
    return outer_layer


# angle in radians
def rotate_one_layer(matrix_outer_layer, angle, clockwise=True):
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle = 2 * np.pi - angle
        clockwise = not clockwise

    rotated_layer = matrix_outer_layer
    num_of_clicks = int((angle / (2 * np.pi)) * len(matrix_outer_layer))
    for i in range(num_of_clicks):
        rotated_layer = rotate_one_layer_one_click(rotated_layer, clockwise=clockwise)
    return rotated_layer


def peel_matrix(matrix):
    peeled = []
    for i in range(1, len(matrix) - 1):
        new_row = []
        for j in range(1, len(matrix[i]) - 1):
            new_row.append(matrix[i][j])
        peeled.append(new_row)

    return peeled


# squared matrix !
def rotate(matrix, angle):
    #     print("before:")
    #     print_mat(matrix)
    #     print(len(matrix))
    if len(matrix) <= 1:
        return matrix
    peeled_matrix = peel_matrix(matrix)
    #     print("peeled:")
    #     print_mat(peeled_matrix)atom://teletype/portal/2b9176f3-a05a-4be5-bbe8-912fb84fff41
    matrix_outer_layer = get_outer_layer_of_matrix(matrix)
    #     print("outer:")
    #     print(matrix_outer_layer)
    rotated_outer_layer = rotate_one_layer(matrix_outer_layer, angle)
    #     print("rot outer:")
    #     print(rotated_outer_layer)
    rotated_peeled_matrix = rotate(peeled_matrix, angle)
    #     print("rotated peeled matrix:")
    #     print(rotated_peeled_matrix)
    new_mat = merge_peeled_and_outer_layer(rotated_peeled_matrix, rotated_outer_layer)

    #     print("after:")
    #     print_mat(new_mat)
    return new_mat
