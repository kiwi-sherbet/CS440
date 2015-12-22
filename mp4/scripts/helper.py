from operator import mul, add

def dot_product(ra1, ra2):
    """Performs the dot product on two 1D arrays.

    Arguments:
        ra1: the first 1D array
        ra2: the second 1D array

    Returns:
        prod: the dot product
    """
    return sum(p*q for p,q in zip(ra1, ra2))

def add_vectors(ra1, ra2):
    """Adds to vectors represented as 1D arrays.

    Arguments:
        ra1: the first 1D array
        ra2: the second 1D array

    Returns:
        sum_vector: the sum of the vectors
    """
    return map(add, ra1, ra2)

def scalar_mult(c, ra):
    """Multiplies all elements in an array by a scalar.

    Arguments:
        c: the constant to multiply
        ra: the array to multiply

    Returns:
        result_ra: the resulting array
    """
    return map(mul, [c for a in range(0, len(ra))], ra)

def compute_rate_correct(estimated_labels, actual_labels, num_classes):
    """Compares actual vs estimated labels to compute the rate
        estimated correctly.

    Arguments:
        estimated_labels: list of labels estimated from testing
        actual_labels: list of given actual labels

    Returns:
        classification_rate: the percentage of test images classified
            correctly
    """
    overall_accuracy = 0
    classification_rates = [0 for x in range(0,num_classes)]
    test_ex_per_class_ra = [0 for x in range(0,num_classes)]
    for i in range(0,len(estimated_labels)):
        actual_label = get_class(int(actual_labels[i]))
        test_ex_per_class_ra[actual_label] += 1
        if int(estimated_labels[i]) == actual_label:
            classification_rates[actual_label] += 1
            overall_accuracy += 1
    overall_accuracy /= float(sum(test_ex_per_class_ra))
    for i,count in enumerate(test_ex_per_class_ra):
        classification_rates[i] /= float(test_ex_per_class_ra[i])
    return classification_rates, overall_accuracy

def print_confusion_matrix(matrix, num_classes):
    """Prints a 10x10 matrix of confusion rates between classes.

    Arguments:
        matrix: the 10x10 matrix of confusion rates
    """
    for i in range(0,num_classes):
        for j in range(0,num_classes):
            print('{0:0.10f}'.format(matrix[i][j])),
        print('')

def compute_confusion_matrix(estimated_labels, actual_labels, num_classes):
    """Computes a matrix of the confusion rates between classes.

    Arguments:
        estimated_labels: list of labels estimated from testing
        actual_labels: list of given actual labels

    Returns:
        confusion_matrix: 10x10 matrix of confusion rates between
            classes, where the row (first) index of the matrix is the
            actual class of an image and the column (second) index is
            the class it was classified as
    """
    confusion_matrix = [[0 for j in range(0,num_classes)] for i in range(0,num_classes)]
    test_images_per_class_ra = [0 for cls in range(0,num_classes)]
    for classification in actual_labels:
        cls = int(classification)
        test_images_per_class_ra[cls] += 1
    for x in range(0,len(estimated_labels)):
        i = int(actual_labels[x])
        j = int(estimated_labels[x])
        confusion_matrix[i][j] += 1
    for i in range(0,num_classes):
        for j in range(0,num_classes):
            confusion_matrix[i][j] /= float(test_images_per_class_ra[i])
    return confusion_matrix

def get_class(i):
    """Returns the class index within a data array.

    Arguments:
        i: class label in data file

    Returns:
        0 if label is -1, else the original label
    """
    if i == -1:
        return 0
    return i

def read_file_into_line_ra(fil):
    """Parses a file into an array of lines without newlines.

    Arguments:
        fil: file with lines to read

    Returns:
        lines: array of stripped lines from the file
    """
    lines = None
    with open(fil) as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def read_file_into_char_ra(fil):
    """Parses a file into a 2D character array.

    Arguments:
        fil: file to read

    Returns:
        char_ra: 2D character array where rows are lines
    """
    char_ra = None
    with open(fil) as f:
        char_ra = [list(line) for line in f.readlines()]
    return char_ra

def print_list(l):
    """Prints each item in a list on its own line.

    Arguments:
        l: the list to print
    """
    for item in l:
        print('{0:0.10f}'.format(item)),
    print('')

def print_2d_array(ra):
    """Prints an ra in 2D.

    Arguments:
        ra: the 2D array to print
    """
    for row in ra:
        line = ''
        for col in row:
            line += str(col)
        print(line.rstrip('\n'))