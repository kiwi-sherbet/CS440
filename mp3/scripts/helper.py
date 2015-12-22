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