#!/usr/bin/python

import argparse
import math
import os
import sys
from helper import *

CWD = os.path.dirname(os.path.abspath(__file__))
V = 2
FACE = True

if FACE:
    IMG_WIDTH = 61
    IMG_HEIGHT = 70
    NUM_CLASSES = 2
    NUM_TRAINING_EX = 451
    NUM_TESTING_EX = 150
    TRAINLABEL = '../data/facedata/facedatatrainlabels'
    TRAINDATA = '../data/facedata/facedatatrain'
    TESTLABEL = '../data/facedata/facedatatestlabels'
    TESTDATA = '../data/facedata/facedatatest'
else:
    IMG_WIDTH = 28
    IMG_HEIGHT = 28
    NUM_CLASSES = 10
    NUM_TRAINING_EX = 5000
    NUM_TESTING_EX = 1000
    TRAINLABEL = '../data/digitdata/traininglabels'
    TRAINDATA = '../data/digitdata/trainingimages'
    TESTLABEL = '../data/digitdata/testlabels'
    TESTDATA = '../data/digitdata/testimages'



argument_parser = argparse.ArgumentParser();
argument_parser.add_argument('k',
                             help='k constant for Laplace smoothing',
                             nargs='?',
                             default='1')
argument_parser.add_argument('traininglabels',
                             help='file with training labels of images',
                             nargs='?',
                             default= TRAINLABEL )
argument_parser.add_argument('trainingimages',
                             help='file with training images',
                             nargs='?',
                             default= TRAINDATA )
argument_parser.add_argument('testimages',
                             help='file with test images',
                             nargs='?',
                             default= TESTDATA )
argument_parser.add_argument('testlabels',
                             help='file with test labels',
                             nargs='?',
                             default= TESTLABEL )

def parse_cl_args():
    """Parses command-line arguments.

    Returns:
        k: constant for Laplace smoothing
        training_labels_file: path to traininglabels file
        training_images_file: path to trainingimages file
        test_images_file: path to testimages file
        test_labels_file: path to testlabels file
    """
    cl_args = argument_parser.parse_args()
    training_labels_file = CWD + '/' + os.path.relpath(cl_args.traininglabels, CWD)
    training_images_file = CWD + '/' + os.path.relpath(cl_args.trainingimages, CWD)
    test_images_file = CWD + '/' + os.path.relpath(cl_args.testimages, CWD)
    test_labels_file = CWD + '/' + os.path.relpath(cl_args.testlabels, CWD)
    k = 1
    try:
        k = int(cl_args.k)
    except:
        print('Inputted k is not an integer. Defaulting to k=1.')
    return k, training_labels_file, training_images_file, test_images_file, test_labels_file

def compute_priors(training_labels_file):
    """Computes P(class) and the total number of training examples per
        class, where class is the digits 0-9.

    Arguments:
        training_labels_file: file containing one label per
            line for each 28x28 pixel image

    Returns:
        priors_ra: length-10 array, where value at index i is
            prior for digit i
        ex_per_class_ra: length-10 array, where value at index i is
            number of training examples in class i
    """
    lines = read_file_into_line_ra(training_labels_file)
    ex_per_class_ra = [0 for i in range(0,NUM_CLASSES)]
    num_training_ex = 0
    for line in lines:
        cls = int(line)
        ex_per_class_ra[cls] += 1
        num_training_ex += 1
    priors_ra = [float(num)/num_training_ex for num in ex_per_class_ra]
    return priors_ra, ex_per_class_ra

def is_foreground(c):
    """Returns true if the character is '+' or '#'.

    Arguments:
        c: the character to check

    Returns:
        True if c is '+' or '#'
    """
    return c == '+' or c == '#'

def compute_tot_foreground(training_labels_file, training_images_file):
    """Computes the total number of training examples that are in
        class cls and have a foreground value at coordinate (i,j).

    Arguments:
        training_labels_file: file with one actual label per line for
            each 28x28 pixel image
        training_images_file: file with 28x28 pixel images

    Returns:
        total_foreground_ra: ixjxcls array of the total number of
            training examples in class cls that have a foreground value at
            each (i,j)
        total_background_ra: ixjxcls array of the total number of
            training examples in class cls that have a background value at
            each (i,j)
    """
    all_labels = read_file_into_line_ra(training_labels_file)
    all_images = read_file_into_char_ra(training_images_file)
    total_foreground_ra = [[[0 for cls in range(0,NUM_CLASSES)] for j in range(0,IMG_WIDTH)] for i in range(0,IMG_HEIGHT)]
    total_background_ra = [[[0 for cls in range(0,NUM_CLASSES)] for j in range(0,IMG_WIDTH)] for i in range(0,IMG_HEIGHT)]
    for x in range(0,NUM_TRAINING_EX):
        cls = int(all_labels[x])
        curr_img = all_images[IMG_HEIGHT*x:IMG_HEIGHT*x+IMG_HEIGHT][0:IMG_HEIGHT]
        for i in range(0,IMG_HEIGHT):
            for j in range(0,IMG_WIDTH):
                if is_foreground(curr_img[i][j]):
                    total_foreground_ra[i][j][cls] += 1
                else:
                    total_background_ra[i][j][cls] += 1
    return total_foreground_ra, total_background_ra

def compute_likelihood(total_foreground_ra, total_background_ra, ex_per_class_ra, k):
    """Computes the likelihood of each (i,j) being foreground within a class cls.

    Arguments:
        total_foreground_ra: ixjxcls array of the total number of
            training examples in class cls that have a foreground value at
            each coordinate (i,j)
        total_background_ra: ixjxcls array of the total number of
            training examples in class cls that have a background value at
            each coordinate (i,j)
        ex_per_class_ra: length-10 array, where value at index i is
            number of training examples in class i
        k: constant for Laplace smoothing

    Returns:
        foreground_likelihood_ra: ixjxcls array of the likelihood for an example
            within class cls to have a foreground value at (i,j)
        background_likelihood_ra: ixjxcls array of the likelihood for an example
            within class cls to have a background value at (i,j)
    """
    foreground_likelihood_ra = total_foreground_ra
    background_likelihood_ra = total_background_ra
    for cls in range(0,NUM_CLASSES):
        num_ex_per_class = ex_per_class_ra[cls]
        for i in range(0,IMG_HEIGHT):
            for j in range(0,IMG_WIDTH):
                foreground_likelihood_ra[i][j][cls] = (float(foreground_likelihood_ra[i][j][cls]) + k) / (float(num_ex_per_class) + k*V)
                background_likelihood_ra[i][j][cls] = (float(background_likelihood_ra[i][j][cls]) + k) / (float(num_ex_per_class) + k*V)
    # for i in range(0,IMG_WIDTH):
    #     for j in range(0, IMG_HEIGHT):
    #         for cls in range (0, NUM_CLASSES):
    #             print (foreground_likelihood_ra[i][j][cls], background_likelihood_ra[i][j][cls])
    return foreground_likelihood_ra, background_likelihood_ra

def training(training_labels_file, training_images_file, k):
    """Performs training by estimating the likelihoods
        P(F={background,foreground} | class={0..9})

    Arguments:
        training_labels_file: file containing one label per line for
            each 28x28 pixel image
        training_images_file: file with 28x28 pixel images
        k: constant for Laplace smoothing

    Returns:
        priors_ra: length-10 array, where value at index if is prior
            for digit i
        foreground_likelihood_ra: ixjxcls array of the likelihood for
            an example within class cls to have a foreground value at
            (i,j)
        background_likelihood_ra: ixjxcls array of the likelihood for
            an example within class cls to have a background value at
            (i,j)
    """
    priors_ra, ex_per_class_ra = compute_priors(training_labels_file)
    total_foreground_ra, total_background_ra = compute_tot_foreground(training_labels_file, training_images_file)
    foreground_likelihood_ra, background_likelihood_ra = compute_likelihood(total_foreground_ra, total_background_ra, ex_per_class_ra, k)
    return priors_ra, foreground_likelihood_ra, background_likelihood_ra

def testing(priors_ra, foreground_likelihood_ra, background_likelihood_ra, test_images_file, test_labels_file):
    """Performs maximum a priori (MAP) classification of test images
        according to the learned Naive Bayes model.

    Arguments:
        priors_ra: length-10 array, where value at index i is prior for digit i
        foreground_likelihood_ra: ixjxcls array of the likelihood for an example within class cls to have a foreground value at (i,j)
        background_likelihood_ra: ixjxcls array of the likelihood for an example within class cls to have a background value at (i,j)
        test_images_file: file with 28x28 pixel images for testing classification
        test_labels_file: file with one actual label per line for each test image

    Returns:
        estimated_labels: list of labels estimated from testing
        highest_posterior_images_ra: length-10 list of 28x28 pixel
            images, where the image at index i has the highest posterior
            for class i
        lowest_posterior_images_ra: length-10 list of 28x28 pixel
            images, where the image at index i has the lowest
            posterior for class i
    """
    all_images = read_file_into_char_ra(test_images_file)
    estimated_labels = [-1 for x in range(0,NUM_TESTING_EX)]
    actual_labels = read_file_into_line_ra(test_labels_file)
    highest_posterior_images_ra = [None for x in range(0,NUM_CLASSES)]
    lowest_posterior_images_ra = [None for x in range(0,NUM_CLASSES)]
    highest_posterior = [-sys.maxint for x in range(0,NUM_CLASSES)]
    lowest_posterior = [sys.maxint for x in range(0,NUM_CLASSES)]
    for x in range(0,NUM_TESTING_EX):
        curr_img = all_images[IMG_HEIGHT*x:IMG_HEIGHT*x+IMG_HEIGHT][0:IMG_HEIGHT]
        curr_label = int(actual_labels[x])
        posteriors_ra = [0 for cls in range(0,NUM_CLASSES)]
        for cls in range(0,NUM_CLASSES):
            posterior = math.log(priors_ra[cls])
            for i in range(0,IMG_HEIGHT):
                for j in range(0,IMG_WIDTH):
                    if is_foreground(curr_img[i][j]):
                        posterior += math.log(foreground_likelihood_ra[i][j][cls])
                    else:
                        posterior += math.log(background_likelihood_ra[i][j][cls])
            posteriors_ra[cls] = posterior
        estimated_labels[x] = posteriors_ra.index(max(posteriors_ra))
        if max(posteriors_ra) > highest_posterior[curr_label]:
            highest_posterior[curr_label] = max(posteriors_ra)
            highest_posterior_images_ra[curr_label] = curr_img
        if max(posteriors_ra) < lowest_posterior[curr_label]:
            lowest_posterior[curr_label] = max(posteriors_ra)
            lowest_posterior_images_ra[curr_label] = curr_img
    return estimated_labels, highest_posterior_images_ra, lowest_posterior_images_ra

def compute_rate_correct(estimated_labels, actual_labels, num_classes):
    """Compares actual vs estimated labels to compute the rate estimated correctly.

    Arguments:
        estimated_labels: list of labels estimated from testing
        actual_labels: list of given actual labels

    Returns:
        classification_rate: the percentage of test images classified correctly
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
            actual class of an image and the column (second) index is the
            class it was classified as
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

def print_image(char_ra):
    """Prints a 28x28 character array representing an image.

    Arguments:
        char_ra: the 28x28 ASCII image to print
    """
    for i in range(0,IMG_HEIGHT):
        for j in range(0,IMG_WIDTH):
            print(char_ra[i][j]),
        print('')

def print_cube(cube):
    """Prints a 28x28x10 3D array.

    Arguments:
        cube: the 28x28x10 3D array.
    """
    for cls in range(0,NUM_CLASSES):
        print('THIS IS CLASS ' + str(cls) + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        for i in range(0,IMG_HEIGHT):
            for j in range(0,IMG_WIDTH):
                print(cube[i][j][cls]),
            print('')
        print('\n\n')

def find_coordinates_of_max_in_2d_ra(ra):
    """Finds four non-diagonal coordinates in a 10x10 matrix with the largest values.

    Arguments:
        ra: the 10x10 matrix

    Returns:
        max_confusion_rate_coor_ra: length-4 list of tuples
            representing the coordinates in the array with the largest
            values
    """
    max_confusion_rate_coor_ra = [-sys.maxint for x in range(0,4)]
    max_confusion_rate_val_ra = [-sys.maxint for x in range(0,4)]
    filled_count = 0
    for i in range(0,NUM_CLASSES):
        for j in range(0,NUM_CLASSES):
            if i == j:
                continue
            if filled_count < 4:
                max_confusion_rate_coor_ra[filled_count] = (i,j)
                max_confusion_rate_val_ra[filled_count] = ra[i][j]
                filled_count += 1
            elif float(ra[i][j]) > min(max_confusion_rate_val_ra):
                min_idx = max_confusion_rate_val_ra.index(min(max_confusion_rate_val_ra))
                max_confusion_rate_coor_ra[min_idx] = (i,j)
                max_confusion_rate_val_ra[min_idx] = ra[i][j]
    return max_confusion_rate_coor_ra

def print_likelihoods_odds(tup, foreground_likelihood_ra):
    """Prints the likelihood maps for both classes in the pair with
        confusion rate, then prints their odds ratio map.

    Arguments:
        tup: tuple representing the actual class and the class an
            image was confused for
        foreground_likelihood_ra: ixjxcls array of the likelihood for
            an example within class cls to have a foreground value at
            (i,j)
    """
    print('Actual class (' + str(tup[0]) + '): ')
    for i in range(0,IMG_HEIGHT):
        for j in range(0,IMG_WIDTH):
            log_likelihood = math.log(foreground_likelihood_ra[i][j][tup[0]])
            if log_likelihood > 0:
                print('!!!'),
            elif log_likelihood > -1:
                print('#'),
            elif log_likelihood > -2:
                print('+'),
            elif log_likelihood > -4:
                print('-'),
            else:
                print(' '),
        print('')
    print('Class confused for (' + str(tup[1]) + '): ')
    for i in range(0,IMG_HEIGHT):
        for j in range(0,IMG_WIDTH):
            log_likelihood = math.log(foreground_likelihood_ra[i][j][tup[1]])
            if log_likelihood > 0:
                print('!!!'),
            elif log_likelihood > -1:
                print('#'),
            elif log_likelihood > -2:
                print('+'),
            elif log_likelihood > -4:
                print('-'),
            else:
                print(' '),
        print('')
    print('Odds ratio:')
    for i in range(0,IMG_HEIGHT):
        for j in range(0,IMG_WIDTH):
            odds_ratio = math.log(foreground_likelihood_ra[i][j][tup[0]]) - math.log(foreground_likelihood_ra[i][j][tup[1]])
            if odds_ratio > 0.8:
                print('#'),
            elif odds_ratio > 0:
                print('+'),
            else:
                print(' '),
        print('')

def main(args):
    k, training_labels_file, training_images_file, test_images_file, test_labels_file = parse_cl_args()
    priors_ra, foreground_likelihood_ra, background_likelihood_ra = training(training_labels_file, training_images_file, k)
    estimated_labels, highest_posterior_images_ra, lowest_posterior_images_ra = testing(priors_ra, foreground_likelihood_ra, background_likelihood_ra, test_images_file, test_labels_file)
    actual_labels = read_file_into_line_ra(test_labels_file)

    classification_rates, overall_accuracy = compute_rate_correct(estimated_labels, actual_labels, NUM_CLASSES)
    print('Overall accuracy: ' + str(overall_accuracy))
    print('Classification rates by class: ')
    print_list(classification_rates)
    print('')

    confusion_matrix = compute_confusion_matrix(estimated_labels, actual_labels, NUM_CLASSES)
    print('Confusion matrix:')
    print_confusion_matrix(confusion_matrix, NUM_CLASSES)
    print('')

    print('Test examples with highest posterior:')
    for cls in range(0,NUM_CLASSES):
        print('===========================')
        print('Class ' + str(cls))
        print_image(highest_posterior_images_ra[cls])
    print('Test examples with lowest posterior:')
    print('')

    for cls in range(0,NUM_CLASSES):
        print('===========================')
        print('Class ' + str(cls))
        print_image(lowest_posterior_images_ra[cls])
    print('')

    maxs_ra = find_coordinates_of_max_in_2d_ra(confusion_matrix)
    print('Tuples of (actual_class, confused_class) with highest confusion rate: ')
    for tup in maxs_ra:
        print(tup)

    for tup in maxs_ra:
        print_likelihoods_odds(tup, foreground_likelihood_ra)


if __name__ == '__main__':
    main(sys.argv)