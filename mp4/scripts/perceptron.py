#!/usr/bin/python

import matplotlib.pyplot as plt

import argparse
import os
import random
import sys
from nearest_neighbor import training as get_train_examples
from helper import dot_product, add_vectors, scalar_mult, read_file_into_line_ra, compute_rate_correct, print_confusion_matrix, compute_confusion_matrix, print_2d_array

CWD = os.path.dirname(os.path.abspath(__file__))
IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 10

# Parameters for tuning.
ZERO_WEIGHT_INIT = True
FIX_EX_ORDER = True
NUM_EPOCHS = 2
NO_BIAS = True
TERNARY = False

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('zero_weight_init',
                             help = 'True to initialize weights to 0',
                             nargs = '?',
                             default = True)
argument_parser.add_argument('fix_train_ex_order',
                             help = 'True to fix training example order',
                             nargs = '?',
                             default = True)
argument_parser.add_argument('num_epochs',
                             help = 'number of epochs to train perceptron with',
                             nargs = '?',
                             default = 10)
argument_parser.add_argument('no_bias',
                             help = 'True to use no bias in weight vectors',
                             nargs = '?',
                             default = True)
argument_parser.add_argument('train_label_file',
                             help = 'file with training image labels',
                             nargs = '?',
                             default = '../data/digitdata/traininglabels')
argument_parser.add_argument('train_image_file',
                             help = 'file with training images',
                             nargs = '?',
                             default = '../data/digitdata/trainingimages')
argument_parser.add_argument('test_label_file',
                             help = 'file with testing image labels',
                             nargs = '?',
                             default = '../data/digitdata/testlabels')
argument_parser.add_argument('test_image_file',
                             help = 'file with testing images',
                             nargs = '?',
                             default = '../data/digitdata/testimages')

def parse_cl_args():
    """Parses command-line arguments.

    Returns:
        train_label_file: path to traininglabels file
        train_image_file: path to trainingimages file
        test_label_file: path to testlabels file
        test_image_file: path to testimages file
    """
    global ZERO_WEIGHT_INIT, FIX_EX_ORDER, NO_BIAS
    cl_args = argument_parser.parse_args()
    train_label_file = CWD + '/' + os.path.relpath(cl_args.train_label_file, CWD)
    train_image_file = CWD + '/' + os.path.relpath(cl_args.train_image_file, CWD)
    test_label_file = CWD + '/' + os.path.relpath(cl_args.test_label_file, CWD)
    test_image_file = CWD + '/' + os.path.relpath(cl_args.test_image_file, CWD)
    ZERO_WEIGHT_INIT = False if cl_args.zero_weight_init=='False' else True
    FIX_EX_ORDER = False if cl_args.fix_train_ex_order=='False' else True
    NO_BIAS = False if cl_args.no_bias=='False' else True
    return train_label_file, train_image_file, test_label_file, test_image_file

def to_1D(ra):
    """Converts a 2D array into a 1D array.

    Arguments:
        ra: the 2D array to convert

    Returns:
        ra_1d: the converted 1D array
    """
    ra_1d = []
    for i in range(0, len(ra)):
        for j in range(0, len(ra[0])):
            if ra[i][j] == ' ': ra_1d.append(0)
            elif ra[i][j] == '+': ra_1d.append(1)
            elif ra[i][j] == '#' and TERNARY: ra_1d.append(2)
            elif ra[i][j] == '#' and not TERNARY: ra_1d.append(1)
    return ra_1d

def training(train_pts, num_classes, img_height, img_width,
             zero_weight_init, fix_ex_order, num_epochs):
    """Performs perceptron training by initializing weight vectors for
        each class, cycling through the training examples during each
        epoch, and updating the weights if it classified a training
        example incorrectly.

    Arguments:
        train_pts: the array of examples into which the training
            examples were stored
        num_classes: number of classes to categorize images into
        img_height: height of the images
        img_width: width of the images
        zero_weight_init: True if weight vectors should be initialized
            to zeros
        fix_ex_order: True if the order in which the algorithm
            iterates through the training examples should be fixed
        num_epochs: number of epochs to train with

    Returns:
        w_c: the weight vectors for each class c
        
    """
    w_c = []
    if zero_weight_init:
        w_c = [[0 for j in range(0, img_height*img_width)] for i in
                range(0, num_classes)]
    else: # Random weight initialized in [0,1).
        w_c = [[random.random() for j in range(0,
                img_height*img_width)] for i in range(0, num_classes)]
    b = 0 if NO_BIAS else 1
    num_train_ex = len(train_pts)
    for epoch in range(0, num_epochs):
        print('Epoch: ' + str(epoch) + '. Accuracy:'),
        num_incorrect = 0
        alpha = 1000/(1000+epoch)
        if not fix_ex_order:
            random.shuffle(train_pts)
        for ex in train_pts:
            maxs = {}
            maxs['max_decision_val'] = -sys.maxint
            maxs['max_decision_val_class'] = -1
            for c, weight_vector in enumerate(w_c):
                decision_val = dot_product(weight_vector, to_1D(ex['image']))
                if decision_val > maxs['max_decision_val']:
                    maxs['max_decision_val'] = decision_val
                    maxs['max_decision_val_class'] = c
            estimated_label = maxs['max_decision_val_class']
            if not estimated_label == ex['label']:
                num_incorrect += 1
                w_c[ex['label']] = add_vectors(w_c[ex['label']], scalar_mult(alpha, to_1D(ex['image'])))
                w_c[estimated_label] = add_vectors(w_c[estimated_label], scalar_mult(-alpha, to_1D(ex['image'])))
        print((num_train_ex - num_incorrect) / float(num_train_ex))
    return w_c

def testing(w_c, test_label_file, test_image_file, img_height):
    """Performs testing of perceptron on the test set.

    Arguments:
        w_c: the weight vectors for each class c
        test_label_file: file containing the labels of the test images
        test_image_file: file containing the test images
        img_height: height of the images

    Returns:
        estimated_labels: array of labels estimated for the test examples
        actual_labels: array of actual labels for the test examples
    """
    estimated_labels = []
    actual_labels = read_file_into_line_ra(test_label_file)
    with open(test_image_file) as f:
        for idx_test_ex in range(0, len(actual_labels)):
            curr_image = []
            while len(curr_image) < img_height:
                curr_image.append([c for c in f.readline()])
            maxs = {}
            maxs['max_decision_val'] = -sys.maxint
            maxs['max_decision_val_class'] = -1
            for c, weight_vector in enumerate(w_c):
                decision_val = dot_product(weight_vector, to_1D(curr_image))
                if decision_val > maxs['max_decision_val']:
                    maxs['max_decision_val'] = decision_val
                    maxs['max_decision_val_class'] = c
            estimated_labels.append(maxs['max_decision_val_class'])
    return estimated_labels, actual_labels

def show_session_info(ZERO_WEIGHT_INIT, FIX_EX_ORDER, NUM_EPOCHS, NO_BIAS):
    print('Initialize weights to zero: ' + str(ZERO_WEIGHT_INIT))
    print('Fix training examples order: ' + str(FIX_EX_ORDER))
    print('Number of epochs for training: ' + str(NUM_EPOCHS))
    print('Use bias: ' + str(not NO_BIAS))

def forceAspect(ax,aspect=1):
    """Force a plot with different x- and y-axis ranges to be square.

    Source:
    http://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
    """
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def main():
    (train_label_file, train_image_file, test_label_file,
            test_image_file) = parse_cl_args()
    show_session_info(ZERO_WEIGHT_INIT, FIX_EX_ORDER, NUM_EPOCHS, NO_BIAS)

    train_pts = get_train_examples(train_label_file, train_image_file,
            IMG_HEIGHT)
    w_c = training(train_pts, NUM_CLASSES,
             IMG_HEIGHT, IMG_WIDTH, ZERO_WEIGHT_INIT, FIX_EX_ORDER, NUM_EPOCHS)
    print_2d_array(w_c)

    estimated_labels, actual_labels = testing(w_c, test_label_file, test_image_file, IMG_HEIGHT)
    classification_rates, overall_accuracy = compute_rate_correct(estimated_labels, actual_labels, NUM_CLASSES)
    print('Classification rates:')
    print(classification_rates)
    print('Overall accuracy:'),
    print(overall_accuracy)

    confusion_matrix = compute_confusion_matrix(estimated_labels, actual_labels, NUM_CLASSES)
    print('Confusion matrix:')
    print_confusion_matrix(confusion_matrix, NUM_CLASSES)
    print('')

    # Reference: matplotlib.org/examples/pylab_examples/colorbar_tick_labelling_demo.html
    fig, ax = plt.subplots()
    cax = ax.imshow(w_c)
    forceAspect(ax, aspect=1)
    cbar = fig.colorbar(cax)
    plt.show()
    

if __name__ == '__main__':
    main()
