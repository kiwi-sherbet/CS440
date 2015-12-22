#!/usr/bin/python

import argparse
import os
import time
from helper import print_2d_array, read_file_into_line_ra, get_class, compute_rate_correct, print_confusion_matrix, compute_confusion_matrix

CWD = os.path.dirname(os.path.abspath(__file__))
IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 10

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('k',
                             help = 'number of nearest neighbors to check',
                             nargs = '?',
                             default = 4)
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
        k : number of nearest neighbors to check
    """
    cl_args = argument_parser.parse_args()
    train_label_file = CWD + '/' + os.path.relpath(cl_args.train_label_file, CWD)
    train_image_file = CWD + '/' + os.path.relpath(cl_args.train_image_file, CWD)
    test_label_file = CWD + '/' + os.path.relpath(cl_args.test_label_file, CWD)
    test_image_file = CWD + '/' + os.path.relpath(cl_args.test_image_file, CWD)
    k = int(cl_args.k)
    return train_label_file, train_image_file, test_label_file, test_image_file, k

def get_pixel_diff(p1, p2):
    """Returns the distance between two pixels, where distance is the
        difference in the intensities represented by the characters at
        each pixel. ' ' means a white/background pixel, '+' means
        gray, and '#' means black.

    Arguments:
        p1: the character at the first pixel
        p2: the character at the second pixel

    Returns:
        dist: 0 if the characters are the same, 1 if they are one
            intensity apart, i.e. white and gray or gray and black,
            and 2 if they are two intensities apart, i.e. white and
            black.
    """
    if p1 == p2:
        return 0
    if (p1==' ' and p2=='#') or (p1=='#' and p2==' '):
        return 2
    return 1

def get_img_dist(img1, img2, img_height, img_width):
    """Returns the distance between two images, where the pixels at
        (i,j) contribute 1 to the distance if they are one intensity
        apart, and 2 to the distance if they are two intensities
        apart.

    Arguments:
        img1: one image to be compared
        img2: second image to be compared
        img_height: height of the images
        img_width: width of the images

    Returns:
        dist: the distance between the two images
    """
    dist = 0
    for i in range(0, img_height):
        for j in range(0, img_width):
            dist += get_pixel_diff(img1[i][j], img2[i][j])
    return dist

def get_mode(ra):
    """Returns the label with the majority vote from an array of
        tuples of the form (distance, label).

    Arguments:
        ra: array of tuples of the form (distance, label)

    Returns:
        mode_label: the mode label in the array
    """
    track_label_counts = {}
    for ex in ra:
        curr_label = ex[1]
        if curr_label in track_label_counts:
            track_label_counts[curr_label] += 1
        else:
            track_label_counts[curr_label] = 1
    mode_label = -1
    mode_label_count = -1
    for label_count in track_label_counts:
        if track_label_counts[label_count] > mode_label_count:
            mode_label_count = track_label_counts[label_count]
            mode_label = label_count
    return mode_label

def training(train_label_file, train_image_file, img_height):
    """Performs training for nearest-neighbors by just memorizing
        training examples into an array of examples. Each example is
        represented by a dictionary with the keys 'id', 'label', and
        'image', where id i indicates that the example is the ith
        training image, label is the image's true label, and image is
        the 2D ASCII array representation of the image.

    Arguments:
        train_label_file: file containing labels for the training
            images
        train_image_file: file containing the training images
        img_height: height of the images

    Returns:
        train_pts: the array of examples into which the training
            examples were stored
    """
    train_pts = [{'id':i} for i in range(0, sum(1 for line in open(train_label_file)))]
    count_ex = 0
    with open(train_label_file) as f:
        for line in f:
            train_pts[count_ex]['idx'] = count_ex
            train_pts[count_ex]['label'] = int(line.strip())
            count_ex += 1
    count_ex = 0
    with open(train_image_file) as f:
        curr_image = []
        line_count = 0
        for line in f:
            if line_count >= img_height:
                train_pts[count_ex]['image'] = curr_image
                curr_image = []
                count_ex += 1
                line_count = 0
            curr_image.append([c for c in line])
            line_count += 1
        train_pts[count_ex]['image'] = curr_image
    return train_pts

def testing(train_pts, test_label_file, test_image_file, k, img_height, img_width):
    """Performs testing for k-NN by finding the k nearest neighbors to
        a test example and assigning to the example the majority label
        of its k nearest neighbors.

    Arguments:
        train_pts: the array of examples into which the training
            examples were stored
        test_label_file: file containing labels for the test images
        test_image_file: file containing the test images
        k: number of nearest neighbors to check
        img_height: height of the images
        img_width: width of the images

    Returns:
        estimated_labels: array of labels estimated for the test
            examples
        actual_labels: array of actual labels for the test examples
    """
    estimated_labels = []
    actual_labels = read_file_into_line_ra(test_label_file)
    with open(test_image_file) as f:
        for idx_test_ex in range(0, len(actual_labels)):
            k_neighbors = []
            curr_image = []
            while len(curr_image) < img_height:
                curr_image.append([c for c in f.readline()])
            for idx_train_ex in range(0, len(train_pts)):
                d = get_img_dist(curr_image, train_pts[idx_train_ex]['image'], img_height, img_width)
                if len(k_neighbors) < k:
                    k_neighbors.append((d, train_pts[idx_train_ex]['label']))
                elif d < max(k_neighbors):
                    replace_idx = k_neighbors.index(max(k_neighbors))
                    k_neighbors[replace_idx] = (d, train_pts[idx_train_ex]['label'])
            majority_label = get_mode(k_neighbors)
            estimated_labels.append(majority_label)
    return estimated_labels, actual_labels

def main():
    train_label_file, train_image_file, test_label_file, test_image_file, k = parse_cl_args()
    start_time = time.time()
    train_pts = training(train_label_file, train_image_file, IMG_HEIGHT)
    end_time = time.time()
    print('Training time: ' + str(end_time - start_time))
    start_time = time.time()
    estimated_labels, actual_labels = testing(train_pts, test_label_file, test_image_file, k, IMG_HEIGHT, IMG_WIDTH)
    end_time = time.time()
    print('Testing time: ' + str(end_time - start_time))

    classification_rates, overall_accuracy = compute_rate_correct(estimated_labels, actual_labels, NUM_CLASSES)
    print('Classification rates:')
    print(classification_rates)
    print('Overall accuracy:'),
    print(overall_accuracy)

    confusion_matrix = compute_confusion_matrix(estimated_labels, actual_labels, NUM_CLASSES)
    print('Confusion matrix:')
    print_confusion_matrix(confusion_matrix, NUM_CLASSES)
    print('')


if __name__ == '__main__':
    main()
