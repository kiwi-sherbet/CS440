#!/usr/bin/python

import argparse
import math
import os
import sys
from helper import *
from digits import compute_rate_correct, compute_confusion_matrix, print_confusion_matrix

CWD = os.path.dirname(os.path.abspath(__file__))
NUM_CLASSES = -1
MAX_OCCURRENCES = 210

argument_parser = argparse.ArgumentParser();
argument_parser.add_argument('bayesmodel',
                             help='Naive Bayes model to use',
                             nargs='?',
                             default='multinomial')
argument_parser.add_argument('k',
                             help='k constant for Laplace smoothing',
                             nargs='?',
                             default='1')
argument_parser.add_argument('trainingfile',
                             help='file with training bags of words and labels',
                             nargs='?',
                             default='../data/spam_detection/train_email.txt')
argument_parser.add_argument('testfile',
                             help='file with test bags of words and actual labels',
                             nargs='?',
                             default='../data/spam_detection/test_email.txt')
argument_parser.add_argument('num_classes',
                             help='number of classes/categories',
                             nargs='?',
                             default=-1)


class pixel_group:
    def __init__(self, v):
        self.number = 2**(N*M)
        self.count = [0 for i in range(0, v)]

def parse_cl_args():
    """Parses command-line arguments.

    Returns:
        k: constant for Laplace smoothing
        training_file: path to training data file
        test_file: path to test data file
        bayes_model: Naive Bayes model to use
    """
    cl_args = argument_parser.parse_args()
    k = 1
    try:
        k = int(cl_args.k)
    except:
        print('Inputted k is not an integer. Defaulting to k=1.')
    training_file = CWD + '/' + os.path.relpath(cl_args.trainingfile, CWD)
    test_file = CWD + '/' + os.path.relpath(cl_args.testfile, CWD)
    bayes_model = cl_args.bayesmodel
    num_classes = int(cl_args.num_classes)
    return k, training_file, test_file, bayes_model, num_classes

def compute_priors(training_file):
    """Computes P(class) and the total number of training examples per
        class, where class for spam detection is 0 for normal, 1 for
        spam, and for movie reviews is -1 for negative review and 1
        for positive.

    Arguments:
        training_file: file containing lines of a document's actual
            label followed by its bag of words (word:number of
            occurrences)

    Returns:
        priors_ra: length-2 array, where value at index i is piror for
            class i
        ex_per_clss_ra: length-2 array, where value at index i is
            number of training examples in class i
    """
    lines = read_file_into_line_ra(training_file)
    ex_per_class_ra = [0 for i in range(0,NUM_CLASSES)]
    num_training_ex = 0
    for line in lines:
        cls = get_class(int(line.split()[0]))
        ex_per_class_ra[cls] += 1
        num_training_ex += 1
    priors_ra = [float(num)/num_training_ex for num in ex_per_class_ra]
    return priors_ra, ex_per_class_ra

def compute_totals(training_file, ex_per_class_ra, use_bernoulli, v):
    """Computes the total number of training examples that are in
        class cls, and have k occurrences of word w_i (multinomial) or
        some/no occurrence (Bernoulli).

    Arguments:
        training_file: file containing lines of a document's label and
            bag of words
        ex_per_class_ra: length-2 array, where value at index i is
            number of training examples in class i

    Returns:
        totals_ra: n x countn x cls array of total number of training
            examples in class cls with countn occurrences of word n
    """
    totals_ra = [{} for cls in range(0,NUM_CLASSES)]
    for line in read_file_into_line_ra(training_file):
        cls = get_class(int(line.split()[0]))
        line = line.split()[1:]
        for word_count in line:
            word = word_count.split(':')[0]
            count = int(word_count.split(':')[1])
            if use_bernoulli:
                count = 1
            if word in totals_ra[cls] and count in totals_ra[cls][word]:
                totals_ra[cls][word][count] += 1
            elif word in totals_ra[cls] and count not in totals_ra[cls][word]:
                totals_ra[cls][word][count] = 1
            else:
                totals_ra[cls][word] = {}
                totals_ra[cls][word][count] = 1
            for c in range(0,NUM_CLASSES):
                if c == cls:
                    continue
                if word not in totals_ra[c]:
                    totals_ra[c][word] = {}
                    for k in range(0,v):
                        totals_ra[c][word][k] = 0
    return totals_ra

def compute_likelihood(totals_ra, ex_per_class_ra, k, v):
    """Computes the likelihood of each word n occurring countn times
        within a class cls.

    Arguments:
        totals_ra: n x countn x cls array of total number of training
            examples in class cls with countn occurrences of word n
        ex_per_class_ralength-2 array, where value at index i is
            number of training examples in class i
        k: constant for Laplace smoothing
        v: total number of values a random variable can take on

    Returns:
        likelihood_ra: n x countn x cls array of likelihoods
    """
    top_twenty_likelihood_words = [['' for y in range(0,20)] for x in range(0,NUM_CLASSES)]
    top_twenty_likelihood_values = [[-sys.maxint for y in range(0,20)] for x in range(0,NUM_CLASSES)]
    filled_count = [0 for x in range(0,NUM_CLASSES)]
    for cls in range(0,NUM_CLASSES):
        num_ex_per_class = ex_per_class_ra[cls]
        for word in totals_ra[cls]:
            if not word in totals_ra[cls]:
                totals_ra[cls][word] = {}
            max_likelihood_value = -sys.maxint
            for count in range(0,v):
                if count in totals_ra[cls][word]:
                    totals_ra[cls][word][count] = (float(totals_ra[cls][word][count]) + k) / (float(num_ex_per_class) + k*v)
                else:
                    totals_ra[cls][word][count] = float(k) / (float(num_ex_per_class) + k*v)
                if totals_ra[cls][word][count] > max_likelihood_value:
                    max_likelihood_value = totals_ra[cls][word][count]
            if filled_count[cls] < 20:
                top_twenty_likelihood_values[cls][filled_count[cls]] = max_likelihood_value
                top_twenty_likelihood_words[cls][filled_count[cls]] = word
                filled_count[cls] += 1
            elif max_likelihood_value > min(top_twenty_likelihood_values[cls]):
                min_idx = top_twenty_likelihood_values[cls].index(min(top_twenty_likelihood_values[cls]))
                top_twenty_likelihood_values[cls][min_idx] = max_likelihood_value
                top_twenty_likelihood_words[cls][min_idx] = word
    return totals_ra, top_twenty_likelihood_words

def multinomial_training(training_file, k):
    """Performs training by estimating the likelihoods P(W_i=k |
        class), where class={0,1} for spam training and class={-1,1}
        for movie review training.

    Arguments:
        training_file: file containing lines of a document's actual label followed by its bag of words
        k: constant for Laplace smoothing

    Returns:
        ex_per_class_ra: array of length NUM_CLASSES, where value at
            index i is the number of training examples in class i
        priors_ra: array of length NUM_CLASSES, where value at index i
            is prior for class i
        likelihood_ra: matrix of multinomial likelihoods
    """
    use_bernoulli = False
    priors_ra, ex_per_class_ra = compute_priors(training_file)
    totals_ra = compute_totals(training_file, ex_per_class_ra, use_bernoulli, MAX_OCCURRENCES)
    likelihood_ra, top_twenty_likelihood_words = compute_likelihood(totals_ra, ex_per_class_ra, k, MAX_OCCURRENCES)
    return ex_per_class_ra, priors_ra, likelihood_ra, top_twenty_likelihood_words

def bernoulli_training(training_file, k):
    """Performs training by estimating the likelihoods P(W_i=0 |
        class) and P(W_i=1 | class), where class={0,1} for spam
        training and class={-1,1} for movie review training.

    Arguments:
        training_file: file containing lines of a document's actual label followed by its bag of words
        k: constant for Laplace smoothing

    Returns:
        ex_per_class_ra: array of length NUM_CLASSES, where value at
            index i is the number of training examples in class i
        priors_ra: array of length NUM_CLASSES, where value at index i
            is prior for class i
        likelihood_ra: matrix of bernoulli likelihoods
    """
    use_bernoulli = True
    priors_ra, ex_per_class_ra = compute_priors(training_file)
    totals_ra = compute_totals(training_file, ex_per_class_ra, use_bernoulli, 2)
    likelihood_ra, top_twenty_likelihood_words = compute_likelihood(totals_ra, ex_per_class_ra, k, 2)
    return ex_per_class_ra, priors_ra, likelihood_ra, top_twenty_likelihood_words

def testing(ex_per_class_ra, priors_ra, likelihood_ra, test_file, use_bernoulli):
    """Performs maximum a priori (MAP) classification of bag-of-words
        documents according to either the multinomial or Bernoulli
        learned Naive Bayes model.

    Arguments:
        ex_per_class_ra: array of length NUM_CLASSES, where value at
            index i is the number of training examples in class i
        priors_ra: array of length NUM_CLASSES, where value at index i
            is prior for document class i
        likelihood_ra: NUM_CLASSESx(number of words)xv matrix of
            likelihoods
        test_file: test examples file containing lines of a document's
            actual label followed by its bag of words
        k: Laplace smoothing constant
        v: total number of values a random variable can take on; 2 for
            Bernoulli, MAX_OCCURRENCES for multinomial
        use_bernoulli: True if Bernoulli model is being used, else False

    Returns:
        estimated_labels: array of estimated labels
        actual_labels: array of corresponding actual labels
    """        
    estimated_labels = []
    actual_labels = []
    for line in read_file_into_line_ra(test_file):
        actual_labels.append(get_class(int(line.split()[0])))
        line = line.split()[1:]
        posteriors_ra = [0 for cls in range(0,NUM_CLASSES)]
        for cls in range(0,NUM_CLASSES):
            posterior = math.log(priors_ra[cls])
            for word_count in line:
                word = word_count.split(':')[0]
                count = int(word_count.split(':')[1])
                if use_bernoulli:
                    count = 1
                if word in likelihood_ra[cls]: # If word not in likelihood_ra[cls]
                    if count in likelihood_ra[cls][word]:
                        posterior += math.log(likelihood_ra[cls][word][count])
                    else:
                        pass
                else:
                    pass
            posteriors_ra[cls] = posterior
        estimated_labels.append(posteriors_ra.index(max(posteriors_ra)))
    return estimated_labels, actual_labels

def main():
    global NUM_CLASSES
    k, training_file, test_file, bayes_model, num_classes = parse_cl_args()
    NUM_CLASSES = num_classes
    priors_ra = []
    likelihood_ra = []
    ex_per_class_ra = []
    top_twenty_likelihood_words = []
    if bayes_model == 'multinomial':
        ex_per_class_ra, priors_ra, likelihood_ra, top_twenty_likelihood_words = multinomial_training(training_file, k)
        estimated_labels, actual_labels = testing(ex_per_class_ra, priors_ra, likelihood_ra, test_file, False)
    elif bayes_model == 'bernoulli':
        ex_per_class_ra, priors_ra, likelihood_ra, top_twenty_likelihood_words = bernoulli_training(training_file, k)
        estimated_labels, actual_labels = testing(ex_per_class_ra, priors_ra, likelihood_ra, test_file, True)
    else:
        print('Bad Bayes model.')
        return

    classification_rates, overall_accuracy = compute_rate_correct(estimated_labels, actual_labels, NUM_CLASSES)
    print('Overall accuracy: ' + str(overall_accuracy))
    print('Classification rates by class: ')
    print_list(classification_rates)
    print('')

    confusion_matrix = compute_confusion_matrix(estimated_labels, actual_labels, NUM_CLASSES)
    print('Confusion matrix:')
    print_confusion_matrix(confusion_matrix, NUM_CLASSES)
    print('')

    for cls in range(0, len(top_twenty_likelihood_words)):
        print('Top 20 words with highest likelihood for class ' + str(cls))
        for word in top_twenty_likelihood_words[cls]:
            print(word)
        print('')


if __name__ == '__main__':
    main()
