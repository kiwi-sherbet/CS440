#!/usr/bin/python3

import argparse
import heapq
import os
from collections import deque

CWD = os.path.dirname(os.path.abspath(__file__))
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('maze_file',
                        help = 'file containing ASCII maze',
                        nargs = '?')
def parse_cl_args():
    """Parse and return command-line arguments.

    Returns:
        maze_file: The name of the input maze file.
        alg: The search algorithm to use.
    """
    cl_args = arg_parser.parse_args()
    maze_file = CWD + '/' + os.path.relpath(cl_args.maze_file, CWD)
    return maze_file

def read_file(fil):
    """Reads a file and returns a list of the lines in it.

    Args:
        fil: The name of the file to read.

    Returns:
        lines: The list of lines from the file.

    """
    lines = []
    with open(fil) as f:
        lines = f.readlines()
    return lines


