#!/usr/bin/python

# RUN: chmod u+x WarNode.py
#      ./WarNode.py ../data/Narvik.txt

import argparse
import os

class WarNode:
    def __init__(self, board, depth):
        self.game_board = board
        self.children = []
        self.parent = None
        self.depth = depth # Parameter depth is node n's n.depth, where n is the parent of the node being created.

class Place:
    def __init__(self, score):
        self.score = score
        self.resources = float(score)
        self.player = None


CWD = os.path.dirname(os.path.abspath(__file__))
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('board_file',
                        help = 'file containing one of the war game boards',
                        nargs = '?')

def parse_cl_args():
    cl_args = arg_parser.parse_args()
    board_file = CWD + '/' + os.path.relpath(cl_args.board_file)
    return board_file

def read_file_lines(fil):
    lines = []
    with open(fil) as f:
        lines = f.readlines()
    return lines

def read_board(board_file):
    board_ra = []
    lines = read_file_lines(board_file)
    for line in lines:
        row = []
        for place in line.split():
            row.append(Place(int(place)))
        board_ra.append(row)
    return board_ra


def print_board(board_ra):
    for row in board_ra:
        for col in row:
            print(col.score),
        print('')


def main():
    board_file = parse_cl_args()
    board_ra = read_board(board_file) # What we store in node n's n.game_board
    print_board(board_ra)
    root = WarNode(board_ra, 0)
    

if __name__ == '__main__':
    main()
