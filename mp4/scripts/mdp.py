#!/usr/bin/python

import argparse
import heapq
import os
import sys
from collections import deque
import pygame
import random
import copy
import math
import time
import matplotlib.pyplot as plt

RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RGB_GRAY = (125, 125, 125)
RGB_RED = (255, 0, 0)
RGB_BLUE = (0, 0, 255)
RGB_GREEN = (0, 255, 0)
RGB_LIME = (50, 205, 50)
RGB_SKYBLUE = (135, 206, 250)
RGB_ORANGE = (255, 165, 0)

pygame.font.init()
FONT = pygame.font.Font(None,50)
SMALL_FONT= pygame.font.Font(None,30)

SQUARE_SIZE = 20
PATH_RADIUS = 3
Wall_TEXTURE = pygame.image.load('../data/graphics/ChippedBricks.png')
GHOST = pygame.image.load('../data/graphics/Ghost.png')
PACMAN = pygame.image.load('../data/graphics/ming.png')
DOT = pygame.image.load('../data/graphics/Home.png')
VISITED_DOT = pygame.image.load('../data/graphics/Visited_Home.png')

DEFAULT = True

RMSE_LIST = []
if DEFAULT:
    IMG_WIDTH = 8
    IMG_HEIGHT = 8
    MAP = '../data/griddata/default'
    ITERATION_DEPTH = 50
else:
    IMG_WIDTH = 23
    IMG_HEIGHT = 8
    MAP = '../data/griddata/tinySearch'
    ITERATION_DEPTH = 200

GAMMA = 0.99

CWD = os.path.dirname(os.path.abspath(__file__))
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('grid_file',
                        help = 'file containing ASCII maze',
                        nargs = '?',
                        default = MAP)

# Environment
class grid_class:
    def __init__(self, width, height):
        self.ra = [[ None for j in range(0,width)] for i in range(0,height)]
        self.width = width
        self.height = height
        self.forward = (1, 0)
        self.backward = (-1, 0)
        self.right =  (0, 1)
        self.left = (0, -1)
        self.directions = [self.forward, self.backward, self.right, self.left]

# Agent, Transition Model
class packman_class:
    """
    State:
        pos: 
        dir:
    Transition:
        forward_prob:
        left_prob:
        right_prob
    Action:
        move:
        get_reward:
    """
    def __init__(self):
        self.pos = (0,0)
        self.dir = (0,0)
        self.set_probabilty(0.8, 0.1, 0.1)

    def set_probabilty(self, forward_val, left_val, right_val):
        self.forward_prob = forward_val
        self.left_prob = left_val
        self.right_prob = right_val

    def set_state(self, pos_coord, dir_coord):
        self.pos = pos_coord
        self.dir = dir_coord

    def randomize_position(self, grid):
        while True:
            tmp_position = (random.randint(0, grid.width -1), random.randint(0, grid.height -1))
            if can_locate_to(tmp_position, grid):
                self.pos = tmp_position
                return True

    def randomize_direction(self, grid):
        self.dir = random.choice(DIRECTIONS)

    def move(self, grid):
        candidates=[]
        curr_pos = copy.deepcopy(self.pos)
        forward_dir = copy.deepcopy(self.dir)
        forward_pos = add_tuples(curr_pos, forward_dir)
        if can_move_to(forward_pos, grid):
            candidates.append([forward_pos, self.forward_prob])
        else:
            candidates.append([curr_pos, self.forward_prob])
        left_pos = add_tuples(curr_pos, rotate_tuples(forward_dir, grid.left))
        if can_move_to(left_pos, grid):
            candidates.append([left_pos, self.left_prob])
        else:
            candidates.append([curr_pos, self.left_prob])
        right_pos = add_tuples(curr_pos, rotate_tuples(forward_dir, grid.right))
        if can_move_to(right_pos, grid):
            candidates.append([right_pos, self.left_prob])
        else:
            candidates.append([curr_pos, self.right_prob])
        self.pos = weighted_random_choice(candidates)


def weighted_random_choice(label_list):
    total_prob = 0.0
    cumm_prob = 0.0
    for item in label_list:
        total_prob += item[1]
    x = random.uniform(0, total_prob)
    for item in label_list:
        cumm_prob += item[1]
        if x < cumm_prob: break
    return copy.deepcopy(item[0])


class markers_class:
    def __init__(self):
        self.start = None
        self.walls = []
        self.small_penalty = []
        self.great_penalty = []
        self.small_reward = []
        self.great_reward = []


def iterate(grid, depth, terminal, learning):
    policy_list = []
    utility_list=[]
    if learning and terminal:
        value_list = []
        visited_ra = [[ 0 for j in range(0, grid.width)] for i in range(0, grid.height)]
        td_q_initializing(value_list, utility_list, policy_list, grid)
        for i in range(0, depth):
            td_q_learning(value_list, policy_list, visited_ra, i+1, grid)
            td_q_solution(value_list, utility_list, policy_list, grid)
    else:
        utility_list.append([[ 0 for j in range(0, grid.width)] for i in range(0, grid.height)])
        policy_list.append([[ None for j in range(0, grid.width)] for i in range(0,grid.height)])
        for i in range(0, depth):
            value_iteration(utility_list, policy_list, grid, terminal)
    return utility_list, policy_list


def compare(value_iteration_utility_ra, grid):
    td_q_learning_utility_list, td_q_learning_policy_list = iterate(grid, 5000, True, True)
    rmse_list = []
    wrong_policy_count_list = []
    for td_q_learning_utility_ra in td_q_learning_utility_list:
        square_error_sum = 0
        count = 0
        for r_idx in range(0, grid.height):
            for c_idx in range(0, grid.width):
                count +=1
                tmp_pos = (c_idx, r_idx)
                if can_move_to(tmp_pos, grid) and td_q_learning_utility_ra[tmp_pos[1]][tmp_pos[0]]!=None:
                    square_error_sum += (td_q_learning_utility_ra[tmp_pos[1]][tmp_pos[0]]-value_iteration_utility_ra[tmp_pos[1]][tmp_pos[0]])**2
        rmse = math.sqrt(square_error_sum/count)
        rmse_list.append(rmse)
    return td_q_learning_utility_list, rmse_list


def value_iteration(utility_list, policy_list, grid, terminal):
    tmp_utility_ra = [[ None for j in range(0, grid.width)] for i in range(0, grid.height)]
    tmp_policy_ra = [[ None for j in range(0, grid.width)] for i in range(0, grid.height)]
    curr_ra = utility_list[-1]
    packman = packman_class() # default
    for r_idx in range(0, grid.height):
        for c_idx in range(0, grid.width):
            curr_pos = (c_idx, r_idx)
            if not can_move_to(curr_pos, grid):
                continue
            utility_max = -1000
            if terminal:
                if not can_locate_to(curr_pos, grid):
                    tmp_utility_ra[r_idx][c_idx] = grid.ra[r_idx][c_idx]
                    tmp_policy_ra[r_idx][c_idx] = (0,0)
                    continue
            tmp_utility_policy_list = []
            for forward_dir in grid.directions:
                forward_pos = add_tuples(curr_pos, forward_dir)
                candidates = []
                utility_expectation = 0
                if can_move_to(forward_pos, grid):
                    candidates.append([forward_pos, packman.forward_prob])
                else:
                    candidates.append([curr_pos, packman.forward_prob])
                left_pos = add_tuples(curr_pos, rotate_tuples(forward_dir, grid.left))
                if can_move_to(left_pos, grid):
                    candidates.append([left_pos, packman.left_prob])
                else:
                    candidates.append([curr_pos, packman.left_prob])
                right_pos = add_tuples(curr_pos, rotate_tuples(forward_dir, grid.right))
                if can_move_to(right_pos, grid):
                    candidates.append([right_pos, packman.right_prob])
                else:
                    candidates.append([curr_pos, packman.right_prob])
                for item in candidates:
                    utility_expectation += curr_ra[item[0][1]][item[0][0]]*item[1]
                if utility_expectation > utility_max:
                    utility_max = utility_expectation
                    dir_max = forward_dir
            tmp_utility_ra[r_idx][c_idx] = grid.ra[r_idx][c_idx] + GAMMA*utility_max
            tmp_policy_ra[r_idx][c_idx] = dir_max
    utility_list.append(tmp_utility_ra)
    policy_list.append(tmp_policy_ra)


def direction_index(tuple):
    return int(2*(1 - tuple[0]**2) + (1-tuple[0]-tuple[1])/2)


def alpha(t):
    val = 100.0/(99.0+t)
    return val


def td_q_initializing(value_list, utility_list, policy_list, grid):
    tmp_value_ra = [[[ None for k in range(0, len(grid.directions))] for j in range(0, grid.width)] for i in range(0, grid.height)]
    tmp_policy_ra = [[ None for j in range(0, grid.width)] for i in range(0, grid.height)]
    tmp_utility_ra = [[ None for j in range(0, grid.width)] for i in range(0, grid.height)]
    packman = packman_class() # default
    for r_idx in range(0, grid.height):
        for c_idx in range(0, grid.width):
            if can_move_to((c_idx, r_idx), grid):
                for dir_idx in range(0, len(grid.directions)):
                    tmp_value_ra[r_idx][c_idx][dir_idx] = 0
                tmp_policy_ra[r_idx][c_idx] = random.choice(grid.directions)
    value_list.append(tmp_value_ra)
    policy_list.append(tmp_policy_ra)
    utility_list.append(tmp_utility_ra)


def td_q_learning(value_list, policy_list, visited_ra, t, grid):
    tmp_value_ra = copy.deepcopy(value_list[-1])
    curr_policy_ra = policy_list[-1]
    packman = packman_class()
    packman.randomize_position(grid)
    running = True
    randomize_direction = False
    past_pos = None
    while running:
        curr_pos = packman.pos
        visited_ra[curr_pos[1]][curr_pos[0]] += 1
        if visited_ra[curr_pos[1]][curr_pos[0]] < 100 or randomize_direction:
            packman.dir = random.choice(grid.directions) #Exploration function
        else:
            packman.dir = curr_policy_ra[curr_pos[1]][curr_pos[0]]
        curr_dir = packman.dir
        if can_locate_to(curr_pos, grid):
            packman.move(grid)
            futr_pos = packman.pos
            if futr_pos == past_pos:
                randomize_direction = True
            else:
                randomize_direction = False
            max_q = max(tmp_value_ra[futr_pos[1]][futr_pos[0]])
            dir_idx = direction_index(curr_dir)
            tmp_value_ra[curr_pos[1]][curr_pos[0]][dir_idx] = tmp_value_ra[curr_pos[1]][curr_pos[0]][dir_idx] + alpha(t)*(grid.ra[curr_pos[1]][curr_pos[0]] + GAMMA*max_q - tmp_value_ra[curr_pos[1]][curr_pos[0]][dir_idx])            
        else:
            for dir_idx in range(0, len(grid.directions)):
                tmp_value_ra[curr_pos[1]][curr_pos[0]][dir_idx] = grid.ra[curr_pos[1]][curr_pos[0]]
            running = False
        past_pos = curr_pos
    value_list.append(tmp_value_ra)


def td_q_solution(value_list, utility_list, policy_list, grid):
    past_value_ra = copy.deepcopy(value_list[-1]) 
    tmp_policy_ra = [[ None for j in range(0, grid.width)] for i in range(0, grid.height)]
    tmp_utility_ra = [[ None for j in range(0, grid.width)] for i in range(0, grid.height)]
    for r_idx in range(0, grid.height):
        for c_idx in range(0, grid.width):
            tmp_pos = (c_idx, r_idx)
            if can_locate_to(tmp_pos, grid):
                test_max = -1000
                for forward_dir in grid.directions:
                    dir_idx = direction_index(forward_dir)
                    test = past_value_ra[tmp_pos[1]][tmp_pos[0]][dir_idx]
                    if test > test_max:
                        test_max = test
                        dir_max = forward_dir
                tmp_utility_ra[tmp_pos[1]][tmp_pos[0]] = test_max ##for printing
                tmp_policy_ra[tmp_pos[1]][tmp_pos[0]] = dir_max
            elif can_move_to(tmp_pos, grid):
                tmp_utility_ra[tmp_pos[1]][tmp_pos[0]] = grid.ra[tmp_pos[1]][tmp_pos[0]] ##for printing
                tmp_policy_ra[tmp_pos[1]][tmp_pos[0]] = (0,0)
    utility_list.append(tmp_utility_ra)
    policy_list.append(tmp_policy_ra)


def parse_cl_args():
    """
    Returns:
        grid_file: The name of the input grid file.
    """
    cl_args = arg_parser.parse_args()
    grid_file = CWD + '/' + os.path.relpath(cl_args.grid_file, CWD)
    return grid_file


def read_file(fil):
    """
    Args:
        fil: The name of the file to read.
    Returns:
        lines: The list of lines from the file.
    """
    lines = []
    with open(fil) as f:
        lines = f.readlines()
    return lines


def populate_array_from_lines(grid_rows, grid):
    """
    Args:
        grid_rows: The list of lines representing the rows of the grid.
        grid: The grid class
    Returns:
        grid: The grid class.
        start: The (c,r) coordinates of 'P', the start point.
        walls: The (c,r) coordinates of '%', the walls.
    """
    markers = markers_class()
    r_idx = c_idx = 0
    for row in grid_rows:
        row = row.strip()
        for c in row:
            if c == '%':
                grid.ra[r_idx][c_idx] = None
                markers.walls.append((c_idx, r_idx))
            elif c == 'p':
                grid.ra[r_idx][c_idx] = -1.0
                markers.small_penalty.append((c_idx, r_idx))
            elif c == 'P':
                grid.ra[r_idx][c_idx] = -3.0
                markers.great_penalty.append((c_idx, r_idx))
            elif c == 'r':
                grid.ra[r_idx][c_idx] = 1.0
                markers.small_reward.append((c_idx, r_idx))
            elif c == 'R':
                grid.ra[r_idx][c_idx] = 3.0
                markers.great_reward.append((c_idx, r_idx))
            else:
                grid.ra[r_idx][c_idx] = -0.04
                if c == 'S':
                    markers.start = (c_idx, r_idx)
            c_idx += 1
        # Reset all the way to the left, move down one row.
        c_idx = 0
        r_idx += 1
    return grid, markers 


def create_maze_array(grid_rows):
    """
    Args:
        grid_rows: The list of lines representing the rows of the grid.
    Returns:
        grid: The grid class
        start: The (r,c) coordinates of 'P', the start point.
    """
    if grid_rows == []:
        return []
    grid = grid_class(len(grid_rows[0])-1, len(grid_rows))
    grid, markers = populate_array_from_lines(grid_rows, grid)
    return grid, markers


def is_in_bounds(grid, coord):
    """
    Args:
        maze_ra: The array populated with the maze characters.
        coord: The (r,c) tuple coordinate to check.

    Returns:
        is_within: True if the coordinates are within the maze.
    """
    if (coord[0] < 0 or coord[1] < 0):
        return False
    if (coord[0] >= grid.width or coord[1] >= grid.height):
        return False
    return True


def can_move_to(coord, grid):
    """
    Args:
        maze_ra: The array populated with the maze characters.
        coord: The (r,c) tuple coordinate to check.

    Returns:
        can_move: False if '%'. True if ' ', 'P', or '.'.
    """
    if not is_in_bounds(grid, coord):
        return False
    if grid.ra[coord[1]][coord[0]] == None:
        return False
    return True


def can_locate_to(coord, grid):
    """
    Args:
        maze_ra: The array populated with the maze characters.
        coord: The (r,c) tuple coordinate to check.

    Returns:
        can_move: False if '%'. True if ' ', 'P', or '.'.
    """
    if not is_in_bounds(grid, coord):
        return False
    if grid.ra[coord[1]][coord[0]] != -0.04:
        return False
    return True


def add_tuples(one, two):
    """
    Args:
        one: The first tuple, containing two elements.
        two: The second tuple, containing two elements.
    Returns:
        summmed: The tuple where corresponding elements were summed.
    """
    first = one[0] + two[0]
    second = one[1] + two[1]
    summed = (first, second)
    return summed


def rotate_tuples(one, two):
    """
    Args:
        one: The first tuple, containing two elements.
        two: The second tuple, containing two elements.
    Returns:
        summmed: The tuple the cross production of the two tuples.
    """

    first = one[0] * two[0] - one[1] * two[1]
    second = one[1] * two[0] + one[0] * two[1]
    rotated = (first, second)
    return rotated


def substract_tuples(one, two):
    """
    Args:
        one: The first tuple, containing two elements.
        two: The second tuple, containing two elements.
    Returns:
        subtracted: The tuple where corresponding elements were summed.
    """
    first = one[0] - two[0]
    second = one[1] - two[1]
    subtracted = (first, second)
    return subtracted


def draw_board(screen, markers):
    draw_walls(screen, markers.walls)
    draw_markers(screen, markers.small_penalty, RGB_ORANGE)
    draw_markers(screen, markers.great_penalty, RGB_RED)
    draw_markers(screen, markers.small_reward, RGB_SKYBLUE)
    draw_markers(screen, markers.great_reward, RGB_BLUE)


def draw_walls(screen, walls):
    for coord in walls:
        position =  (SQUARE_SIZE * coord[0], SQUARE_SIZE * coord[1])
        screen.blit(Wall_TEXTURE, position)


def draw_markers(screen, drawing_markers, color):
    for coord in drawing_markers:
        center = (SQUARE_SIZE * coord[0] + SQUARE_SIZE/2, SQUARE_SIZE * coord[1] + SQUARE_SIZE/2)
        pygame.draw.circle(screen, color, center, 2*PATH_RADIUS)


def draw_policy(screen, policy_ra):
    for coord_x in range(0, len(policy_ra[0])):
        for coord_y in range(0, len(policy_ra)):
            if policy_ra[coord_y][coord_x] == None:
                pass
            elif policy_ra[coord_y][coord_x] == (0,0):
                center = (SQUARE_SIZE * coord_x + SQUARE_SIZE/2, SQUARE_SIZE * coord_y + SQUARE_SIZE/2)
                pygame.draw.circle(screen, RGB_GRAY, center, PATH_RADIUS)
            else:
                triangle_top = (SQUARE_SIZE * coord_x + SQUARE_SIZE/4 * (2 + policy_ra[coord_y][coord_x][0]), SQUARE_SIZE * coord_y + SQUARE_SIZE/4 * (2 + policy_ra[coord_y][coord_x][1]))
                triangle_left = (SQUARE_SIZE * coord_x + SQUARE_SIZE/4 * (2 - policy_ra[coord_y][coord_x][0] - policy_ra[coord_y][coord_x][1]), SQUARE_SIZE * coord_y + SQUARE_SIZE/4 * (2 - policy_ra[coord_y][coord_x][1] + policy_ra[coord_y][coord_x][0]))
                triangle_right = (SQUARE_SIZE * coord_x + SQUARE_SIZE/4 * (2 - policy_ra[coord_y][coord_x][0] + policy_ra[coord_y][coord_x][1]), SQUARE_SIZE * coord_y + SQUARE_SIZE/4 * (2 - policy_ra[coord_y][coord_x][1] - policy_ra[coord_y][coord_x][0]))
                pygame.draw.polygon(screen, RGB_GRAY, [triangle_top, triangle_left, triangle_right], 1)


def draw_pacman(screen, to_go, foot_prints):
    x = to_go.pop(0)
    foot_prints.append(x)
    position = (SQUARE_SIZE * x[0], SQUARE_SIZE * x[1])
    screen.blit(PACMAN, position)


def draw_footprints(screen, foot_prints):
    for x in foot_prints:
        center = (SQUARE_SIZE * x[0] + SQUARE_SIZE/2, SQUARE_SIZE * x[1] + SQUARE_SIZE/2)
        pygame.draw.circle(screen, (255, 0, 0), center, PATH_RADIUS)


def write_number(screen, paths, left_dots, visited_dots, FONT):
    if len(paths) != 0:
        current_coord = paths[-1]
        for dot in left_dots:
            if dot == current_coord:
                left_dots.remove(dot)
                visited_dots.append(current_coord)
        for visiting_number in range(0, len(visited_dots)):
            # text = FONT.font.render("H", True, (0, 128, 0))
            x = visited_dots[visiting_number]
            position = (SQUARE_SIZE * x[0], SQUARE_SIZE * x[1])
            screen.blit(VISITED_DOT, position)
            text = FONT.render(str(visiting_number), True, (0, 128, 0))
            screen.blit(text, position)


def write_status(screen, iteration_idx, size): # draw the board.
    sentence_pos = ((SQUARE_SIZE+1)*0 + SQUARE_SIZE/4, SQUARE_SIZE*size[1] + SQUARE_SIZE/8)    
    sentence_contents = "Iteration: " + str(iteration_idx)
    sentence_text = SMALL_FONT.render(sentence_contents, True, RGB_WHITE)
    screen.blit(sentence_text, sentence_pos)


def animation_main(size, policy_list, markers, path):
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.init()
    screen_size = (SQUARE_SIZE*size[0], SQUARE_SIZE*size[1]+SQUARE_SIZE)
    pygame.display.set_caption('MP4 1.1')
    screen = pygame.display.set_mode(screen_size)   
    clock = pygame.time.Clock()
    running = True
    iteration_idx = 0
    to_go = copy.deepcopy(path)
    foot_prints = []

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        policy_ra = policy_list[iteration_idx]
        screen.fill((0, 0, 0))
        draw_board(screen, markers)
        draw_policy(screen, policy_ra)
        write_status(screen, iteration_idx, size)
        pygame.display.flip()
        if iteration_idx < len(policy_list)-1:
            pygame.time.wait(50) #
            iteration_idx +=1
        else:
            running = False
            pygame.time.wait(300) #
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        policy_ra = policy_list[iteration_idx-1]
        screen.fill((0, 0, 0))
        draw_board(screen, markers)
        draw_policy(screen, policy_ra)
        write_status(screen, iteration_idx, size)
        draw_footprints(screen, foot_prints)
        draw_pacman(screen, to_go, foot_prints)
        pygame.display.flip()
        if len(to_go)!=0:
            pygame.time.wait(100) #
        else:
            running = False
            pygame.time.wait(300) #

def main():
    grid_file = parse_cl_args()
    grid_rows = read_file(grid_file)
    grid, markers = create_maze_array(grid_rows)
    
    print "1.1-1: Infinite State Value Iteration"
    infinite_value_iteration_utility_list, infinite_value_iteration_policy_list = iterate(grid, ITERATION_DEPTH, False, False)

    packman_infinite_test = packman_class()
    packman_infinite_test.pos = markers.start
    path_infinite = []
    follow_infinite = infinite_value_iteration_policy_list[-1]

    for i in range(0, ITERATION_DEPTH):
        packman_infinite_test.dir = follow_infinite[packman_infinite_test.pos[1]][packman_infinite_test.pos[0]]
        packman_infinite_test.move(grid)
        path_infinite.append(packman_infinite_test.pos)

    animation_main((grid.width, grid.height), infinite_value_iteration_policy_list, markers, path_infinite)

    for value_row in infinite_value_iteration_utility_list[-1]:
        print value_row

    for r_idx in range(0, grid.height):
        for c_idx in range(0, grid.width):
            curr_pos = (c_idx, r_idx)
            tmp_plot_list = []
            if can_move_to(curr_pos, grid):
                for infinite_value_iteration_utility_ra in infinite_value_iteration_utility_list:
                    tmp_plot_list.append(infinite_value_iteration_utility_ra[r_idx][c_idx])
                plt.plot(tmp_plot_list, label=str(curr_pos))
                # plt.title(str(curr_pos))
    
    plt.ylabel('Utility')
    plt.xlabel('Iteration')
    plt.title('Utility vs. Iteration')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.show()


    print "" 
    print "========================================================"
    print "1.1-2: Terminal State Value Iteration for 1.1"
    terminal_value_iteration_utility_list, terminal_value_iteration_policy_list = iterate(grid, ITERATION_DEPTH, True, False)

    packman_terminal_test = packman_class()
    packman_terminal_test.pos = markers.start
    path_terminal = []
    follow_terminal = terminal_value_iteration_policy_list[-1]

    while can_locate_to(packman_terminal_test.pos, grid):
        packman_terminal_test.dir = follow_terminal[packman_terminal_test.pos[1]][packman_terminal_test.pos[0]]
        packman_terminal_test.move(grid)
        path_terminal.append(packman_terminal_test.pos)

    animation_main((grid.width, grid.height), terminal_value_iteration_policy_list, markers, path_terminal)

    for value_row in terminal_value_iteration_utility_list[-1]:
        print value_row

    for r_idx in range(0, grid.height):
        for c_idx in range(0, grid.width):
            curr_pos = (c_idx, r_idx)
            tmp_plot_list = []
            if can_move_to(curr_pos, grid):
                for terminal_value_iteration_utility_ra in terminal_value_iteration_utility_list:
                    tmp_plot_list.append(terminal_value_iteration_utility_ra[r_idx][c_idx])
                plt.plot(tmp_plot_list, label=str(curr_pos))
                # plt.title(str(curr_pos))
    
    plt.ylabel('Utility')
    plt.xlabel('Iteration')
    plt.title('Utility vs. Iteration')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.show()

    print "" 
    print "========================================================"
    print "1.2: Compare the result of TD Q-Learning with the one of Value Iteration"

    td_q_learning_utility_list, rmse_list = compare(terminal_value_iteration_utility_list[-1], grid)
    print "Final RMS Error: " + str(rmse_list[-1])
    print ""
    for value_row in td_q_learning_utility_list[-1]:
        print value_row
    print "" 

    plt.plot(rmse_list)
    plt.title('1.2 result')
    plt.ylabel('RMS Error')
    plt.xlabel('Iteration')
    plt.show()

if __name__ == '__main__':
    main()