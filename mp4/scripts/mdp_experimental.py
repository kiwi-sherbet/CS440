#!/usr/bin/python

import argparse
import heapq
import os
import sys
from collections import deque
import pygame
import random
import copy

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
TERMINAL = True
LEARNING = False

if DEFAULT:
    IMG_WIDTH = 8
    IMG_HEIGHT = 8
    MAP = '../data/griddata/default'

GAMMA = 0.7

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

# Agent, Transition Model
class packman_class:
    """
    State:
        pos: 
        dir:
        score:
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
        self.score = 0
        self.directions = [(1,0), (-1,0), (0,1), (0,-1)]
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
            tmp_position = (random.randint(0, grid.width), random.randint(0, grid.height))
            if can_locate_to(tmp_position, grid):
                self.pos = tmp_position
                return True

    def randomize_direction(self, grid):
        self.dir = random.choice(self.directions)

    def move(self, grid):
        candidates=[]
        curr_pos = copy.deepcopy(self.pos)
        forward_dir = copy.deepcopy(self.dir)
        forward_pos = add_tuples(curr_pos, forward_dir)
        if can_move_to(forward_pos, grid):
            candidates.append([forward_pos, self.forward_prob])
        else:
            candidates.append([curr_pos, self.forward_prob])
        left_dir = (-forward_dir[1], forward_dir[0])
        left_pos = add_tuples(curr_pos, left_dir)
        if can_move_to(left_pos, grid):
            candidates.append([left_pos, self.left_prob])
        else:
            candidates.append([curr_pos, self.left_prob])
        right_dir = (forward_dir[1], -forward_dir[0])
        right_pos = add_tuples(curr_pos, right_dir)
        if can_move_to(right_pos, grid):
            candidates.append([right_pos, self.right_prob])
        else:
            candidates.append([curr_pos, self.right_prob])
        self.pos = weighted_random_choice(candidates)

    def get_reward(self, grid):
        self.score += grid.ra[self.pos[1]][self.pos[0]]


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


def iterate(grid, depth):
    policy_list = []
    utility_list=[]
    if LEARNING and TERMINAL:
        value_list = []
        td_q_initializing(value_list, policy_list, utility_list,grid)
        t = 0
        for i in range(0, depth+1):
            td_q_learning(value_list, policy_list, utility_list,t, grid)
            t +=1
            print t
    else:
        utility_list.append([[ 0 for j in range(0, grid.width)] for i in range(0, grid.height)])
        policy_list.append([[ None for j in range(0, grid.width)] for i in range(0,grid.height)])
        for i in range(0, depth):
            value_iteration(utility_list, policy_list, grid)
    return utility_list, policy_list

def value_iteration(value_list, policy_list, grid):
    tmp_ra = [[ 0 for j in range(0, grid.width)] for i in range(0, grid.height)]
    tmp_policy_ra = [[ None for j in range(0, grid.width)] for i in range(0, grid.height)]
    curr_ra = value_list[-1]
    packman = packman_class() # default
    for r_idx in range(0, grid.height):
        for c_idx in range(0, grid.width):
            curr_pos = (c_idx, r_idx)
            if can_move_to(curr_pos, grid)==False:
                continue
            utility_max = -1000
            if TERMINAL:
                if can_locate_to(curr_pos, grid) == False:
                    tmp_ra[r_idx][c_idx] = grid.ra[r_idx][c_idx]
                    tmp_policy_ra[r_idx][c_idx] = (0,0)
                    continue
            tmp_utility_policy_list = []
            for forward_dir in packman.directions:
                forward_pos = add_tuples(curr_pos, forward_dir)
                candidates = []
                expectation = 0
                if can_move_to(forward_pos, grid):
                    candidates.append([forward_pos, packman.forward_prob])
                else:
                    candidates.append([curr_pos, packman.forward_prob])
                left_dir = (-forward_pos[1], forward_pos[0])
                left_pos = add_tuples(curr_pos, left_dir)
                if can_move_to(left_pos, grid):
                    candidates.append([left_pos, packman.left_prob])
                else:
                    candidates.append([curr_pos, packman.left_prob])
                right_dir = (forward_dir[1], -forward_dir[0])
                right_pos = add_tuples(curr_pos, right_dir)
                if can_move_to(right_pos, grid):
                    candidates.append([right_pos, packman.right_prob])
                else:
                    candidates.append([curr_pos, packman.right_prob])
                for item in candidates:
                    expectation += curr_ra[item[0][1]][item[0][0]]*item[1]
                tmp_utility_policy_list.append([expectation, forward_dir])
            for tmp_utility_policy in tmp_utility_policy_list:
                if tmp_utility_policy[0] > utility_max:
                    utility_max = tmp_utility_policy[0]
                    dir_max = tmp_utility_policy[1]
            tmp_ra[r_idx][c_idx] = grid.ra[r_idx][c_idx] + GAMMA*utility_max
            tmp_policy_ra[r_idx][c_idx] = dir_max
    value_list.append(tmp_ra)
    policy_list.append(tmp_policy_ra)


def alpha(t):
    val = 60.0/(59.0+t)
    return val

def td_q_initializing(value_list, policy_list, utility_list, grid):
    tmp_ra = [[ None for j in range(0, grid.width)] for i in range(0, grid.height)]
    tmp_policy_ra = [[ None for j in range(0, grid.width)] for i in range(0, grid.height)]
    tmp_utility_ra = copy.deepcopy(grid.ra)
    packman = packman_class() # default
    for r_idx in range(0, grid.height):
        for c_idx in range(0, grid.width):
            if can_move_to((c_idx, r_idx), grid):
                tmp_ra[r_idx][c_idx]=[ None for i in range(0, len(packman.directions))]
                for dir_idx in range(0, len(packman.directions)):
                    tmp_ra[r_idx][c_idx][dir_idx] = grid.ra[r_idx][c_idx]
                tmp_policy_ra[r_idx][c_idx] = random.choice(packman.directions)
    value_list.append(tmp_ra)
    policy_list.append(tmp_policy_ra)
    utility_list.append(tmp_utility_ra)

def td_q_learning(value_list, policy_list, utility_list, t, grid):
    curr_ra = value_list[-1]
    tmp_ra = copy.deepcopy(curr_ra)
    curr_policy_ra = policy_list[-1]
    tmp_policy_ra = copy.deepcopy(curr_policy_ra)
    tmp_utility_ra = copy.deepcopy(utility_list[-1])
    packman = packman_class()
    packman.randomize_position(grid)
    packman.dir =tmp_policy_ra[packman.pos[1]][packman.pos[0]]
    running = True
    while running:
        curr_pos = packman.pos
        packman.dir = curr_policy_ra[curr_pos[1]][curr_pos[0]]
        curr_dir = packman.dir
        packman.move(grid)
        futr_pos = packman.pos
        max_q = max(curr_ra[futr_pos[1]][futr_pos[0]])
        dir_idx = int(2*(1 - curr_dir[0]**2) + (1-curr_dir[0]-curr_dir[1])/2)
        tmp_ra[curr_pos[1]][curr_pos[0]][dir_idx] = curr_ra[curr_pos[1]][curr_pos[0]][dir_idx] + alpha(t)*(grid.ra[curr_pos[1]][curr_pos[0]] + GAMMA*max_q - curr_ra[curr_pos[1]][curr_pos[0]][dir_idx])

        if can_locate_to(futr_pos, grid)==False:
            tmp_policy_ra[futr_pos[1]][futr_pos[0]] = (0,0)
            running = False
        else:
            test_max = -1000
            for forward_dir in packman.directions:
                dir_idx = int(2*(1 - forward_dir[0]**2) + (1-forward_dir[0]-forward_dir[1])/2)
                test = tmp_ra[curr_pos[1]][curr_pos[0]][dir_idx]
                if test > test_max:
                    test_max = test
                    dir_max = forward_dir
            tmp_utility_ra[curr_pos[1]][curr_pos[0]] = round(test_max,4) ##for printing
            tmp_policy_ra[curr_pos[1]][curr_pos[0]] = dir_max
    value_list.append(tmp_ra)
    policy_list.append(tmp_policy_ra)
    utility_list.append(tmp_utility_ra)

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
            pygame.time.wait(30)
            iteration_idx +=1
        else:
            running = False
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
            pygame.time.wait(1000)
        else:
            running = False

def main():
    grid_file = parse_cl_args()
    grid_rows = read_file(grid_file)
    grid, markers = create_maze_array(grid_rows)
    utility_list, policy_list = iterate(grid, 50)
   # value_list, policy_list = iterate(grid, 50)

    packman_test = packman_class()
    packman_test.pos = markers.start
    path = []
    follow = policy_list[-1]

    for value_ra in utility_list:
        for value_row in value_ra:
            print value_row
        print "" 

    while can_locate_to(packman_test.pos, grid):
        packman_test.dir = follow[packman_test.pos[1]][packman_test.pos[0]]
        packman_test.move(grid)
        path.append(packman_test.pos)

    animation_main((grid.width, grid.height), policy_list, markers, path)


if __name__ == '__main__':
    main()
