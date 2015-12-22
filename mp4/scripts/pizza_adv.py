#!/usr/bin/python

import argparse
import heapq
import os
import sys
from collections import deque
import pygame
import random
import copy
import time

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
WALL_TEXTURE = pygame.image.load('../data/graphics/ChippedBricks.png')
PACKMAN = pygame.image.load('../data/graphics/ming.png')
WITH_INGREDIENTS = pygame.image.load('../data/graphics/with_ingredients.png')
WITH_PIZZA = pygame.image.load('../data/graphics/pizza.png')
STUDENT = pygame.image.load('../data/graphics/Home.png')
PIZZA_RESTAURANT = pygame.image.load('../data/graphics/pizza_restaurant.png')
GROCERY_STORE = pygame.image.load('../data/graphics/grocery_store.png')

DEFAULT = True
LEARNING = True

if DEFAULT:
    IMG_WIDTH = 15
    IMG_HEIGHT = 9
    MAP = '../data/griddata/pizza'

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
        self.start = None        
        self.walls = []
        self.pizza = []
        self.grocery = []
        self.student = []


# Agent, Transition Model
class deliveryman_class:
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
        self.state = 0 # 0: Nothing, 1: With ingredients, 2: With pizza 3: With both ingredients and pizza
        self.wait = False
        self.last_visited_pizza_chain = (0,0)
        self.set_probabilty(0.9, 0.05, 0.05, 0)

    def set_probabilty(self, forward_val, left_val, right_val, stay_val):
        self.forward_prob = forward_val
        self.left_prob = left_val
        self.right_prob = right_val
        self.stay_prob = stay_val

    def set_state(self, pos_coord, dir_coord):
        self.pos = pos_coord
        self.dir = dir_coord

    def randomize_position(self, grid):
        while True:
            tmp_position = (random.randint(0, grid.width-1), random.randint(0, grid.height -1))
            if can_move_to(tmp_position, grid):
                self.pos = tmp_position
                return True


    def randomize_direction(self, grid):
        self.dir = random.choice(grid.directions)

    def randomize_state(self, grid):
        self.state = random.randint(0,3)
        self.last_visited_pizza_chain = random.choice(grid.pizza)

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
        candidates.append([curr_pos, self.stay_prob])
        self.pos = weighted_random_choice(candidates)

    def reward(self, grid):
        if grid.ra[self.pos[1]][self.pos[0]] == 3:
            if self.state == 2 or self.state ==3:
                return 5 + 0.1*manhattan_dist(self.last_visited_pizza_chain, self.pos)
        return -0.1


def dropy_by(deliveryman, new_state):
    if deliveryman.wait:
        deliveryman.state = new_state
        deliveryman.wait = False
    else:
        deliveryman.wait = True

def process_step(deliveryman, grid):

    if grid.ra[deliveryman.pos[1]][deliveryman.pos[0]] == 1: # Drop by a pizza place
        if deliveryman.state == 1: # If the deliveryman has ingredients
            dropy_by(deliveryman, 2)
    elif grid.ra[deliveryman.pos[1]][deliveryman.pos[0]] ==2: # Drop by a grocery store
            if deliveryman.state == 0:
                dropy_by(deliveryman, 1)
            elif deliveryman.state == 2:
                dropy_by(deliveryman, 3)
    elif grid.ra[deliveryman.pos[1]][deliveryman.pos[0]] == 3: # Drop by a student's home
            if deliveryman.state == 2:
                dropy_by(deliveryman, 0)
            elif deliveryman.state == 3:
                dropy_by(deliveryman, 1)

    if deliveryman.wait:
        deliveryman.set_probabilty(0.0, 0.0, 0.0, 1.0) # Wait
    elif deliveryman.state == 2 or deliveryman.state == 3: # Have pizza
        if grid.ra[deliveryman.pos[1]][deliveryman.pos[0]] == 1:
            deliveryman.set_probabilty(0.6, 0.05, 0.05, 0.3) # Reheat
            deliveryman.last_visited_pizza_chain = copy.copy(deliveryman.pos)
        else:
            deliveryman.set_probabilty(0.8, 0.05, 0.05, 0.1)
    else: # Not have pizza
        deliveryman.set_probabilty(0.9, 0.05, 0.05, 0.0)

    deliveryman.move(grid)


def manhattan_dist(curr, end):
    """Computes the Manhattan distance between current and end nodes.

    Manhattan distance is defined as the distance between two points
    if only movement along the horizontal and vertical axes are
    allowed.

    Args:
        curr: The (r,c) tuple coordinates of the current node.
        end: The (r,c) tuple coordinates of the end node.

    Returns:
        int: The Manhattan distance between the current and end nodes.
    """
    return abs(end[0] - curr[0]) + abs(end[1] - curr[1])


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
    r_idx = c_idx = 0
    for row in grid_rows:
        row = row.strip()
        for c in row:
            if c == '%':
                grid.ra[r_idx][c_idx] = None
                grid.walls.append((c_idx, r_idx))
            elif c == 'P':
                grid.ra[r_idx][c_idx] = 1
                grid.pizza.append((c_idx, r_idx))
            elif c == 'G':
                grid.ra[r_idx][c_idx] = 2
                grid.grocery.append((c_idx, r_idx))
            elif c == 'S':
                grid.ra[r_idx][c_idx] = 3
                grid.student.append((c_idx, r_idx))
            else:
                grid.ra[r_idx][c_idx] = 0
            c_idx += 1
        c_idx = 0
        r_idx += 1
    return grid 


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
    grid = populate_array_from_lines(grid_rows, grid)
    return grid


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


def weighted_random_choice(label_list):
    """
    Args:
        label_list: The list of pairs. Each pair consists of a label and probabilty.
    Returns:
        choice: The label of the randomized output.
    """
    total_prob = 0.0
    cumm_prob = 0.0
    for item in label_list:
        total_prob += item[1]
    x = random.uniform(0, total_prob)
    for item in label_list:
        cumm_prob += item[1]
        if x < cumm_prob: break
    return copy.deepcopy(item[0])


def iterate(grid, depth):
    policy_list = []
    utility_list=[]
    if LEARNING:
        value_list = []
        visited_ra = [[[ 0 for k in range(0,4)] for j in range(0, grid.width)] for i in range(0, grid.height)]
        td_q_initializing(value_list, utility_list, policy_list, grid)
        t = 1
        for i in range(0, depth+1):
            td_q_learning(value_list, policy_list, visited_ra, t, grid)
            td_q_solution(value_list, utility_list, policy_list, grid)
            t +=1
    else:
        utility_list.append(copy.deepcopy(grid.ra))
        policy_list.append([[ None for j in range(0, grid.width)] for i in range(0,grid.height)])
        for i in range(0, depth):
            value_iteration(utility_list, policy_list, grid)
    return utility_list, policy_list





def direction_index(tuple):
    return int(2*(1 - tuple[0]**2) + (1-tuple[0]-tuple[1])/2)


def alpha(t):
    val = 200.0/(199.0+t)
    return val




def td_q_initializing(value_list, utility_list, policy_list, grid):
    tmp_value_ra = [[[[ None for l in range(0, len(grid.directions))] for k in range(0, 4)] for j in range(0, grid.width)] for i in range(0, grid.height)]
    tmp_policy_ra = [[[ None for k in range(0, 4)] for j in range(0, grid.width)] for i in range(0, grid.height)]
    tmp_utility_ra = [[[ None for k in range(0, 4)] for j in range(0, grid.width)] for i in range(0, grid.height)]
    for r_idx in range(0, grid.height):
        for c_idx in range(0, grid.width):
            if can_move_to((c_idx, r_idx), grid):
                for state_idx in range(0,4):
                    for dir_idx in range(0, len(grid.directions)):
                        tmp_value_ra[r_idx][c_idx][state_idx][dir_idx] = 0
                    tmp_policy_ra[r_idx][c_idx][state_idx] = random.choice(grid.directions)
    value_list.append(tmp_value_ra)
    policy_list.append(tmp_policy_ra)
    utility_list.append(tmp_utility_ra)


def td_q_learning(value_list, policy_list, visited_ra, t, grid):
    tmp_value_ra = copy.deepcopy(value_list[-1]) 
    curr_policy_ra = policy_list[-1]
    deliveryman = deliveryman_class()
    deliveryman.randomize_position(grid)
    deliveryman.randomize_state(grid)
    randomize_direction = False
    past_pos = None
    past_state = None
    for iteration_idx in range(0, 100):
        curr_pos = deliveryman.pos
        curr_state = deliveryman.state
        if visited_ra[curr_pos[1]][curr_pos[0]][curr_state] < 500 or randomize_direction:
            deliveryman.dir = random.choice(grid.directions) #Exploration function
        else:
            deliveryman.dir = curr_policy_ra[curr_pos[1]][curr_pos[0]][curr_state]
        curr_dir = deliveryman.dir
        reward = deliveryman.reward(grid)       
        process_step(deliveryman, grid)
        futr_pos = deliveryman.pos
        futr_state = deliveryman.state
        if futr_pos == past_pos and futr_state == past_state:
            randomize_direction = True
        else:
            randomize_direction = False
        max_q = max(tmp_value_ra[futr_pos[1]][futr_pos[0]][futr_state])
        dir_idx = int(2*(1 - curr_dir[0]**2) + (1-curr_dir[0]-curr_dir[1])/2)
        tmp_value_ra[curr_pos[1]][curr_pos[0]][curr_state][dir_idx] = tmp_value_ra[curr_pos[1]][curr_pos[0]][curr_state][dir_idx] + alpha(t)*(reward + GAMMA*max_q - tmp_value_ra[curr_pos[1]][curr_pos[0]][curr_state][dir_idx])
        past_pos = curr_pos
        past_state = curr_state
    value_list.append(tmp_value_ra)



def td_q_solution(value_list, utility_list, policy_list, grid):
    past_value_ra = copy.deepcopy(value_list[-1]) 
    tmp_policy_ra = [[[ None for k in range(0, 4)] for j in range(0, grid.width)] for i in range(0, grid.height)]
    tmp_utility_ra = [[[ None for k in range(0, 4)] for j in range(0, grid.width)] for i in range(0, grid.height)]
    for r_idx in range(0, grid.height):
        for c_idx in range(0, grid.width):
            tmp_pos = (c_idx, r_idx)
            if can_move_to(tmp_pos, grid):
                for tmp_state in range(0,4):
                    test_max = -1000000
                    for forward_dir in grid.directions:
                        dir_idx = direction_index(forward_dir)
                        test = past_value_ra[tmp_pos[1]][tmp_pos[0]][tmp_state][dir_idx]
                        if test > test_max:
                            test_max = test
                            dir_max = forward_dir
                    tmp_utility_ra[tmp_pos[1]][tmp_pos[0]][tmp_state] = test_max ##for printing
                    tmp_policy_ra[tmp_pos[1]][tmp_pos[0]][tmp_state] = dir_max
    utility_list.append(tmp_utility_ra)
    policy_list.append(tmp_policy_ra)



def draw_board(screen, grid):
    draw_walls(screen, grid.walls)
    draw_grocery_stores(screen, grid.grocery)
    draw_pizza_resaurants(screen, grid.pizza)
    draw_students(screen, grid.student)

def draw_walls(screen, walls):
    for coord in walls:
        position =  (SQUARE_SIZE * coord[0], SQUARE_SIZE * coord[1])
        screen.blit(WALL_TEXTURE, position)


def draw_grocery_stores(screen, grocery):
    for coord in grocery:
        position =  (SQUARE_SIZE * coord[0], SQUARE_SIZE * coord[1])
        screen.blit(GROCERY_STORE, position)


def draw_pizza_resaurants(screen, pizza):
    for coord in pizza:
        position =  (SQUARE_SIZE * coord[0], SQUARE_SIZE * coord[1])
        screen.blit(PIZZA_RESTAURANT, position)

def draw_students(screen, student):
    for coord in student:
        position =  (SQUARE_SIZE * coord[0], SQUARE_SIZE * coord[1])
        screen.blit(STUDENT, position)


def draw_markers(screen, drawing_markers, color):
    for coord in drawing_markers:
        center = (SQUARE_SIZE * coord[0] + SQUARE_SIZE/2, SQUARE_SIZE * coord[1] + SQUARE_SIZE/2)
        pygame.draw.circle(screen, color, center, 2*PATH_RADIUS)



def draw_policy(screen, policy_ra, state):
    for coord_x in range(0, len(policy_ra[0])):
        for coord_y in range(0, len(policy_ra)):
            if policy_ra[coord_y][coord_x][state] == None:
                pass
            elif policy_ra[coord_y][coord_x][state] == (0,0):
                center = (SQUARE_SIZE * coord_x + SQUARE_SIZE/2, SQUARE_SIZE * coord_y + SQUARE_SIZE/2)
                pygame.draw.circle(screen, RGB_GRAY, center, PATH_RADIUS)
            else:
                triangle_top = (SQUARE_SIZE * coord_x + SQUARE_SIZE/4 * (2 + policy_ra[coord_y][coord_x][state][0]), SQUARE_SIZE * coord_y + SQUARE_SIZE/4 * (2 + policy_ra[coord_y][coord_x][state][1]))
                triangle_left = (SQUARE_SIZE * coord_x + SQUARE_SIZE/4 * (2 - policy_ra[coord_y][coord_x][state][0] - policy_ra[coord_y][coord_x][state][1]), SQUARE_SIZE * coord_y + SQUARE_SIZE/4 * (2 - policy_ra[coord_y][coord_x][state][1] + policy_ra[coord_y][coord_x][state][0]))
                triangle_right = (SQUARE_SIZE * coord_x + SQUARE_SIZE/4 * (2 - policy_ra[coord_y][coord_x][state][0] + policy_ra[coord_y][coord_x][state][1]), SQUARE_SIZE * coord_y + SQUARE_SIZE/4 * (2 - policy_ra[coord_y][coord_x][state][1] - policy_ra[coord_y][coord_x][state][0]))
                pygame.draw.polygon(screen, RGB_GRAY, [triangle_top, triangle_left, triangle_right], 1)
                

def draw_pacman(screen, to_go, foot_prints):
    curr = to_go.pop(0)
    foot_prints.append(curr.pos)
    if curr.state == 0:
        img = PACKMAN
    elif curr.state ==1:
        img = WITH_INGREDIENTS
    else:
        img = WITH_PIZZA
    position = (SQUARE_SIZE * curr.pos[0], SQUARE_SIZE * curr.pos[1])
    screen.blit(img, position)

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


def animation_main(size, policy_list, state, grid):
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.init()
    screen_size = (SQUARE_SIZE*size[0], SQUARE_SIZE*size[1]+SQUARE_SIZE)
    pygame.display.set_caption('MP4 1.3')
    screen = pygame.display.set_mode(screen_size)   
    clock = pygame.time.Clock()
    running = True
    iteration_idx = 0

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        policy_ra = policy_list[iteration_idx]
        screen.fill((0, 0, 0))
        draw_board(screen, grid)
        draw_policy(screen, policy_ra, state)
        write_status(screen, iteration_idx, size)
        pygame.display.flip()
        if iteration_idx < len(policy_list)-1:
            pygame.time.wait(10)
            iteration_idx +=1
        else:
            running = False
    


def simulation_main(size, policy_list, grid, path):
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.init()
    screen_size = (SQUARE_SIZE*size[0], SQUARE_SIZE*size[1]+SQUARE_SIZE)
    pygame.display.set_caption('MP4 1.3')
    screen = pygame.display.set_mode(screen_size)   
    clock = pygame.time.Clock()
    running = True
    to_go = copy.deepcopy(path)
    foot_prints = []

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        policy_ra = policy_list[-1]
        screen.fill((0, 0, 0))
        draw_board(screen, grid)
        draw_policy(screen, policy_ra, to_go[0].state)
        write_status(screen, 10001, size)
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
    grid = create_maze_array(grid_rows)
    utility_list, policy_list = iterate(grid, 100)
#    value_list, policy_list = iterate(grid, 50)

    for value_ra in policy_list:
        for value_row in value_ra:
            print value_row
        print "" 


    animation_main((grid.width, grid.height), policy_list, 0, grid)
    #time.sleep(30)
    animation_main((grid.width, grid.height), policy_list, 1, grid)
    #time.sleep(30)
    animation_main((grid.width, grid.height), policy_list, 2, grid)
    #time.sleep(30)
    animation_main((grid.width, grid.height), policy_list, 3, grid)
    #time.sleep(30)


    deliveryman_test = deliveryman_class()
    deliveryman_test.pos = random.choice(grid.pizza)
    path = []
    follow = policy_list[-1]

    for i in range(0, 100):
        deliveryman_test.dir = follow[deliveryman_test.pos[1]][deliveryman_test.pos[0]][deliveryman_test.state]
        process_step(deliveryman_test, grid)
        path.append(deliveryman_test)
        print deliveryman_test.pos

    simulation_main((grid.width, grid.height), policy_list, grid, path)

if __name__ == '__main__':
    main()