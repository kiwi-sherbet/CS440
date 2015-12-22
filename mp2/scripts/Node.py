#!/usr/bin/python

import math
import random
import sys
import pygame
import copy
import time
from collections import deque
from pygame import gfxdraw

SCALE = 50
GRID_SIZE = 10
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
COLORS = [RED, GREEN, BLUE, YELLOW]
NUM_COLORS = 4
TOTAL_ASSIGN = 0

class Node:
    num_nodes = 0
    quadrant_size = GRID_SIZE
    node_list = []

    def __init__(self):
        self.name = str(Node.num_nodes)
        x = random.randint(0,Node.quadrant_size)
        y = random.randint(0,Node.quadrant_size)
        while duplicate_coor(self, x, y):
            x = random.randint(0,Node.quadrant_size)
            y = random.randint(0,Node.quadrant_size)
        self.coor = (x,y)
        self.adj_list = []
        self.colors_left = [RED, GREEN, BLUE, YELLOW]

        Node.node_list.append(self)
        Node.num_nodes += 1


def duplicate_coor(self, x, y):
    is_duplicate = False
    for node in Node.node_list:
        if node.coor[0] == x and node.coor[1] == y:
            is_duplicate = True
    return is_duplicate

def printGraph():
    for node in Node.node_list:
        print('Node: ' + node.name)
        print(str(node.coor))
        print('Adjacency list: ' + str(node.adj_list))


def calculate_cosine_of_points(a,o,b):
    a = a.coor
    o = o.coor
    b = b.coor
    a_o = (a[0] - o[0], a[1] - o[1])
    b_o = (b[0] - o[0], b[1] - o[1])
    inner_product = a_o[0]*b_o[0] + a_o[1]*b_o[1]
    length_a_o = math.sqrt(a_o[0]*a_o[0] + a_o[1]*a_o[1])
    length_b_o = math.sqrt(b_o[0]*b_o[0] + b_o[1]*b_o[1])
    return inner_product/(length_a_o*length_b_o)

def make_line(segment):
    """
    Computes a,b,c, where ax + by + c = 0 is the line of the segment.

    Arguments:
        segment - ((a_0, a_1), (b_0, b_1))
    
    Returns:
        a, b, c - constants of the line segment equation.
    """
    a_0 = segment[0].coor[0]
    a_1 = segment[0].coor[1]
    b_0 = segment[1].coor[0]
    b_1 = segment[1].coor[1]
    tmp1 = b_0 - a_0
    tmp2 = b_1 - a_1
    if tmp1 == 0:
        a = 1
        b = 0
        c = -a_0 # =b_1
    elif tmp2 == 0:
        a = 0
        b = 1
        c = -a_1 # =b_1
    else:
        a = -1.0/tmp1
        b = 1.0/tmp2
        c = 1.0*a_0/tmp1 - 1.0*a_1/tmp2
    return a, b, c

def gen_sign(tmp):
    if tmp > 0:
        sign = 1
    elif tmp < 0:
        sign = -1
    else:
        sign = 0
    return sign

def draw_dots(screen, node_list, font):
    i=1
    for node in node_list:
        center = (SCALE * node.coor[0] + SCALE/2, SCALE * (GRID_SIZE - node.coor[1]) + SCALE/2)
        pygame.draw.circle(screen, WHITE, center, SCALE/4+1)
        pygame.draw.circle(screen, node.colors_left[0], center, SCALE/4)
        text = font.render(str(i), True, (0, 128, 0))
        screen.blit(text, center)
        i += 1

def draw_lines(screen, segments):
    for segment in segments:
        point_1 = (SCALE * segment[0].coor[0] + SCALE/2, SCALE * (GRID_SIZE - segment[0].coor[1]) + SCALE/2)
        point_2 = (SCALE * segment[1].coor[0] + SCALE/2, SCALE * (GRID_SIZE - segment[1].coor[1]) + SCALE/2)
        pygame.draw.line(screen, WHITE, point_1, point_2)

def animation_main(node_list, segments):
    pygame.init()
    FONT = pygame.font.Font(None, 36)
    pygame.display.set_caption("MP2 1.2 ")
    size = (SCALE * (GRID_SIZE + 1), SCALE * (GRID_SIZE + 1))
    screen = pygame.display.set_mode(size)   
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False

        screen.fill((0, 0, 0))
        draw_lines(screen, segments)
        draw_dots(screen, node_list, FONT)
        pygame.display.flip()
        pygame.display.flip()

def is_separate_segments(s0, s1, x, y):
    ret_bool = False
    if max(s0.coor[0], s1.coor[0]) <= min(x.coor[0], y.coor[0]):
        ret_bool = True
    if min(s0.coor[0], s1.coor[0]) >= max(x.coor[0], y.coor[0]):
        ret_bool = True
    return ret_bool

def gen_graph(n):
    for x in range(0,n):
        Node()

def print_segments(segments):
    print("SEGMENTS")
    for item in segments:
        print(str(item[0].coor) + ' ' + str(item[1].coor))

def populate_adjacencies(segments):
    for item in segments:
        first_node = item[0]
        second_node = item[1]
        first_node.adj_list.append(second_node)
        second_node.adj_list.append(first_node)

def print_adjacencies():
    print("ADJACENCIES")
    for node in Node.node_list:
        print(node.name + ': ' + str(node.coor) + ' ===================================')
        for neighbor in node.adj_list:
            print(str(neighbor.coor) + ' ' + str(neighbor.colors_left))

def still_unassigned():
    for node in Node.node_list:
        if len(node.colors_left) > 1:
            return True
    return False

# def do_random(n):
#     varieables_list = Node.node_list
#     frontier = deque([]) # Stack: append(), popright()
#     frontier.append(copy.deepcopy(varieables_list))
#     unassigned_variables = copy.deepcopy(varieables_list)
#     random.shuffle(COLORS)
#     while still_unassigned():
#         varieables_list = frontier.pop()
#         node = random.choice(unassigned_variables)
#         unassigned_variables.remove(node)
#         for color in COLORS:
#             node.colors_left = [color]
#             satisfy_constraints = True
#             for neighbor in node.adj_list:
#                 if node.colors_left[0] == neighbor.colors_left[0]:
#                     satisfy_constraints = False
#             if satisfy_constraints == True:
#                 frontier.append(copy.deepcopy(varieables_list))

def is_safe_assign(node, color_to_assign):
    for neighbor in node.adj_list:
        if len(neighbor.colors_left) == 1 and color_to_assign == neighbor.colors_left[0]:
            return False
    return True

def random_helper(i):
    global TOTAL_ASSIGN
    if i == len(Node.node_list):
        return True
    node = Node.node_list[i]
    all_colors = copy.deepcopy(COLORS)
    random.shuffle(all_colors)
    for color in all_colors:
        if is_safe_assign(node, color):
            node.colors_left = [color]
            TOTAL_ASSIGN += 1
            if random_helper(i+1):
                return True
            all_colors.remove(color)
    return False

def do_random():
    random.shuffle(Node.node_list)
    if random_helper(0):
        print('Got a solution.')
        print('Total assignments: ' + str(TOTAL_ASSIGN))

# def color_helper(i, color_to_assign):
#     if not still_unassigned():
#         return True
#     node = Node.node_list[i]
#     if is_safe_assign(node, color_to_assign):
#         node.colors_left = [color]
#         if color_helper(i, color_to_assign):
#             return True
        

# def do_color(n):
#     if color_helper(0):
#         print('Has solution.')

def do_color(n):
    total_assignments = 0
    while still_unassigned():
        total_assignments += 1
        # Find most constrained variable to assign next.
        most_constrained = None
        fewest_colors_rem = 5
        for node in Node.node_list:
            if fewest_colors_rem > len(node.colors_left) and len(node.colors_left) > 1:
                fewest_colors_rem = len(node.colors_left)
                most_constrained = node
        print('Most constrained: ' + most_constrained.name + ' ' + str(most_constrained.adj_list))
        # Find value to try assigning.
        min_num_colors_reduced_by = n+1
        least_constraining = None
        for color in most_constrained.colors_left:
            num_colors_reduced_by = 0
            for neighbor in most_constrained.adj_list:
                if color in neighbor.colors_left:
                    num_colors_reduced_by += 1
            if num_colors_reduced_by < min_num_colors_reduced_by:
                min_num_colors_reduced_by = num_colors_reduced_by
                least_constraining = color
        print('Least constraining: ' + str(least_constraining) + ' ')
        most_constrained.colors_left = [least_constraining]
        for neighbor in most_constrained.adj_list:
            if least_constraining in neighbor.colors_left:
                neighbor.colors_left.remove(least_constraining)
        print('====================================================')
    for node in Node.node_list:
        print(node.name + ': ' + str(node.colors_left))
    print('Total assignments: ' + str(total_assignments))

def gen_rand_assign():
    for node in Node.node_list:
        r = random.randint(0,NUM_COLORS-1)
        node.colors_left = [COLORS[r]]

def try_new_color(curr_node_color):
    if curr_node_color == RED:
        return GREEN
    # else:
    #     return RED
    elif curr_node_color == GREEN:
        return BLUE
    elif curr_node_color == BLUE:
        return YELLOW
    elif curr_node_color == YELLOW:
        return RED

def print_list(l):
    for item in l:
        print(item),

def not_consistent():
    for node in Node.node_list:
        node_color = node.colors_left[0]
        for neighbor in node.adj_list:
            if node_color == neighbor.colors_left[0]:
                return True
    return False

def do_local(n):
    num_total_assignments = n

    gen_rand_assign()
    for node in Node.node_list:
        print(node.name + ': '),
        print_list(node.colors_left)
        print('')
    print('========================')
    while not_consistent():
        nodes_to_fix = deque([])
        visited = []
        curr_node = Node.node_list[0]
        nodes_to_fix.append(curr_node)
        while nodes_to_fix:
            if curr_node in visited:
                curr_node = nodes_to_fix.popleft()
                continue
            nodes_to_fix.append(curr_node)
            visited.append(curr_node)
            curr_node_color = curr_node.colors_left[0]
            for neighbor in curr_node.adj_list:
                if neighbor.colors_left[0] == curr_node_color:
                    neighbor.colors_left = [try_new_color(curr_node_color)]
                    num_total_assignments += 1
                nodes_to_fix.append(neighbor)
            curr_node = nodes_to_fix.popleft()
    print("Total number of assignments: " + str(num_total_assignments))
    for node in Node.node_list:
        print(node.name + ': ' + str(node.colors_left))

def print_num_edges():
    num_edges = 0
    for node in Node.node_list:
        for neighbor in node.adj_list:
            num_edges += 1
    num_edges = num_edges / 2.0
    print(num_edges)

def main(args):
    n = int(args[1])
    gen_graph(n)

    tested = []
    segments = []
    break_out_of_y = False
    for x in Node.node_list:
        tmp_segments = []
        for y in tested:
            for segment in segments:
                if segment[0] == x or segment[1] == y or segment[1] == x or segment[0] == y:
                    continue

                a,b,c = make_line(segment)
                tmp = a*x.coor[0] + b*x.coor[1] + c
                sign_x = gen_sign(tmp)
                tmp = a*y.coor[0] + b*y.coor[1] + c
                sign_y = gen_sign(tmp)

                if sign_x == sign_y: # Same side or on the line.
                    if sign_x == 0: # On the line.
                        if is_separate_segments(segment[0], segment[1], x, y): # If separate segments.
                            continue
                        else: # If overlapping segments.
                            break_out_of_y = True
                            break
                    else: # Same side.
                        continue

                elif sign_x * sign_y == 0: # Above segment in one dimension, below in the other.
                    continue

                else:
                    cos_axb = calculate_cosine_of_points(segment[0], x, segment[1])
                    cos_axy = calculate_cosine_of_points(segment[0], x, y)
                    cos_yxb = calculate_cosine_of_points(y, x, segment[1])
                    if min(cos_axy, cos_yxb) < cos_axb:
                        continue
                    else:
                        break_out_of_y = True
                        break

            if not break_out_of_y:
                tmp_segments.append((x,y))
            else:
                break_out_of_y = False
        segments.extend(tmp_segments)
        tested.append(x)
    populate_adjacencies(segments)

    start_time = time.time()
    # do_color(n)
    # do_random()
    do_local(n)
    end_time = time.time()
    print('Time to color: ' + str(end_time - start_time))

    # Debug prints and visualizations.
    # printGraph()
    # print('\n')
    print_num_edges()
    # print_segments(segments)
    # print_adjacencies()
    animation_main(Node.node_list, segments)


if __name__ == '__main__':
    main(sys.argv)
