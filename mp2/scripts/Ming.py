#!/usr/bin/python

import math
import random
import sys

class Node:
    num_nodes = 0
    quadrant_size = 10
    node_list = []

    #def __init__(self):
    #    self.name = str(Node.num_nodes)
    #    x = random.randint(0,Node.quadrant_size)
    #    y = random.randint(0,Node.quadrant_size)
    #    self.coor = (x,y)
    #    self.adj_list = []

    #    Node.node_list.append(self)
    #    Node.num_nodes += 1
    def __init__(self, x, y):
        self.name = str(Node.num_nodes)
        x = x
        y = y
        self.coor = (x,y)
        self.adj_list = []

        Node.node_list.append(self)
        Node.num_nodes += 1

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
    b_o = (b[0] - o[0], b[0] - o[1])
    inner_product = a_o[0]*b_o[0] + a_o[1]*b_o[1]
    length_a_o = math.sqrt((a_o[0])**2+ (a_o[1])**2)
    length_b_o = math.sqrt((b_o[0])**2+ (b_o[1])**2)
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

def main(args):
    n = int(args[1])
    #for x in range(0,n):
    #    Node()
    #printGraph()
    #print('\n')

    # Node(0,0)
    # Node(2,2)
    # Node(4,0)
    # Node(0,4)
    # Node(4,4)

    tested = []
    segments = []
    break_out_of_y = False
#####################
    i = 0
    while i != n:
        n = random.randint(0,Node.quadrant_size)
        m = random.randint(0,Node.quadrant_size)
        is_in_tested = False
        for node in Node.node_list:
            if (n, m) == node.coor:
                is_in_tested = True
        if is_in_tested:
            flag_adding_int = False
            continue
        else:
            flag_adding_int = True
            x = Node(n, m)
        #if x in Node.node_list:
        #    flag_adding_int = False
####################
#    for x in Node.node_list:
        tmp_segments = []
        for y in tested:
            print("X: " + str(x.coor))
            print("Y: " + str(y.coor))


            for segment in segments:
                if segment[0] == x:
                    print('ONE' + ' ' + str(segment[0].coor))
                    continue
                if segment[1] == y:
                    print('TWO' + ' ' + str(segment[1].coor))
                    continue
                if segment[1] == x:
                    print('THREE' + ' ' + str(segment[1].coor))
                    continue
                if segment[0] == y:
                    print('FOUR' + ' ' + str(segment[0].coor))
                    continue

                a,b,c = make_line(segment)
                tmp = a*x.coor[0] + b*x.coor[1] + c
                sign_x = gen_sign(tmp)
                tmp = a*y.coor[0] + b*y.coor[1] + c
                sign_y = gen_sign(tmp)
                print('sign: ' + str(sign_x) + ' ' + str(sign_y))

#########################################
                d,e,f = make_line((x,y))
                t = d*segment[0].coor[0] + e*segment[0].coor[1] + f
                other_sign_x = gen_sign(t)
                t = d*segment[1].coor[0] + e*segment[1].coor[1] + f
                other_sign_y = gen_sign(t)
                print('other sign: ' + str(other_sign_x) + ' ' + str(other_sign_y))
#########################################

                if sign_x == sign_y: # Same side or on the line.
                    if sign_x == 0: # At least one on the extension of the segment.
                        print("IN HERE")
                        if (max(segment[0].coor[0], segment[1].coor[0]) <= min(x.coor[0], y.coor[0])) or min(segment[0].coor[0], segment[1].coor[0]) >= max(x.coor[0], y.coor[0]): # If separate segments.
                            continue
                        else: # If overlap.
                            break
                    else: # Same side.
                        continue
                elif other_sign_x == other_sign_y:
                    if other_sign_x == 0: # At least one on the extension of the segment.
                        print("IN HERE")
                        if (max(x.coor[0], y.coor[0]) <= min(segment[0].coor[0], segment[1].coor[0])) or min(x.coor[0], y.coor[0]) >= max(segment[0].coor[0], segment[1].coor[0]): # If separate segments.
                            continue
                        else: # If overlap.
                            break
                    else: # Same side.
                        continue
                elif sign_x * sign_y == 0 or other_sign_x * other_sign_y == 0:
                # elif sign_x * sign_y == 0:
                    break_out_of_y = True
                    print('THIS BREAK')
                    break
                else:
                    cos_axb = calculate_cosine_of_points(segment[0], x, segment[1])
                    cos_axy = calculate_cosine_of_points(segment[0], x, y)
                    cos_yxb = calculate_cosine_of_points(y, x, segment[1])
                    print(cos_axb)
                    print(cos_axy)
                    print(cos_yxb)
                    if min(cos_axy, cos_yxb)<=cos_axb:
                        break_out_of_y = True
                        print('BREAKING')
                        break
                    else:
                        continue
            if not break_out_of_y:
                tmp_segments.append((x,y))
            else:
                break_out_of_y = False
                # break
            print("segments:")
            for item in segments:
                print(str(item[0].coor) + ' ' + str(item[1].coor))
            print('tmp_segments:')
            for item in tmp_segments:
                print(str(item[0].coor) + ' ' + str(item[1].coor))

        if len(tmp_segments) == 0 and len(segments) != 0:
            flag_adding_int = False
        segments.extend(tmp_segments)
        tested.append(x)
        flag_adding_int = True

        if flag_adding_int :
            i += 1
    print("FINAL")
    for item in segments:
        print(str(item[0].coor) + ' ' + str(item[1].coor))


if __name__ == '__main__':
    main(sys.argv)
