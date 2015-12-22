#!/usr/bin/python
import copy
import pygame
import random
from random import sample,randint
from WarNode import *

SQUARE_SIZE = 60
GRID_SIZE = 6

RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RGB_GRAY = (125, 125, 125)
RGB_RED = (255, 0, 0)
RGB_GREEN = (0, 255, 0)
RGB_BLUE = (0, 0, 255)

GAME = "battle"
ATTRITION = 10

GREEN_AI = False
BLUE_AI = 1

DEPTH = 1

# Initialize game to initialize FONT(s) for below functions.
pygame.font.init()
FONT = pygame.font.Font(None,50)
SMALL_FONT= pygame.font.Font(None,30)


def on_board(coord):
    if coord[0] in range(0, GRID_SIZE) and coord[1] in range(0, GRID_SIZE):
            return True
    else:
        return False


def evaluate(board_ra_temp):
    value = 0
    for x in range(0, GRID_SIZE):
        for y in range(0, GRID_SIZE):
            if board_ra_temp[x][y].player != None:
                value += board_ra_temp[x][y].player*board_ra_temp[x][y].score
    return value    


def apply_blitz_rule(board_ra_temp, click_coord, player):
    board_ra_temp[click_coord[0]][click_coord[1]].player = player
    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    blitz = False
    captures = []
    for d in directions:
        x = click_coord[0] + d[0]
        y = click_coord[1] + d[1]
        if on_board((x,y)): 
            temp = board_ra_temp[x][y].player
            if temp == player:
                blitz = True
            elif temp == - player:
                captures.append((x,y))
    if blitz:
        for c in captures:
            board_ra_temp[c[0]][c[1]].player = player


def apply_battle_rule(board_ra_temp, click_coord, player, resources_ra):
    board_ra_temp[click_coord[0]][click_coord[1]].player = player
    green_count=0
    blue_count=0
    green_resources = 0
    blue_resources = 0
    for x in range(0, GRID_SIZE):
        for y in range(0, GRID_SIZE):
            if player == 1: 
                green_resources += resources_ra[x][y].score
                green_count += 1
            elif player == -1:
                blue_resources += resources_ra[x][y].score
                blue_count += 1 #if player = green, +=1, else +=0
            if resources_ra[x][y]>0:
                resources_ra[x][y] -= ATTRITION
    if green_count == 0:
        green_force = 0.00
    else:
        green_force = 1.0*green_resources/green_count
    if blue_count == 0:
        blue_force = 0.00
    else:
        blue_force = 1.0*blue_resources/blue_count
    # print accum_attrition
    # print green_force, blue_force
    board_ra_temp[click_coord[0]][click_coord[1]].player = player

    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    battle = False
    captures = []
    allies = 1
    for d in directions:
        x = click_coord[0] + d[0]
        y = click_coord[1] + d[1]
        if on_board((x,y)): 
            temp = board_ra_temp[x][y].player
            if temp == player:
                battle = True
                allies += 1
            elif temp == - player:
                captures.append((x,y))

    if battle:
        for c in captures:
            enemies = 1
            for d in directions:
                x = c[0] + d[0]
                y = c[1] + d[1]
                if on_board((x,y)):
                    temp = board_ra_temp[x][y].player
                    if temp == -player:
                        enemies += 1
            # print allies, enemies, (green_force*(player + 1)/2 - blue_force*(player -1)/2)*allies, (green_force*(-player + 1)/2 + blue_force*(player +1)/2)*enemies
            if (green_force*(player + 1)/2 - blue_force*(player -1)/2)*allies - (green_force*(-player + 1)/2 + blue_force*(player +1)/2)*enemies >0:
                # if allies is stronger than enemies at the battle
                board_ra_temp[c[0]][c[1]].player = player
                green_count +=player # if player = green, +=1, else -=1
                blue_count -=player #if player = blue, +=1, else -=1


def apply_duel_rule(board_ra_temp, click_coord, player, resources_ra):
    board_ra_temp[click_coord[0]][click_coord[1]].player = player
    green_count=0
    blue_count=0
    green_resources = 0
    blue_resources = 0
    for x in range(0, GRID_SIZE):
        for y in range(0, GRID_SIZE):
            if player == 1: 
                green_resources += resources_ra[x][y].score #if player = green, +=score, else +=0, add accumulate attrition
                green_count += 1 #if player = green, +=1, else +=0
            elif player == -1:
                blue_resources += resources_ra[x][y].score #if player = green, +=score, else +=0, add accumulate attrition
                blue_count += 1 #if player = green, +=1, else +=0
            if resources_ra[x][y]>0:
                resources_ra[x][y] -= ATTRITION
    if green_count == 0:
        green_force = 0.0
    else:
        green_force = 1.0*green_resources/green_count
    if blue_count == 0:
        blue_force = 0.0
    else:
        blue_force = 1.0*blue_resources/blue_count
    #print green_force, blue_force
    board_ra_temp[click_coord[0]][click_coord[1]].player = player

    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    duel = False
    captures = []
    for d in directions:
        x = click_coord[0] + d[0]
        y = click_coord[1] + d[1]
        if on_board((x,y)): 
            temp = board_ra_temp[x][y].player
            if temp == player:
                duel = True
            elif temp == - player:
                captures.append((x,y))

    if duel and player*(green_force - blue_force) > 0:
        for c in captures:
            board_ra_temp[c[0]][c[1]].player = player
            green_count +=player # if player = green, +=1, else -=1
            blue_count -=player #if player = blue, +=1, else -=1


def expand_node(parent, click_coord, node_count):
    green_score = 0
    blue_score = 0
    board_ra_temp = copy.deepcopy(parent[0])
    resources_ra_temp = copy.deepcopy(parent[0])
    if GAME == "battle":
        apply_battle_rule(board_ra_temp, click_coord, parent[1], resources_ra_temp)
    elif GAME == "duel":
        apply_duel_rule(board_ra_temp, click_coord, parent[1], resources_ra_temp)
    else:
        apply_blitz_rule(board_ra_temp, click_coord, parent[1])
    for x in range(0, GRID_SIZE):
        for y in range(0, GRID_SIZE):
            temp = board_ra_temp[x][y].player
            if temp==1:
                green_score += board_ra_temp[x][y].score*board_ra_temp[x][y].player
            elif temp==-1:
                blue_score -= board_ra_temp[x][y].score*board_ra_temp[x][y].player
    return (board_ra_temp, -parent[1], resources_ra_temp), node_count+1


def monte_carlo(parent, empty_spaces):
    if len(empty_spaces) == 0:
        return parent
    else:
        click_coord = random.choice(empty_spaces)
        child_empty_spaces = copy.copy(empty_spaces)
        child_empty_spaces.remove(click_coord)
        child, node_count_temp = expand_node(parent, click_coord, 0)
        candidate = monte_carlo(child, child_empty_spaces)
        return candidate


def monte_carlo_simulation(parent, empty_spaces, number):
    sumvalue = 0
    for n in range(0, number):
        model = monte_carlo(parent, empty_spaces)
        sumvalue += evaluate(model[0])
    return sumvalue/number

def minimax(parent, empty_spaces, depth, node_count):
    player = parent[1]
    if depth == 0:
        monte_carlo_simulation(parent, empty_spaces, 5)
        #print empty_spaces
        return parent, None, node_count# evaluation function here
    if player == 1:
        best_value = -1000000
        best_candidate = None
        best_click_coord = None
        for click_coord in empty_spaces:
            child_empty_spaces = copy.copy(empty_spaces)
            child_empty_spaces.remove(click_coord)
            child, node_count = expand_node(parent, click_coord, node_count)
            value = evaluate(parent[0])
            candidate, next_best_coord, node_count = minimax(child, child_empty_spaces, depth - 1, node_count)
            if value > best_value:
                best_candidate = candidate
                best_click_coord = click_coord
                #print best_click_coord
        return best_candidate, best_click_coord, node_count
    elif player == -1:
        best_value = 1000000
        best_candidate = None
        best_click_coord = None
        for click_coord in empty_spaces:
            child_empty_spaces = copy.copy(empty_spaces)
            child_empty_spaces.remove(click_coord)
            child, node_count = expand_node(parent, click_coord, node_count)
            value = evaluate(parent[0])
            candidate, next_best_coord, node_count = minimax(child, child_empty_spaces, depth - 1, node_count)
            if value < best_value:
                best_candidate = candidate
                best_click_coord = click_coord
        return best_candidate, best_click_coord, node_count


def alphabeta(parent, empty_spaces, alpha, beta, depth, node_count):
    player = parent[1]
    if depth == 0:
        monte_carlo_simulation(parent, empty_spaces, 5)
        #print empty_spaces
        return parent, None, node_count # evaluation function here
    if player == 1:
        best_value = -1000000
        best_candidate = None
        best_click_coord = None
        for click_coord in empty_spaces:
            child_empty_spaces = copy.copy(empty_spaces)
            child_empty_spaces.remove(click_coord)
            child, node_count = expand_node(parent, click_coord, node_count)
            value = evaluate(parent[0])
            candidate, next_best_coord, node_count = alphabeta(child, child_empty_spaces, alpha, beta, depth - 1, node_count)
            if value > best_value:
                best_candidate = candidate
                best_click_coord = click_coord
            if best_value > alpha:
                alpha = best_value
            if alpha >= beta:
                break
                #print best_click_coord
        return best_candidate, best_click_coord, node_count

    elif player == -1:
        best_value = 1000000
        best_candidate = None
        best_click_coord = None
        for click_coord in empty_spaces:
            child_empty_spaces = copy.copy(empty_spaces)
            child_empty_spaces.remove(click_coord)
            child, node_count = expand_node(parent, click_coord, node_count)
            value = evaluate(parent[0])
            candidate, next_best_coord, node_count = alphabeta(child, child_empty_spaces, alpha, beta, depth - 1, node_count)
            if value < best_value:
                best_candidate = candidate
                best_click_coord = click_coord
            if best_value < beta:
                beta = best_value
            if alpha >= beta:
                break
        return best_candidate, best_click_coord, node_count


def do_ai(board_ra, player, resources_ra, depth, ai_mode): ######working
    empty_spaces = []
    for x in range(0, GRID_SIZE):
        for y in range(0, GRID_SIZE):
            if board_ra[x][y].player == None:
                empty_spaces.append((x,y))
    #print len(empty_spaces)
    parent = (copy.copy(board_ra), player, resources_ra)
    node_count = 0
    if ai_mode == 1:
        best_candidate, best_click_coord, node_count = minimax(parent, empty_spaces, depth, node_count)
    else:
        alpha = -1000000
        beta = 1000000
        best_candidate, best_click_coord, node_count = alphabeta(parent, empty_spaces, alpha, beta, depth, node_count)
    #board_ra[best_click_coord[0]][best_click_coord[1]].player = player

    return best_click_coord, node_count


def draw_board(screen, board_ra): # draw the board.
    for x in range(0, GRID_SIZE):
        for y in range(0, GRID_SIZE):
            letter_pos = ((SQUARE_SIZE+1)*x + SQUARE_SIZE/4, (SQUARE_SIZE+1)*y + SQUARE_SIZE/4)
            square = ((SQUARE_SIZE+1)*x, (SQUARE_SIZE+1)*y, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, RGB_WHITE, square)
            text = FONT.render(str(board_ra[x][y].score), True, RGB_BLACK)
            screen.blit(text, letter_pos)


def draw_cicle(screen, coord, player): # draw circle on the given coodrination. 
    if player == 1:
        color = RGB_GREEN
    elif player == -1:
        color = RGB_BLUE
    square = ((SQUARE_SIZE+1)*coord[0], (SQUARE_SIZE+1)*coord[1], SQUARE_SIZE, SQUARE_SIZE)
    pygame.draw.ellipse(screen, color, square, 10)


def write_status(screen, green_score, blue_score, turn): # draw the board.
    rectangle = ((SQUARE_SIZE+1)*0, (SQUARE_SIZE+1)*GRID_SIZE, (SQUARE_SIZE+1) * GRID_SIZE -1, SQUARE_SIZE)
    pygame.draw.rect(screen, RGB_BLACK, rectangle)
    sentence_pos_1 = ((SQUARE_SIZE+1)*0 + SQUARE_SIZE/4, (SQUARE_SIZE+1)*GRID_SIZE + SQUARE_SIZE/5)
    sentence_pos_2 = ((SQUARE_SIZE+1)*0 + SQUARE_SIZE/4, (SQUARE_SIZE+1)*GRID_SIZE + 3*SQUARE_SIZE/5)    
    sentence_1 = "Green: " + str(green_score)
    sentence_2 = "Blue: " + str(blue_score)
    text_1 = SMALL_FONT.render(sentence_1, True, RGB_WHITE)
    text_2 = SMALL_FONT.render(sentence_2, True, RGB_WHITE)
    screen.blit(text_1, sentence_pos_1)
    screen.blit(text_2, sentence_pos_2)


def click(screen, board_ra, player):
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            return False, (-GRID_SIZE, -GRID_SIZE)
        elif e.type == pygame.MOUSEBUTTONDOWN:
            x = int(e.pos[0]/(SQUARE_SIZE+1))
            y = int(e.pos[1]/(SQUARE_SIZE+1))
            if on_board((x,y)) and board_ra[x][y].player == None:
                #board_ra[x][y].player = player
                return True, (x, y)


def end_turn(screen, board_ra):
    green_score = 0
    blue_score = 0
    for x in range(0, GRID_SIZE):
        for y in range(0, GRID_SIZE):
            temp = board_ra[x][y].player
            if temp==1:
                green_score += board_ra[x][y].score*board_ra[x][y].player
                draw_cicle(screen, (x,y), temp)
            elif temp==-1:
                blue_score -= board_ra[x][y].score*board_ra[x][y].player
                draw_cicle(screen, (x,y), temp)
    pygame.display.flip()
    return green_score, blue_score


def game():
    board_file = parse_cl_args()
    board_ra = read_board(board_file)
    resources_ra = copy.deepcopy(board_ra)

    click_coord = (-GRID_SIZE, -GRID_SIZE)
    player =1 # who will play the game first
    turn = 0
    green_score = 0
    blue_score = 0
    green_count = 0
    blue_count = 0
    game_running = True
    game_continue = False
    green_node_count_sum = 0
    blue_node_count_sum = 0

    screen = pygame.display.set_mode(((SQUARE_SIZE+1) * GRID_SIZE -1, (SQUARE_SIZE+1)* GRID_SIZE -1 + SQUARE_SIZE))
    pygame.display.set_caption("MP2 2 "+ GAME)
    draw_board(screen, board_ra)
    write_status(screen, green_score, blue_score, turn)
    pygame.display.flip()

    while game_running:
        if player==1: # green's turn
            if GREEN_AI != False:
                click_coord, node_count = do_ai(board_ra, player, resources_ra, DEPTH, GREEN_AI)
                print ("Green's node: " + str(node_count))
                green_node_count_sum += node_count
            else:
                game_running, click_coord = click(screen, board_ra, player)
            #print click_coord
        else: # blue's turn
            if BLUE_AI != False:
                click_coord, node_count = do_ai(board_ra, player, resources_ra, DEPTH, BLUE_AI)
                print ("Blue's node: " + str(node_count))
                blue_node_count_sum += node_count
            else:
                game_running, click_coord = click(screen, board_ra, player)

        if GAME == "battle":
            apply_battle_rule(board_ra, click_coord, player, resources_ra)
        elif GAME == "duel":
            apply_duel_rule(board_ra, click_coord, player, resources_ra)
        else:
            apply_blitz_rule(board_ra, click_coord, player)
        player = -player
        turn += 1
        draw_board(screen, board_ra)
        green_score, blue_score = end_turn(screen, board_ra)
        write_status(screen, green_score, blue_score, turn)
        pygame.display.flip()

        if turn == GRID_SIZE*GRID_SIZE :
            print "game over"
            print ("Avergae of Green's node: " + str(1.0*green_node_count_sum/18))
            print ("Avergae of Blue's node: " + str(1.0*blue_node_count_sum/18))

            game_running = False
            game_continue = True
            write_status(screen, green_score, blue_score, turn)
            if green_score > blue_score:
                game_over_text = FONT.render("Green WIN",1, RGB_WHITE, RGB_BLACK)
            elif green_score < blue_score:
                game_over_text = FONT.render("Blue WIN",1, RGB_WHITE,RGB_BLACK)
            else:
                game_over_text = FONT.render("DRAW",1,RGB_BLACK,RGB_GRAY)
            screen.blit(game_over_text, ((SQUARE_SIZE+1) * GRID_SIZE -1 -3* SQUARE_SIZE, (SQUARE_SIZE+1)*GRID_SIZE + SQUARE_SIZE/3))
            pygame.display.flip()

    return game_continue


def main():
    game_continue = game()
    while game_continue:
        e = pygame.event.wait()
        if e.type == pygame.QUIT: break
        elif e.type == pygame.MOUSEBUTTONDOWN: game()

if __name__ == '__main__':
    main()