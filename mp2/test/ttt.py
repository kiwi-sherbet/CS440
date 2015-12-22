import pygame
from random import sample,randint
pygame.font.init()
police = pygame.font.Font(None,50)

def game():
    scr = pygame.display.set_mode((302,302))
    rects = [scr.fill(-1,(x,y,100,100)).inflate(-25,-25) for x in 0,101,202 for y in 0,101,202]
    pygame.display.flip()
    
    grid = ['']*9
    combi = lambda: (grid[0:3],grid[3:6],grid[6:9],grid[0:7:3],grid[1:8:3],grid[2:9:3],grid[0:9:4],grid[2:7:2])
    ai = sample(range(9),9)
    player = randint(-1,0)
    
    def find_index(player):
        for i in range(9):
            if grid[i] == '':
                grid[i] = player
                if [player]*3 in combi():
                    grid[i] = ''
                    return i
                grid[i] = ''
        return None

    def play():
        if player:
            while True:
                ev = pygame.event.wait()
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    index = ev.pos[0]/101*3+ev.pos[1]/101
                    if grid[index] == '':
                        grid[index] = player
                        pygame.display.update(pygame.draw.ellipse(scr,0xa00000,rects[index],10))
                        break
        else:
            index = find_index(player)
            if index == None: index = find_index(~player)
            if index == None:
                while grid[ai[-1]] != '': ai.pop()
                index = ai.pop()
            grid[index] = player
            pygame.draw.line(scr,0x0000a0,rects[index].topright,rects[index].bottomleft,10)
            pygame.display.update(pygame.draw.line(scr,0x0000a0,rects[index].topleft,rects[index].bottomright,10))
            
    play()
    for coup in range(9):
        if coup == 8:
            txt = police.render("aucun gagnant",1,(0,0,0),(240,240,255))
            break
        player = ~player
        play()
        if [player]*3 in combi():
            txt = police.render(("ordi","humain")[player]+" gagne",1,(0,0,0),(240,240,255))
            break
    rect = txt.get_rect()
    rect.center = 156,156
    pygame.display.update(scr.blit(txt,rect))

game()
while True:
    ev = pygame.event.wait()
    if ev.type == pygame.QUIT: break
    elif ev.type == pygame.MOUSEBUTTONDOWN: game()