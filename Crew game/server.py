import socket, threading, pygame
import time
from generation import *
from math import *
from random import randint
import os
#from BipAnim import *

pygame.init()

def dist(a, b = [0, 0]):
    return sqrt(sum([(a[i] - b[i]) * (a[i] - b[i]) for i in range(2)]))

def imload(name, pos = [0, 0]):
    img = pygame.image.load(name)
    img.set_colorkey(img.get_at(pos))
    return img

def blit_centred(scr, img, pos):
    W = img.get_width()
    H = img.get_height()
    scr.blit(img, [pos[0] - W // 2, pos[1] - H // 2])

def rotate(vec, rot):
    D = dist(vec)
    cur = atan2(*vec)
    an = cur + rot
    X = cos(an) * D
    Y = sin(an) * D
    return [X, Y]

WHEEL = imload('assets/wheel.bmp')
BODY = imload('assets/body.bmp')
SIGHT = imload('assets/sight.bmp')
WALL = imload('assets/1.bmp')
RESOURCE = imload('assets/2.bmp')
SPACE = pygame.image.load('assets/-1.bmp')
TIRE = imload('assets/tire.bmp')

WORLD_IMAGES = []
names = [-1, 1, 2, 3]
for name in names:
    WORLD_IMAGES.append(imload('assets/' + str(name) + '.bmp'))
DELTA_INDEX = 1
INDESTR_DELTA = 1

PI = 3.1415926535

font = pygame.font.Font(None, 24)

WORLD_SIDE = 200
SECTOR_SIDE = 20
BLOCK_SIDE = 64

blocks = [[2] * WORLD_SIDE for x in range(WORLD_SIDE)]
creatures = [[[]] * (WORLD_SIDE // SECTOR_SIDE)
             for x in range(WORLD_SIDE // SECTOR_SIDE)]

for i in range(1, WORLD_SIDE - 1):
    for j in range(1, WORLD_SIDE - 1):
        varis = list([-1] * 60 + [0, 1])
        blocks[i][j] = varis[randint(0, len(varis) - 1)]

lab = ['##########',
       '#        #',
       '#        #',
       '#         ',
       '#         ',
       '#         ',
       '#         ',
       '#        #',
       '#        #',
       '##########']

for i in range(10):
    for j in range(10):
        if lab[j][i] == '#':
            blocks[i][j] = 2
        else:
            blocks[i][j] = -1

world = pygame.Surface([WORLD_SIDE * BLOCK_SIDE] * 2)
world.fill([255, 200, 50])

def update_at(pos):
    global world
    i, j = pos
    im = WORLD_IMAGES[blocks[i][j] + DELTA_INDEX]
    world.blit(im, [i * BLOCK_SIDE, j * BLOCK_SIDE])

for i in range(WORLD_SIDE):
    for j in range(WORLD_SIDE):
        update_at([i, j])

shots = []
tires = []

def sign(x):
    if x == 0:
        return 0
    else:
        return int(x / abs(x))

def lerp(A, B, k):
    return [int(A[x] * k + B[x] * (1 - k)) for x in range(len(A))]

class shot:
    start = [0, 0]
    stop = [0, 0]
    s_time = 0
    power = 2
    color = [0, 0, 0]
    t_color = [255, 200, 50]
    life_time = 1
    def __init__(self, st, fn, color = [255, 100, 0], t_color = color, power = 2, lt = 1):
        self.start = st.copy()
        self.stop = fn.copy()
        self.color = color.copy()
        self.power = power
        self.life_time = lt
        self.s_time = time.monotonic()
    def draw(self, scr, player):
        tm = time.monotonic()
        if tm > self.s_time + self.life_time:
            shots.remove(self)
        else:
            transp = ((tm - self.s_time) / self.life_time)
            #print(transp)
            pygame.draw.line(scr, lerp(self.t_color, self.color, transp), [self.start[x] - player.pos[x] + SZS[x] // 2 for x in range(2)],
                             [self.stop[x] - player.pos[x] + SZS[x] // 2 for x in range(2)], self.power)
        return self

class tire:
    pos = [0, 0]
    rot = 0
    s_time = 0
    life_time = 1
    def __init__(self, pos, rot, lt = 1):
        self.pos = pos.copy()
        self.rot = rot
        self.life_time = lt
        self.s_time = time.monotonic()
    def draw(self, scr, player):
        tm = time.monotonic()
        if tm > self.s_time + self.life_time:
            tires.remove(self)
        else:
            transp = int(255 - ((tm - self.s_time) / self.life_time) * 255)
            tr = TIRE.copy()
            tr.set_alpha(transp)
            blit_centred(scr, pygame.transform.rotate(tr, self.rot), [self.pos[x] - player.pos[x] + SZS[x] // 2 for x in range(2)])
            #print([self.pos[x] - player.pos[x] + SZS[x] // 2 for x in range(2)])
        return self

def is_insec(a, b, c, d):
    arr = sorted([[a, 1], [b, -1], [c, 1], [d, -1]])
    sig = 0
    answ = 0
    for i, el in enumerate(arr):
        if sig == 2:
            answ += el[0] - arr[i - 1][0]
        sig += el[1]
    return answ

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

class hit_result:
    pos = [0, 0]
    is_border = False
    insec = [0, 0]
    def __init__(self, pos, insec, bord = False):
        self.pos = pos.copy()
        self.is_border = bord
        self.insec = insec.copy()

def aim_to(start, pos, is_deadly = False):
    delt = [pos[i] - start[i] for i in range(2)]
    D = dist(delt)
    N = int(D / (BLOCK_SIDE / 2))
    if N > 0:
        step = D / N
        delt = [delt[i] * step / D for i in range(2)]
        cur = start.copy()
        for i in range(N):
            cur = [cur[x] + delt[x] for x in range(2)]
            b_pos = [int(cur[x] / BLOCK_SIDE) for x in range(2)]
            if blocks[b_pos[0]][b_pos[1]] >= 0:
                return cur
            for enem in enemyes:
                if dist(enem.pos, cur) < enem.hit_size:
                    if is_deadly:
                        enem.dye()
                    return cur
    return pos

class creature:
    pos = [0, 0]
    hit_size = 30
    def __init__(self, pos, size = 30):
        global creatures
        self.pos = pos.copy()
        self.hit_size = size
        i = self.pos[0] // SECTOR_SIDE
        j = self.pos[1] // SECTOR_SIDE
        creatures[i][j].append(self)
    def get_hits(self, shift):
        global blocks
        res_pos = [self.pos[i] + shift[i] for i in range(2)]
        b_pos = [int(self.pos[i] // BLOCK_SIDE) for i in range(2)]
        answ = []
        for i in range(b_pos[0] - 1, b_pos[0] + 2):
            for j in range(b_pos[1] - 1, b_pos[1] + 2):
                if 0 <= i < WORLD_SIDE and 0 <= j < WORLD_SIDE:
                    if i == b_pos[0] or j == b_pos[1]:
                        cond1 = is_insec(i * BLOCK_SIDE, (i + 1) * BLOCK_SIDE, res_pos[0] - self.hit_size, res_pos[0] + self.hit_size)
                        cond2 = is_insec(j * BLOCK_SIDE, (j + 1) * BLOCK_SIDE, res_pos[1] - self.hit_size, res_pos[1] + self.hit_size)
                        if (cond1 or cond2) and blocks[i][j] >= 0:#    все блоки с отрицательным индексом физически прозрачны
                            answ.append(hit_result([i, j], [cond1, cond2]))
                    else:
                        length = BLOCK_SIDE * sqrt(2) / 2
                        side = BLOCK_SIDE / 2
                        cen = [(side + list([i, j])[x] * BLOCK_SIDE) for x in range(2)]
                        dirr = [(b_pos[x] - list([i, j])[x]) * side for x in range(2)]
                        point = [cen[x] + dirr[x] for x in range(2)]
                        #pygame.draw.circle(scr, [0, 255, 0], [int(x) for x in point], 3)
                        DIST = dist(point, res_pos)
                        if DIST < self.hit_size and blocks[i][j] >= 0:
                            vec = [res_pos[x] - point[x] for x in range(2)]
                            D = dist(vec)
                            answ.append(hit_result([i, j], [vec[x] * (self.hit_size - DIST) / D for x in range(2)], True))
        return answ
    def move(self, d_shift):
        global creatures, blocks
        ps = self.pos.copy()
        N = 10
        for i in range(N):
            shift = [d_shift.copy()[x] / N for x in range(2)]
            hits = self.get_hits(shift)
            #print(len(hits))
            b_pos = [self.pos[i] // BLOCK_SIDE for i in range(2)]
            for hit in hits:
                if not hit.is_border:
                    delt = 0
                    if hit.pos[0] == b_pos[0]:
                        delt = hit.insec[1]
                    elif hit.pos[1] == b_pos[1]:
                        delt = hit.insec[0]
                    arrow = [-(hit.pos[i] - b_pos[i]) * delt for i in range(2)]
                    shift = [shift[i] + arrow[i] for i in range(2)]
                else:
                    shift = [shift[i] + hit.insec[i] for i in range(2)]
            self.pos = [self.pos[i] + shift[i] for i in range(2)]
        return dist(ps, self.pos)
    def shoot(self, pos, color = None):
        if color == None:
            shots.append(shot(self.pos, pos))
        else:
            shots.append(shot(self.pos, pos, color))
        b_pos = [int(pos[x] / BLOCK_SIDE) for x in range(2)]
        if len(WORLD_IMAGES) - DELTA_INDEX - INDESTR_DELTA > blocks[b_pos[0]][b_pos[1]] >= 0:
            blocks[b_pos[0]][b_pos[1]] = -1
            update_at(b_pos)

enemyes = []

def linear(posA, posB):
    delta = [posA[x] - posB[x] for x in range(2)]
    D = dist(delta)
    return [delta[x] / D for x in range(2)]

class enemy(creature):
    atack_code = ''
    death_code = ''
    anim_ticks = []
    delta_frame = 0.25
    def __init__(self, dirname, pos):
        self.pos = pos.copy()
        folder = 'assets/' + dirname
        params_f = open(folder + '/' + 'params.txt')
        exec(params_f.read())
        atack_f = open(folder + '/' + 'atack.txt')
        self.atack_code = atack_f.read()
        death_f = open(folder + '/' + 'death.txt')
        self.death_code = death_f.read()
        for name in os.listdir(folder):
            print(name)
            if '.bmp' in name:
                print('newframe')
                self.anim_ticks.append(imload(folder + '/' + name))
    def dye(self):
        exec(self.death_code)
        enemyes.remove(self)
    def atack(self):
        exec(self.atack_code)
    def update(self, dt):
        global player
        dirrection = self.pathfinding_mode(player.pos, self.pos)
        delta_pos = [dirrection[x] * dt * self.speed for x in range(2)]
        result = self.move(delta_pos)
        delta_target = dist(self.pos, player.pos)
        if delta_target < self.atack_dist:
            self.atack()
        if result < dist(delta_pos) * 0.5:
            self.atack()
    def draw(self, scr):
        if dist(self.pos, player.pos) < 2000:
            blit_centred(scr, self.anim_ticks[int(time.monotonic() / self.delta_frame) % len(self.anim_ticks)], [self.pos[x] - player.pos[x] + SZS[x] // 2 for x in range(2)])

def decrese(x, delt):
    X = abs(x)
    if X <= delt:
        return 0
    else:
        return (X - delt) * sign(x)

class vehicle(creature):
    dirrect = 0
    vel = 0
    t_input = 0
    checkpoint = [0, 0]
    def __init__(self, pos, sz = 30):
        self.pos = pos.copy()
        self.hit_size = sz
    def update(self, com):
        if abs(self.t_input) < 1:
            self.t_input -= com[0] * delta_time * 3
        r_vel = self.move([cos(self.dirrect) * self.vel, sin(self.dirrect) * self.vel])
        self.vel = r_vel * sign(self.vel)
        self.dirrect -= self.t_input * self.vel * delta_time
        self.t_input = decrese(self.t_input, 1 * delta_time)
        self.vel = decrese(self.vel, 2.5 * delta_time)
        if abs(self.vel) <= 5:
            self.vel += com[1] * -5 * delta_time
    def draw(self, scr, player):
        base = BODY.copy()
        shift = [8, 13]
        szs = [base.get_width(), base.get_height()]
        OK = False
        for i in [1, -1]:
            for j in [-1, 1]:
                com = [i, j]
                origin = [(shift[x] if com[x] > 0 else szs[x] - shift[x]) for x in range(2)]
                wheel_pos = rotate(origin, self.dirrect)
                delta_wheel = rotate([-(szs[x] - shift[x]) // 2 for x in range(2)], self.dirrect)
                wheel_pos = [wheel_pos[x] + delta_wheel[x] for x in range(2)]
                if j == 1:
                    img = pygame.transform.rotate(WHEEL, degrees(self.t_input * 0.2))
                else:
                    img = WHEEL.copy()
                blit_centred(base, img, origin)
                if dist(self.checkpoint, self.pos) > 5 and j == 1:
                    tires.append(tire([wheel_pos[x] + self.pos[x] for x in range(2)], degrees(self.dirrect) if j == -1 else degrees(self.dirrect + self.t_input * 0.2), 3))
                    OK = True
        if OK:
            self.checkpoint = self.pos.copy()
        base.blit(BODY, [0, 0])
        blit_centred(scr, pygame.transform.rotate(base, degrees(-self.dirrect - PI / 2)), [self.pos[x] - player.pos[x] + SZS[x] // 2 for x in range(2)])

info = pygame.display.Info()

SZX = info.current_w
SZY = info.current_h
SZS = [SZX, SZY]

scr = pygame.display.set_mode([SZX, SZY], pygame.FULLSCREEN)

player = creature([100, 100], 20)
m_target = [0, 0]

veh = vehicle([250, 400])
p_veh = None

kg = True

tm = time.monotonic()
pygame.mouse.set_visible(False)
while kg:
    TM = time.monotonic()
    delta_time = TM - tm
    tm = TM
    scr.fill([255, 200, 50])
    mpos = list(pygame.mouse.get_pos())
    am = aim_to(player.pos, [mpos[x] + player.pos[x] - SZS[x] // 2 for x in range(2)])
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            kg = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                am_r = aim_to(player.pos, [mpos[x] + player.pos[x] - SZS[x] // 2 for x in range(2)], True)
                player.shoot(am_r)
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE]:
                kg = False
            if event.key in [pygame.K_w]:
                m_target[1] -= 1
            if event.key in [pygame.K_s]:
                m_target[1] += 1
            if event.key in [pygame.K_a]:
                m_target[0] -= 1
            if event.key in [pygame.K_d]:
                m_target[0] += 1
            if event.key in [pygame.K_e]:
                if p_veh == None:
                    if dist(player.pos, veh.pos) < 100:
                        p_veh = veh
                else:
                    p_veh = None
            if event.key == pygame.K_q:
                enemyes.append(enemy('expl_bot', [player.pos[x] + 300 for x in range(2)]))
        if event.type == pygame.KEYUP:
            if event.key in [pygame.K_w]:
                m_target[1] -= -1
            if event.key in [pygame.K_s]:
                m_target[1] += -1
            if event.key in [pygame.K_a]:
                m_target[0] -= -1
            if event.key in [pygame.K_d]:
                m_target[0] += -1
    left, top = [int((player.pos[x] // BLOCK_SIDE) - (SZS[x] // BLOCK_SIDE) - 1) for x in range(2)]
    right, down = [int((player.pos[x] // BLOCK_SIDE) + (SZS[x] // BLOCK_SIDE) + 1) for x in range(2)]
    scr.blit(world, [0, 0], [player.pos[x] - SZS[x] // 2 for x in range(2)] + SZS)
    for tr in tires:
        tr.draw(scr, player)
    for sht in shots:
        sht.draw(scr, player)
    if type(p_veh) == type(None):
        if m_target != [0, 0]:
            shft = m_target.copy()
            D = dist(shft)
            shft = [shft[i] * delta_time * 150 / D for i in range(2)]
        else:
            shft = [0, 0]
        player.move(shft)
        veh.update([0, 0])
        veh.draw(scr, player)
    else:
        player.pos = p_veh.pos
        p_veh.update(m_target)
        p_veh.draw(scr, player)
    pygame.draw.circle(scr, [0, 0, 255], [SZX // 2, SZY // 2], player.hit_size - 10)
    if delta_time > 0 and 30 > 1 / delta_time:
        scr.blit(font.render(str(int(1 / delta_time)), 1, [10, 10, 10, 0]), [10, 10])
    for enem in enemyes:
        enem.update(delta_time)
        enem.draw(scr)
    blit_centred(scr, SIGHT, mpos)
    pygame.display.update()
pygame.quit()



##sock = socket.socket()
##sock.bind(('', 9090))
##sock.listen(1)
##conn, addr = sock.accept()
##
##print ('connected:', addr)
##
##while True:
##    data = conn.recv(2)
##    if not data:
##        break
##    conn.send(data.upper())
##
##conn.close()
