import pygame
import time
import os
from math import *

class frame:
    time_key = 0
    img = pygame.Surface([100, 100])
    def __init__(self, img, t_key):
        self.time_key = t_key
        self.img = img
    def __lt__(A, B):
        return A.time_key < B.time_key

class anim:
    frames = []
    max_time = 0
    def check(self):
        X = self.frames[0].img.get_width()
        Y = self.frames[0].img.get_height()
        for frm in self.frames:
            x = frm.img.get_width()
            y = frm.img.get_height()
            if not(x == X and y == Y):
                return False
        return True
    def __init__(self, frames = []):
        self.frames = frames
    def add(self, frm):
        self.frames.append(frm)
        self.frames = sorted(self.frames)
    def get_at(self, timee):
        for frm in self.frames[::-1]:
            if frm.time_key <= timee:
                return frm
        return None
    def play(self, scr, is_nat = False):
        tm = time.monotonic()
        while True:
            TM = time.monotonic()
            timee = TM - tm
            frm = self.get_at(timee).img
            if is_nat:
                scr.blit(frm, [0, 0])
            else:
                scr.blit(pygame.transform.scale(frm, [scr.get_width(), scr.get_height()]), [0, 0])
            pygame.display.update()
            if timee >= self.max_time:
                return self
    def write(self, name):
        try:
            os.mkdir(name)
        except OSError:
            for fl in os.listdir(name):
                os.remove(name + '/' + fl)
        tkeys = open(name + '/tkeys.txt', 'w')
        for i, frm in enumerate(self.frames):
            tkeys.write(str(frm.time_key) + '\n')
            pygame.image.save(frm.img, name + '/' + str(i) + '.bmp')
        tkeys.write(str(self.max_time))
        tkeys.close()
    def read(self, name):
        tkeys = open(name + '/tkeys.txt', 'r')
        for file in os.listdir(name):
            if file[0] in '1234567890':
                img = pygame.image.load(name + '/' + file)
                st = tkeys.readline()
                print(st)
                self.add(frame(img, float(st)))
        self.max_time = float(tkeys.readline())
        return self

class state_anim:
    anm = anim([])
    rot = 0
    centre = True
    def __init__(self, anm, rot = 0, centre = True):
        self.anm = anm
        self.rot = rot.copy()
        self.centre = centre

class game_anim:
    s_time = 0
    cur_frame = 1
    anm = None
    def __init__(self, anm):
        self.anm = anm
        self.s_time = time.monotonic()
    def draw(scr, pos):
        tm = time.monotonic() - self.s_time
        while self.cur_frame < len(self.anm.anm.frames) - 1 and self.anm.anm.frames[self.cur_frame + 1].time_key < tm:
            self.cur_frame += 1
        frm = self.anm.anm.frames[self.cur_frame].img
        rot = self.anm.rot
        img = frm.transform.rotate(frm, degrees(rot))
        sz = list(img.rect())
        if self.anm.centre:
            scr.blit(img, [pos[x] - sz[x] for x in range(2)])
        else:
            scr.blit(img, pos)
