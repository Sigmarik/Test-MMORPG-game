from BipAnim import *

anm = anim([])
anm.read('test_anim')
scr = pygame.display.set_mode([128, 128])
kg = True
while kg:
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == pygame.K_r:
                anm.play(scr)
            if event.key == pygame.K_p:
                anim.play(scr, True)
            if event.key == pygame.K_s:
                anim.write()
    anm.play(scr)
