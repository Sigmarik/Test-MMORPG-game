from randomiser import *

def gen(seed, side):
    rs = rand(seed)
    answ = [[0] * side for _ in range(side)]
    for i in range(side):
        for j in range(side):
            num = rs.randint(0, 100000) / 1000
            chanses = [70, 90, 95]
            for ind, chanse in enumerate(chanses):
                if num < chanse:
                    answ[i][j] = ind
                    break
    return answ
