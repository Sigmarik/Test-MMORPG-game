class rand:
    seed = 0
    def __init__(self, seed):
        self.seed = seed
    def randint(self, A = 0, B = 10**9):
        self.seed = (self.seed * sum([int(x) for x in str(self.seed)] + [3]) / 7 + 11) % 1000000
        return A + self.seed % (B - A)
