from chess_board import chess_board
import random

size = 8
board = chess_board(size=8)


population = []
found_solution = False
for i in range(0, 100):
    population.append([random.randint(0, 63) for i in range(8)])

for generation in range(0, 1000000):
    if found_solution == True:
        break
    f_best = 0
    c_best = []
    second_best = 0
    c_second_best = []
    for chromosome in population:
        f = board.nonattacking_pairs(chromosome)
        if f > f_best:
            f_best = f
            c_best = chromosome
        if f > second_best and chromosome != c_best:
            second_best = f
            c_second_best = chromosome

    board.show_state(c_best)
    if f_best == 28:
        found_solution = True
        board.show_state(c_best)
        print("DONE:", c_best)
        print("Generation:", generation)
    new_population = []
    for i in range(0, 100):
        if i % 2 == 0:
            new_c = c_best.copy()
            new_c[random.randrange(
                0, 8)] = c_second_best[random.randrange(0, 8)]
            new_c[random.randrange(
                0, 8)] = c_second_best[random.randrange(0, 8)]
            mutation = random.randrange(0, 7)
            if mutation >= 4:
                new_c[mutation] = random.randrange(0, 63)
            new_population.append(new_c)
        else:
            new_second = c_second_best.copy()
            new_second[random.randrange(
                0, 8)] = c_best[random.randrange(0, 8)]
            new_second[random.randrange(
                0, 8)] = c_best[random.randrange(0, 8)]

            mutation = random.randrange(0, 7)
            if mutation >= 4:
                new_second[mutation] = random.randrange(0, 63)
            new_population.append(new_second)
    population = new_population.copy()