import random

random.seed('ai_class_03')

POPULATION_SIZE = 5
MUTATION_RATE = 0.2
SIZE = 6

actual = []


# 해답을 생성하는 연산
def generate_answer_randomly():
    global actual

    completed = set()
    while len(actual) < SIZE:
        gene = random.randint(0, 9)
        if gene in completed: continue
        completed.add(gene)
        actual.append(gene)


# 초기 염색체를 생성하는 연산
def generate_initial_population(n):
    initial_population = []
    for i in range(n):
        initial_population.append(list())
        completed = set()
        while len(initial_population[i]) < SIZE:
            gene = random.randint(0, 9)
            if gene in completed: continue
            completed.add(gene)
            initial_population[i].append(gene)
    return initial_population


# 적합도를 계산하는 연산
def evaluate_fitness(solution):
    fitness_value = 0
    index = 0
    for expect in solution:
        if expect == actual[index]:
            fitness_value += 5
        elif expect in actual:
            fitness_value += 1
        index += 1
    return fitness_value


# 적합도를 기준으로 염색체를 선택하는 연산
def select(pop):
    max_value = sum([evaluate_fitness(sol) for sol in current_population])
    pick = random.uniform(0, max_value)
    current = 0

    for sol in pop:
        current += evaluate_fitness(sol)
        if current > pick:
            return sol


# 선택된 염색체로부터 자손을 생성하는 연산
def crossover(pop):
    parent_1 = select(pop)
    parent_2 = select(pop)

    index = random.randint(1, SIZE - 1)

    child1 = parent_1[:index] + parent_2[index:]
    child2 = parent_2[:index] + parent_1[index:]

    return (child1, child2)


# 돌연변이 연산
def mutate(c):
    for i in range(SIZE):
        if random.random() < MUTATION_RATE:
            while True:
                gene = random.randint(0, 9)
                if gene in c: continue
                c[i] = gene
                break


if __name__ == "__main__":
    for i in range(8):
        actual = []
        best_score = -1
        n_iter = 0

        generate_answer_randomly()
        current_population = generate_initial_population(POPULATION_SIZE)

        while best_score < SIZE * 5:
            max_f = max(current_population, key=lambda f: evaluate_fitness(f))

            if evaluate_fitness(max_f) > best_score:
                best_score = evaluate_fitness(max_f)
                best_solution = max_f

            new_population = []

            for _ in range(POPULATION_SIZE // 2):
                child_1, child_2 = crossover(current_population)

                new_population.append(child_1)
                new_population.append(child_2)

            current_population = new_population.copy()

            for c in current_population:
                mutate(c)

            n_iter += 1

        print(f'POPULATION_SIZE = {POPULATION_SIZE}\t\t\tn_iter = {n_iter}')
        print(f'ANSWER = {actual}\t\tSOLUTION = {best_solution}')
        print('-------------------------------------------------------------')

        POPULATION_SIZE *= 2  # 염색체 개수를 늘려가며 실험한다.

"""
    출력 결과 : 염색체 개수가 늘어날수록, n_iter은 감소하는 경향을 보인다.
    
    POPULATION_SIZE = 5			n_iter = 427
    ANSWER = [9, 4, 5, 0, 7, 6]		SOLUTION = [9, 4, 5, 0, 7, 6]
    -------------------------------------------------------------
    POPULATION_SIZE = 10			n_iter = 109
    ANSWER = [4, 7, 9, 5, 3, 1]		SOLUTION = [4, 7, 9, 5, 3, 1]
    -------------------------------------------------------------
    POPULATION_SIZE = 20			n_iter = 13
    ANSWER = [7, 8, 4, 0, 6, 3]		SOLUTION = [7, 8, 4, 0, 6, 3]
    -------------------------------------------------------------
    POPULATION_SIZE = 40			n_iter = 96
    ANSWER = [6, 9, 3, 0, 8, 7]		SOLUTION = [6, 9, 3, 0, 8, 7]
    -------------------------------------------------------------
    POPULATION_SIZE = 80			n_iter = 27
    ANSWER = [0, 8, 3, 7, 2, 6]		SOLUTION = [0, 8, 3, 7, 2, 6]
    -------------------------------------------------------------
    POPULATION_SIZE = 160			n_iter = 5
    ANSWER = [5, 0, 7, 8, 1, 3]		SOLUTION = [5, 0, 7, 8, 1, 3]
    -------------------------------------------------------------
    POPULATION_SIZE = 320			n_iter = 9
    ANSWER = [7, 4, 3, 5, 2, 8]		SOLUTION = [7, 4, 3, 5, 2, 8]
    -------------------------------------------------------------
    POPULATION_SIZE = 640			n_iter = 7
    ANSWER = [2, 9, 3, 8, 1, 4]		SOLUTION = [2, 9, 3, 8, 1, 4]
"""


