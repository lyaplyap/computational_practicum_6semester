import random

# Диагональная матрица
def diag(n, seed = 42):
    random.seed(seed)
    temp = [[] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                temp[i].append(random.randint(1, n))
            else:
                temp[i].append(0)
    return temp 

# Трёхдиагональная ("плохая") матрица 
# (с 2 на главной диагонали и −1 на побочных)
def terdiag(n):
    temp = [[] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                temp[i].append(2)
            elif j == i - 1 or j == i + 1:
                temp[i].append(-1)
            else:
                temp[i].append(0)
    return temp

# Матрица Гильберта
def hilbert(n):
    return [[1/(i + 1 + j) for j in range(0, n)] for i in range(0, n)]

# Матрица со случайными значениями
def rand(n, seed = 42):
    random.seed(seed)
    temp = [[] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            temp[i].append(random.randint(1, n))
    return temp 

