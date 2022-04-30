import random
import numpy as np

# Класс, описывающий СЛАУ
class LinearSystem:
    def __init__(self, A):
        self.A = [[elem for elem in row] for row in A]
        self.b = np.dot(A, [[1] for i in range(0, len(A))])
        self.dim = len(A)

    # Определитель матрицы
    def det(self):
        return np.linalg.det(self.A)

    # Обратная матрица
    def inv(self):
        return np.linalg.inv(self.A)

    # Спектральный критерий обусловленности
    def cond_s(self):
        return np.linalg.norm(self.A) * np.linalg.norm(self.inv())

    # Объёмный критерий (критерий Ортеги)
    def cond_v(self):
        scal = lambda array: sum([x ** 2 for x in array]) ** 0.5
        
        det_A = abs(self.det())
        temp = [scal(a) for a in self.A]

        return np.prod(temp)/det_A

    # Угловой критерий
    def cond_a(self):
        scal = lambda array: sum([x ** 2 for x in array]) ** 0.5
        temp = [scal(self.A[i])*scal(np.array(self.inv()[:,i])) for i in range (0, self.dim)]

        return max(temp)

    # Сравнение решений СЛАУ и СЛАУ с погрешностью
    def solve_compare(self, seed, degree, _A = [], _b = []):
        if len(_A) == 0 and len(_b) == 0:
            _A, _b = self.variate(seed, degree)

        _x = np.linalg.solve(_A, _b)
        variation = sum([(x_i[0] - 1) ** 2 for x_i in _x]) ** 0.5
        return _x, variation

    # Вариация матрицы и правой части
    def variate(self, seed, degree):
        random.seed(seed)
        variation = random.random() / 100

        _A = [[elem for elem in row] for row in self.A]
        _b = [[elem for elem in row] for row in self.b]

        for i in range(0, self.dim):
            for j in range(0, self.dim):
                if _A[i][j] != 0:
                    _A[i][j] += round((variation + (i + 1)*(j + 1) / (10 ** degree)), degree + 2)
            if _b[i][0] != 0:
                _b[i][0] += round((variation + (i + 1) / (10 ** degree)), degree + 2)

        return _A, _b
    