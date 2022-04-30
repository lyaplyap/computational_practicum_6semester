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

    # Печать чисел обусловленности
    def print_conds(self):
        print('cond_s =', round(self.cond_s(), 6))
        print('cond_v =', round(self.cond_v(), 6))
        print('cond_a =', round(self.cond_a(), 6))

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

    # LU-разложение
    def lu_decomposition(self):
        rows, columns = np.shape(self.A)

        L = np.zeros((rows, columns))
        U = np.zeros((rows, columns))
        for i in range(columns):
            for j in range(i):
                total = 0
                for k in range(j):
                    total += L[i][k] * U[k][j]
                L[i][j] = (self.A[i][j] - total) / U[j][j]
            L[i][i] = 1
            for j in range(i, columns):
                total = 0
                for k in range(i):
                    total += L[i][k] * U[k][j]
                U[i][j] = self.A[i][j] - total
        return L, U

    # Решение СЛАУ с помощью LU-разложения
    def lu_solve(self):
        L, U = self.lu_decomposition()
        y = np.linalg.solve(L, self.b)
        x = np.linalg.solve(U, y)

        return x

    # QR-разложение
    def qr_decomposition(self):
        Q = np.eye(self.dim)
        R = np.copy(self.A)

        for i in range(self.dim):
            v = np.copy(R[i:, i]).reshape((self.dim - i, 1))
            v[0] = v[0] + np.sign(v[0]) * np.linalg.norm(v)
            v = v / np.linalg.norm(v)
            R[i:, i:] = R[i:, i:] - 2 * v @ v.T @ R[i:, i:]
            Q[i:] = Q[i:] - 2 * v @ v.T @ Q[i:]

        return Q[:self.dim].T, R[:self.dim]

     # Решение СЛАУ с помощью QR-разложения
    def qr_solve(self):
        Q, R = self.qr_decomposition()
        y = np.linalg.solve(Q, self.b)
        x = np.linalg.solve(R, y)

        return x

    # Решение СЛАУ с помощью метода простой итерации 
    def iterative_solve(self, eps):
        _A = np.zeros((self.dim, self.dim))
        _b = np.zeros((self.dim, 1))

        for i in range(0, self.dim):
            for j in range(0, self.dim):
                if i != j:
                    _A[i][j] = - (self.A[i][j] / self.A[i][i])
                else:
                    _A[i][j] = 0

        for i in range(0, self.dim):
            _b[i][0] = self.b[i][0] / self.A[i][i]
        
        x_k = np.copy(_b)
        iter_count = 1

        while (np.linalg.norm(x_k - 1) > eps):
            #print(x_k, '\n')
            x_k = np.add(np.dot(_A, x_k), _b) 
            iter_count += 1

        return x_k, iter_count

    # Решение СЛАУ с помощью метода Зейделя
    def seidel_solve(self, eps):
        _A = np.copy(self.A)
        _b = np.copy(self.b)

        x_k = np.zeros(self.dim)

        converge = False
        iter_count = 1
        while not converge:
            x_new = np.copy(x_k)
            for i in range(0, self.dim):
                sum1 = sum(_A[i][j] * x_new[j] for j in range(i))
                sum2 = sum(_A[i][j] * x_k[j] for j in range(i + 1, self.dim))
                x_new[i] = (_b[i] - sum1 - sum2) / _A[i][i]

            converge = np.sqrt(sum((x_new[i] - x_k[i]) ** 2 for i in range(0, self.dim))) <= eps
            x_k = x_new
            iter_count += 1

        return x_k, iter_count
    