from linear_system import LinearSystem
from matrix_generators import diag, terdiag, hilbert, rand, identity
import pandas as pd
import numpy as np
import random


def lu_testing(linear_system, seed = 42):
    #print('\nМатрица A:\n', pd.DataFrame(linear_system.A))
    print('\nЧисла обусловленности матрицы A:')
    linear_system.print_conds()

    L, U = linear_system.lu_decomposition()
    
    #print('\nМатрица L:\n', pd.DataFrame(L))
    print('\nЧисла обусловленности матрицы L:')
    LinearSystem(L).print_conds()
    
    #print('\nМатрица U:\n', pd.DataFrame(U))
    print('\nЧисла обусловленности матрицы U:')
    LinearSystem(U).print_conds()

    print('\nРешение СЛАУ с помощью LU-разложения:')
    print(pd.DataFrame(linear_system.lu_solve()))

    print('\n----------------------------------------------------------------------')

def reg_testing(linear_system, seed = 42):
    random.seed(seed)
    scal = lambda vector, pwr = 0.5: sum([x[0] ** 2 for x in vector]) ** pwr

    min_diff = 42
    min_aplha = -1

    for alpha in range(12, 0, -1):
        A_temp = np.add(linear_system.A, (10 ** (-alpha))*np.array(identity(linear_system.dim)).reshape(linear_system.dim, linear_system.dim))
        b_temp = linear_system.b
        #b_temp = np.add(linear_system.b, (10 ** (-alpha))*np.array([[1] for i in range(0, linear_system.dim)])) 

        diff = scal(np.linalg.solve(A_temp, b_temp) - 1)
        print(f'\naplha = 10^(-{alpha})')
        print('\nЧисла обусловленности:')
        LinearSystem(A_temp).print_conds()
        print(f'\nНорма погрешности решения = {diff}')

        if diff < min_diff:
            min_diff = diff
            min_aplha = alpha

        print('\n----------')

    print(f'\nНаилучшее aplha = 10^(-{min_aplha})')
    x_rand = [[1 + random.randint(0, 10)/100] for i in range(0, linear_system.dim)]
    b_rand = np.dot(linear_system.A, x_rand)
    
    A_rand = np.add(linear_system.A, (10 ** (-min_aplha))*np.array(identity(linear_system.dim)))

    print(f'\nНорма погрешности наилучшего решения = {scal(np.add(np.linalg.solve(A_rand, b_rand), (-1)*np.array(x_rand)))}')
    print('(на другом случайном векторе)')
    
    print('\n----------------------------------------------------------------------')
    

def qr_testing(linear_system, seed = 42):
    #print('\nМатрица A:\n', pd.DataFrame(linear_system.A))
    print('\nЧисла обусловленности матрицы A:')
    linear_system.print_conds()

    Q, R = linear_system.qr_decomposition()
    
    #print('\nМатрица L:\n', pd.DataFrame(Q))
    print('\nЧисла обусловленности матрицы Q:')
    LinearSystem(Q).print_conds()
    
    #print('\nМатрица U:\n', pd.DataFrame(R))
    print('\nЧисла обусловленности матрицы R:')
    LinearSystem(R).print_conds()

    print('\nРешение СЛАУ с помощью QR-разложения:')
    print(pd.DataFrame(linear_system.qr_solve()))

    #print('\n', np.equal(linear_system.A, Q@R))
    print('\n', pd.DataFrame(linear_system.A))
    print('\n', pd.DataFrame(Q@R))

    print('\n----------------------------------------------------------------------')


# ТЕСТЫ
if __name__=='__main__':

    print('\n\t\tLU-разложение\n')
    # Матрица со случайными значениями
    print('\n\tМатрица со случайными значениями (порядка 20)')
    ls = LinearSystem(rand(20))
    lu_testing(ls)

    # Диагональная матрица
    print('\n\tДиагональная матрица (порядка 20)')
    ls = LinearSystem(diag(20))
    lu_testing(ls)
    
    # Матрицы Гильберта порядка 4, 7 и 10:
    for dim in [9, 12, 15]:
        print(f'\n\tМатрица Гильберта (порядка {dim})')
        ls = LinearSystem(hilbert(dim))
        lu_testing(ls)

    # Трёхдиагональная (плохая) матрица
    print(f'\n\tТрёхдиагональная (плохая) матрица (порядка 12)')
    ls = LinearSystem(terdiag(12))
    lu_testing(ls)

    # Трёхдиагональная матрица с диагональным преобладанием
    print(f'\n\tТрёхдиагональная матрица с диагональным преобладанием (порядка 5)')
    ls = LinearSystem([
        [5, 1, 0, 0, 0, 0],
        [1, 5, 2, 0, 0, 0],
        [0, 2, 5, 1, 0, 0],
        [0, 0, 1, 5, 2, 0],
        [0, 0, 0, 2, 5, 1],
        [0, 0, 0, 0, 1, 5]
    ])
    lu_testing(ls)

    '''
    # Регуляризация матриц Гильберта
    print('\n\t\tРЕГУЛЯРИЗАЦИЯ\n')
    for dim in [20, 25, 30]:
        print(f'\n\tМатрица Гильберта (порядка {dim})')
        reg_testing(LinearSystem(hilbert(dim)))
    '''
    
    print('\n\t\tQR-разложение\n')

    # Диагональная матрица
    print('\n\tДиагональная матрица (порядка 20)')
    ls = LinearSystem(diag(20))
    qr_testing(ls)
    
    # Матрицы Гильберта порядка 4, 7 и 10:
    for dim in [9, 12, 15]:
        print(f'\n\tМатрица Гильберта (порядка {dim})')
        ls = LinearSystem(hilbert(dim))
        qr_testing(ls)

    # Трёхдиагональная (плохая) матрица
    print(f'\n\tТрёхдиагональная (плохая) матрица (порядка 12)')
    ls = LinearSystem(terdiag(12))
    qr_testing(ls)

    # Трёхдиагональная матрица с диагональным преобладанием
    print(f'\n\tТрёхдиагональная матрица с диагональным преобладанием (порядка 5)')
    ls = LinearSystem([
        [5, 1, 0, 0, 0, 0],
        [1, 5, 2, 0, 0, 0],
        [0, 2, 5, 1, 0, 0],
        [0, 0, 1, 5, 2, 0],
        [0, 0, 0, 2, 5, 1],
        [0, 0, 0, 0, 1, 5]
    ])
    qr_testing(ls)