from linear_system import LinearSystem
from matrix_generators import diag, terdiag, hilbert, rand, identity
import pandas as pd
import numpy as np

def iterative_testing(linear_system, eps_list):
    for eps in eps_list:
        solve, count = linear_system.iterative_solve(10 ** (-eps))

        print(f'\nМетод простой итерации (eps = 10^(-{eps})):')
        # Найденное решение
        #print(pd.DataFrame(solve))
        print(f'Количество итераций: {count}')

    print('\n----------------------------------------------------------------------')

def seidel_testing(linear_system, eps_list):
    for eps in eps_list:
        solve, count = linear_system.seidel_solve(10 ** (-eps))

        print(f'\nМетода Зейделя (eps = 10^(-{eps})):')
        # Найденное решение
        #print(pd.DataFrame(solve))
        print(f'Количество итераций: {count}')

    print('\n----------------------------------------------------------------------')


# ТЕСТЫ
if __name__=='__main__':
    print('\n\t\tЗадание 4. Итерационные методы решения СЛАУ\n')
    
    # Простая итерация
    print('\n\n\t------------ Метод простой итерации ------------\n')

    # Трёхдиагональная (плохая) матрица
    for dim in [20, 40, 60]:
        print(f'\n\tТрёхдиагональная (плохая) матрица (порядка {dim})')
        ls = LinearSystem(terdiag(dim))
        iterative_testing(ls, [2, 4, 6, 8])

    # Матрица 1 из методички
    print(f'\n\tМатрица 1 из методички А.Н.Пакулиной (порядка 2)')
    ls = LinearSystem([
        [-401.98, 200.34], 
        [1202.04, -602.32]
    ])
    iterative_testing(ls, [2, 4, 6, 8])  

    # Матрица 2 из методички
    print(f'\n\tМатрица 2 из методички А.Н.Пакулиной (порядка 2)')
    ls = LinearSystem([
        [-402.90, 200.70],
        [1204.20, -603.60]
    ])
    iterative_testing(ls, [2, 4, 6, 8])


    # Метод Зейделя
    print('\n\n\t---------------- Метод Зейделя ----------------\n')

    # Трёхдиагональная (плохая) матрица
    print('\n\tТрёхдиагональная (плохая) матрица (порядка 10)')
    ls = LinearSystem(terdiag(20))
    seidel_testing(ls, [2, 4, 6, 8])

    # Матрица 1 из методички
    print(f'\n\tМатрица 1 из методички А.Н.Пакулиной (порядка 2)')
    ls = LinearSystem([
        [-401.98, 200.34], 
        [1202.04, -602.32]
    ])
    seidel_testing(ls, [2, 4, 6, 8]) 

    # Матрицы Гильберта порядка 40, 60 и 100:
    #for dim in [20, 40, 60]:
    #    print(f'\n\tМатрица Гильберта (порядка {dim})')
    #    ls = LinearSystem(hilbert(dim))
    #    seidel_testing(ls, [2, 4, 5])

    # Большая трёхдиагональная (плохая) матрица
    print('\n\tТрёхдиагональная (плохая) матрица (порядка 300)')
    ls = LinearSystem(terdiag(300))
    iterative_testing(ls, [2, 3, 4, 5, 6])