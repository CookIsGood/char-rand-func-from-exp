import numpy as np
import math as mh
import matplotlib.pyplot as plt
import random
from scipy import interpolate


# Функция вычисления мат.ожидания
def mat(a):
    a = np.mean(a, axis=0)
    return a


# Функция вычисления исправленное дисперсии
def disp(x):
    matrixkv = np.square(x)
    mid = np.mean(matrixkv, axis=0)
    mat = np.mean(x, axis=0)
    b = 1.5 * (mid - np.square(mat))
    return b


# СКО
def sko(x):
    a = np.sqrt(x)
    print(a)
    return a


# Оценка корреляционной функции
# Входные данные: Исходная матрица и матрица мат.ожидания
def korrfunc(x, x1):
    a = np.shape(x)
    p = a[0]
    n = a[1]
    tk1 = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            tk = x1[i] * x1[j]
            tt = x[:, i] * x[:, j]
            a = (np.sum(tt) / p)
            tk1[i][j] = 1.5 * (a - tk)

    print(tk1)
    return tk1


# Оценка нормированной корреляционной функции
# Входные данные:Матрица корреляционной функции и матрица СКО
def normfunc(x, x1):
    a = np.shape(x)
    n = a[0]
    tk1 = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            tk1[i][j] = x[i][j] / (x1[i] * x1[j])

    print(tk1)
    return tk1


# Находим среднее мат.ожидания
def midmat(x):
    x = np.mean(x, axis=0)
    print(x)
    return x


# Находим среднюю дисперсию
def middisp(x):
    x = np.mean(x, axis=0)
    print(x)
    return x


# Находим среднее СКО
def midsko(x):
    a = np.sqrt(x)
    print(a)
    return a


# Находим нормированную корреляционную функцию стационарной случайной функции
# Входные данные: Матрица нормированной корреляционной функции
def normsfunc(x):
    a = np.shape(x)
    n = a[0]
    p = [1 for i in range(n)]
    t = np.arange(1, n + 1, 1)
    t = np.flip(t, axis=0)
    tk1 = [1 for i in range(n)]
    for i in range(n):
        p[i] = np.trace(x, offset=i)
        tk1[i] = p[i] / t[i]
    print(tk1)
    return (tk1)


def buildgrath(x1, a, b):
    fig, ax = plt.subplots()
    f = np.shape(x1)
    size = f[1]
    N = f[0]
    bmax = np.amax(b)
    bmin = np.amin(b)
    # Максимальное значение в исходной матрице
    maxznak = np.amax(x1)
    # Минимальное значение в исходной матрице
    minznak = np.amin(x1)
    # Определяем количесво t на интервале
    # (зависит от количества столбцов в исходной матрице и изначально заданных t)
    # size,size
    x = np.linspace(bmin, bmax, size)
    # Определяем границы на графике по y
    plt.ylim(minznak, maxznak)
    # Делаем интерполяцию(гладкие графики)
    tck, u = interpolate.splprep([x, a1], s=0)
    unew = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(unew, tck)
    # Строим график мат.ожидания
    plt.xlabel('t', color='black')
    plt.ylabel('x(t)', color='black')
    plt.grid(True)
    plt.title('Кусочно-линейные интерполяции реализации и мат.ожидание случайной функции', fontsize=7, color='black')
    ax.plot(x, a1, 'blue', out[0], out[1])
    # Cтроим графики x(t)
    tk1 = [[0] * N for i in range(N)]
    for i in range(N):
        tk1[i] = x1[i, :]
        ax.plot(x, tk1[i], c=np.random.rand(3, ))


# Отображаем изображение

# График кусочко-линейной интерполяции r(t)
def buildr(x1, b):
    fig1, ax1 = plt.subplots()
    plt.xlabel('t', color='black')
    plt.ylabel('r(t)', color='black')
    plt.grid(True)
    plt.title('График кусочно-линейной интерполяции r(t)', fontsize=7)
    tck, u = interpolate.splprep([b, x1], s=0)
    unew = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(unew, tck)
    maxznak = np.amax(b)
    minznak = np.amin(b)
    bmax = np.amax(x1)
    bmin = np.amin(x1)
    plt.xlim(minznak, maxznak)
    plt.ylim(bmin, bmax)
    ax1.plot(b, x1, 'blue', out[0], out[1])


# Задаем матрицу из условия
X = np.array([[1.80, 2.33, 2.10, 1.82, 1.14, 1.20, 2.34, 1.61],
              [1.32, 1.54, 1.70, 2.26, 3.24, 3.02, 3.14, 3.18],
              [1.32, 1.72, 1.62, 2.82, 1.26, 1.46, 2.56, 2.59],
              [3.66, 2.59, 3.44, 2.06, 2.58, 2.86, 2.62, 2.60],
              [0.66, 2.04, 1.44, 2.44, 3.36, 3.26, 2.46, 2.56],
              [1.08, 2.08, 1.23, 1.50, 3.36, 3.49, 1.82, 1.26],
              [0.60, 3.01, 1.23, 1.74, 1.08, 1.14, 1.46, 1.95],
              [2.64, 1.95, 2.77, 1.50, 2.34, 2.35, 2.52, 3.12],
              [1.92, 2.52, 2.21, 2.88, 3.42, 3.01, 1.98, 1.00],
              [0.90, 2.70, 1.33, 3.06, 2.34, 2.08, 2.30, 2.14],
              [2.82, 2.62, 2.77, 3.28, 1.44, 1.78, 2.44, 1.60],
              [2.70, 3.47, 3.12, 2.04, 1.68, 1.94, 2.74, 1.46]])
# Тестовая матрица
X1 = np.array([[-8, -3, -6, 0],
               [2, 4, -3, 8],
               [9, -4, 8, -4]])
# Вспомогательная матрица размер и значения изменять согласно условию и размерам
# Исходной матрицы(значения t)
print('Входная матрица')
print(X1)
print('Значения t')
helpmatrix = np.array([-2, -1, 0, 1])
print(helpmatrix)
print(" Мат. ожидания")
a1 = mat(X1)
print(a1)
print("Массив с исправленными дисперсиями. ")
a2 = disp(X1)
print(a2)
print("СКО")
a3 = sko(a2)
print("Оценка корреляционной функции")
a4 = korrfunc(X1, a1)
print("Оценка нормированной корреляционной функции")
a5 = normfunc(a4, a3)
print("Среднее мат.ожидания")
a6 = midmat(a1)
print("Средняя дисперсия")
a7 = middisp(a2)
print("Среднее СКО")
a8 = midsko(a7)
print("Значения для графика кусочно-линейной интерполяции функции r(t)")
a9 = normsfunc(a5)
# Построение графиков мат. ожидания и x(t)
a10 = buildgrath(X1, a1, helpmatrix)
# Еще один график
a11 = buildr(a9, helpmatrix)
plt.show(a11)
plt.show(a10)
