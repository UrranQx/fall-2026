"""Собственная реализация алгоритма: PCA."""

# TODO: реализовать алгоритм согласно tasks/task-04/README.md.

import sys

import numpy as np

np.set_printoptions(
    threshold=sys.maxsize,  # выводить все элементы без ...
    linewidth=sys.maxsize,  # не переносить длинные строки
    precision=4,
    suppress=True,
)
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

data, y = make_regression(
    n_samples=15,
    n_features=5,
    n_informative=5,
    effective_rank=2,
    tail_strength=0.1,
    noise=0.01,
    random_state=42,
)

# make_regression возвращает наблюдения в случайном порядке. Поэтому график
# по индексу образца выглядит «штормящим»: соседние точки не являются
# соседними значениями функции. Упорядочим данные для наглядной визуализации.
order = np.argsort(y)
data = data[order]
y = y[order]


print(f"data = \n{data}\n")
dataT_x_data = data.T @ data
# U, D, Vh = np.linalg.svd(dataT_x_data, full_matrices=True)
# в лекции V @ D @ U.T
# Вместо U @ D V.T
# Получается в нашем варианте
V, D, Uh = np.linalg.svd(data, full_matrices=False)
print(f"V =\n{V}\n\nD =\n{D}\n\nU.T =\n{Uh}\n")
print(
    np.linalg.norm(data - V @ np.diag(D) @ Uh)
)  # разница минимальна. Разложение хорошее

eigs_dataT_x_data, _ = np.linalg.eig(dataT_x_data)
# eigs, _ = scipy.linalg.eig(data)# Не работает с не квадратными матрицами
eigs = D

eigs_dataT_x_data = np.sort(eigs_dataT_x_data.real, descending=True)
eigs = np.sort(eigs, descending=True)


print(f"eigs F (SVD) =\t\t\t\t{eigs}\n")

print(f"eigs F.T @ F (EIG) =\t\t\t{eigs_dataT_x_data}\n")
print(f"sqrt of eigs F.T @ F (SQRT OF EIG))=\t{np.sqrt(eigs_dataT_x_data)}\n")


print(f"difff {eigs - np.sqrt(eigs_dataT_x_data)}")
# Если разницы особо нет
# То мы показали, что собственные значения D полученные в SVD для матрицы F = V @ D @ U.T
# Равны соответсвенно корняи из собственных значений уже квадратной матрицы F.T @ F; т.е. sqrt(eig(F.T @ F))

# Пускай задан какой-то m - max
# Пускай он будет на 2 меньше чем оригинальный D
maximum = len(D)  # Инициализация
print(f"maximum = {maximum}")
# Тогда в цикле будем считать функцию: E(maximum)
utu = Uh @ Uh.T
vtv = V.T @ V
Im = np.eye(maximum)
print(f"[U.T @ U - Im] = \n{np.abs(utu - Im)}\tshape = {utu.shape}\n")
print(f"[V.T @ V - Im] = \n{np.abs(vtv - Im)}\tshape = {vtv.shape}")

eigs_full_sum = np.sum(D)
em = []
for m in range(1, maximum):
    Em = np.sum(D[m:]) / eigs_full_sum
    print(Em)
    em.append(Em)


plt.plot(range(1, len(em) + 1), em, marker="o")
plt.xlabel("m")
plt.ylabel(r"$E_m$")
plt.grid(True)
plt.show(block=False)
plt.close()


# F = G @ U.T
# D = sqrt (eig(F.T @ F)) === _, D, _ = svd(F)
# G = V @ D === G = V @ sqrt(LAMBDA) where LAMBDA = eig(F.T @ F)


# Пускай
effective_m = 3
Uh_effective = Uh[:effective_m]  # Для U (а не U.T) берем топ m собственных вектором,
V_effective = V[:, :effective_m]  # А может надо было V[:, :effective]??? TODO уточнить
# а затем снова транспонируем, получается берем топ m строк U.T (Uh)

L = D[:effective_m]
G = data @ Uh_effective.T  # - Матрица новых признаков
F_restored = G @ Uh_effective
print(f"G = \n{G}\nF=\n{data}")  # Новые признаки vs  старые признаки
# print(f"Frestored- F = {F_restored - data}")

print(f"||Frestored- F|| ^2 = {np.linalg.norm(F_restored - data)}")

# Посчитаем ошибку
# || G @ Uh_effective @ alpha - targets|| ^ 2 -> min
# beta = Uh_effective @ alpha
# beta = D.inv() @ V.T @ targets
# alhpa = Uh_effective.T @ beta
# = np.sum([])

print(f"L = \n{L}")
print(f"L^-1 =\n{np.linalg.inv(np.diag(L))}")
print(f"1/L = {1 / L}")
# Вроде бы вот в лекции пишется про
# # beta = D.inv() @ V.T @ targets
# Но что если вот на самом деле имеется ввиду не D.inv, а L.inv? И не одно ли это тоже, если мы будем рассматривать задачу уже в конткесте
# new Data = G
# new alhpa = beta
# targets = targets
# Но это что значит, мне надо делать SVD теперь и для G?
# А вдруг мне уже доастаточно изанчального результата SVD,
# И V мне надо тоже сделать теперь V_effective = V[:effective_m]

# Проверим если F = V_effective @ L @ Uh_effective
print(np.linalg.norm(data - V_effective @ np.diag(L) @ Uh_effective))
beta = np.diag(1 / L) @ V_effective.T @ y
weigts_alpha = Uh_effective.T @ np.diag(1 / L) @ V_effective.T @ y
print(weigts_alpha)
print(weigts_alpha.shape)
print(f"y = {y}")
print(f"ans = {G @ beta}")
print(np.linalg.norm(G @ beta - y))
print(np.linalg.norm(data @ weigts_alpha - y))

plt.plot(
    range(1, len(y) + 1),
    y,
)
plt.plot(range(1, len(y) + 1), G @ beta, color="pink")
plt.xlabel("x")
plt.ylabel(r"$y$")
plt.grid(True)
plt.show()

# а теперь попробуем прямой способ без PCA
print(len(D))
tau = 0.00
weigts_a = (
    Uh.T @ np.linalg.inv(np.diag(D) ** 2 + tau * np.eye(len(D))) @ np.diag(D) @ V.T @ y
)
print(f"y = {y}")
print(f"ans = {data @ weigts_a}")
print(np.linalg.norm(data @ weigts_a - y))  # Без регуряризации = 0.07265627416184514
# С tau = 0.1
plt.plot(range(1, len(y) + 1), y)
plt.plot(range(1, len(y) + 1), data @ weigts_a, color="pink")
plt.xlabel("x")
plt.ylabel(r"$y$")
plt.grid(True)
plt.show()
