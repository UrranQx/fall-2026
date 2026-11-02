"""Собственная реализация алгоритма: PCA."""

# TODO: реализовать алгоритм согласно tasks/task-04/README.md.

import sys

import matplotlib.pyplot as plt
import numpy as np

""" 
Это не относится к PCA
F, y = make_regression(
    n_samples=1000,
    n_features=5,
    n_informative=5,
    effective_rank=2,
    tail_strength=0.1,
    noise=1,
    random_state=42,
) """

# make_regression возвращает наблюдения в случайном порядке. Поэтому график
# по индексу образца выглядит "штормящим": соседние точки не являются
# соседними значениями функции. Упорядочим данные для наглядной визуализации.


XY = np.arange(1, 64).reshape(-1, 7)
F = XY[:, :-1]
y = XY[:, -1]
FT_x_F = F.T @ F
# U, D, Vh = np.linalg.svd(FT_x_F, full_matrices=True)
# в лекции V @ D @ U.T
# Вместо U @ D V.T
# Получается в нашем варианте
V, D, Uh = np.linalg.svd(F, full_matrices=False)
# print(f"V =\n{V}\n\nD =\n{D}\n\nU.T =\n{Uh}\n")
# print(np.linalg.norm(F - V @ np.diag(D) @ Uh))  # разница минимальна. Разложение хорошее

eigs_FT_x_F, _ = np.linalg.eig(FT_x_F)
# eigs, _ = scipy.linalg.eig(F)# Не работает с не квадратными матрицами

eigs_FT_x_F = np.sort(eigs_FT_x_F.real, descending=True)


# print(f"Singular values (SVD) =\t\t\t\t{D}\n")

# print(f"eigs F.T @ F (EIG) =\t\t\t{eigs_FT_x_F}\n")
# print(f"sqrt of eigs F.T @ F (SQRT OF EIG))=\t{np.sqrt(eigs_FT_x_F)}\n")


# print(f"difff {D - np.sqrt(eigs_FT_x_F)}")
# Если разницы особо нет
# То мы показали, что сингулярные значения D полученные в SVD для матрицы F = V @ D @ U.T
# Равны соответсвенно корняи из собственных значений уже квадратной матрицы F.T @ F; т.е. sqrt(eig(F.T @ F))

# Пускай задан какой-то m - max
# Пускай он будет на 2 меньше чем оригинальный D
maximum = len(D)  # Инициализация
# print(f"maximum = {maximum}")
# Тогда в цикле будем считать функцию: E(maximum)
utu = Uh @ Uh.T
vtv = V.T @ V
Im = np.eye(maximum)
# print(f"[U.T @ U - Im] = \n{np.abs(utu - Im)}\tshape = {utu.shape}\n")
# print(f"[V.T @ V - Im] = \n{np.abs(vtv - Im)}\tshape = {vtv.shape}")

eigs_full_sum = np.sum(D**2)
em = []
for m in range(1, maximum):
    Em = np.sum(D[m:] ** 2) / eigs_full_sum
    print(Em)
    em.append(Em)


plt.plot(range(1, len(em) + 1), em, marker="o")
plt.xlabel("m")
plt.ylabel(r"$E_m$")
plt.grid(True)
plt.show()
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
G = F @ Uh_effective.T  # - Матрица новых признаков
F_restored = G @ Uh_effective
# print(f"G = \n{G}\nF=\n{F}")  # Новые признаки vs  старые признаки
# # print(f"Frestored- F = {F_restored - F}")

# print(f"||Frestored- F|| ^2 = {np.linalg.norm(F_restored - F)}")


print(np.linalg.norm(F - V_effective @ np.diag(L) @ Uh_effective))
beta = np.diag(1 / L) @ V_effective.T @ y
weigts_alpha = Uh_effective.T @ np.diag(1 / L) @ V_effective.T @ y
print(weigts_alpha)


model_out = G @ beta
model_out2 = F @ weigts_alpha


# а теперь попробуем прямой способ без PCA но с добавлением регурялизации
tau = 0.1
w_a = Uh.T @ np.linalg.inv(np.diag(D) ** 2 + tau * np.eye(len(D))) @ np.diag(D) @ V.T @ y
Ridge_regression_predict = F @ w_a


# Аналитическое решение через псевдообратную матрицу той же train-выборки.
weights_pinv = np.linalg.pinv(F) @ y # F^+ @ y
