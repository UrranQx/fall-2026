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
from sklearn.model_selection import train_test_split

data, y = make_regression(
    n_samples=1000,
    n_features=5,
    n_informative=5,
    effective_rank=2,
    tail_strength=0.1,
    noise=1,
    random_state=42,
)

# make_regression возвращает наблюдения в случайном порядке. Поэтому график
# по индексу образца выглядит "штормящим": соседние точки не являются
# соседними значениями функции. Упорядочим данные для наглядной визуализации.
order = np.argsort(y)
data = data[order]
y = y[order]

# Все разложения и веса вычисляем только на обучающей части.
data, data_test, y, y_test = train_test_split(
    data, y, test_size=0.3, random_state=42
)
train_order = np.argsort(y)
test_order = np.argsort(y_test)
data, y = data[train_order], y[train_order]
data_test, y_test = data_test[test_order], y_test[test_order]


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

eigs_dataT_x_data = np.sort(eigs_dataT_x_data.real, descending=True)



print(f"Singular values (SVD) =\t\t\t\t{D}\n")

print(f"eigs F.T @ F (EIG) =\t\t\t{eigs_dataT_x_data}\n")
print(f"sqrt of eigs F.T @ F (SQRT OF EIG))=\t{np.sqrt(eigs_dataT_x_data)}\n")


print(f"difff {D - np.sqrt(eigs_dataT_x_data)}")
# Если разницы особо нет
# То мы показали, что сингулярные значения D полученные в SVD для матрицы F = V @ D @ U.T
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

eigs_full_sum = np.sum(D**2)
em = []
for m in range(1, maximum):
    Em = np.sum(D[m:]**2) / eigs_full_sum
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
tau = 0.1
weigts_a = (
    Uh.T @ np.linalg.inv(np.diag(D) ** 2 + tau * np.eye(len(D))) @ np.diag(D) @ V.T @ y
)
print(f"y = {y}")
print(f"ans = {data @ weigts_a}")
print(np.linalg.norm(data @ weigts_a - y))  # Ошибка на train при выбранном tau
# С tau = 0.1
plt.plot(range(1, len(y) + 1), y)
plt.plot(range(1, len(y) + 1), data @ weigts_a, color="pink")
plt.xlabel("x")
plt.ylabel(r"$y$")
plt.grid(True)
plt.show()

# Аналитическое решение через псевдообратную матрицу той же train-выборки.
weights_pinv = np.linalg.pinv(data) @ y

# На тесте используем направления PCA, найденные на train, без нового SVD.
G_test = data_test @ Uh_effective.T
predictions = {
    "PCA": (G @ beta, G_test @ beta),
    "Псевдообратная матрица": (data @ weights_pinv, data_test @ weights_pinv),
    f"Ridge SVD (tau={tau:g})": (data @ weigts_a, data_test @ weigts_a),
}

print("\nСравнение на train и test (RMSE):")
print(f"{'Метод':<25} {'Train':>12} {'Test':>12}")
for name, (train_prediction, test_prediction) in predictions.items():
    train_rmse = np.sqrt(np.mean((train_prediction - y) ** 2))
    test_rmse = np.sqrt(np.mean((test_prediction - y_test) ** 2))
    print(f"{name:<25} {train_rmse:>12.6f} {test_rmse:>12.6f}")

plt.figure()
test_indices = range(1, len(y_test) + 1)
plt.plot(test_indices, y_test, "o-", label="y_test")
for name, (_, test_prediction) in predictions.items():
    plt.plot(test_indices, test_prediction, marker="x", linestyle="--", label=name)
plt.xlabel("Индекс тестового образца (по возрастанию y_test)")
plt.ylabel(r"$y$")
plt.title("Сравнение предсказаний на тестовой выборке")
plt.legend()
plt.grid(True)
plt.show()
