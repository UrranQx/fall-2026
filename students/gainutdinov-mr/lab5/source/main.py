"""Эксперименты лабораторной работы № 5. Логистическая регрессия."""

from pathlib import Path

import numpy as np
from data import load_data

OUTPUT_DIR = Path(__file__).with_name("figures")
RANDOM_STATE = 42


def sigmoid(x):
    # return np.exp(1/(1+np.exp(-x)))
    # Сигмоида без переполнения exp при больших отрицательных x.
    return np.exp(-np.logaddexp(0.0, -x))


def margin(X, y, w):
    return (X @ w) * y


def irls(F, y, tol=1e-5, max_iter=100):
    """
    F - матрица признаков (l x n), y - вектор меток (l x 1).
    Свободный член задаётся столбцом единиц в F - w0
    Возвращает коэффициенты w и число выполненных итераций.
    """

    # Начальное приближение - обычный МНК.
    w = np.linalg.pinv(F) @ y
    sigmas = sigmoid(margin(F, y, w))  # (l x 1)

    for iteration in range(1, max_iter + 1):
        # Ограничиваем вероятности только для вычисления весов и y_approx,
        # чтобы избежать деления на ноль при насыщении сигмоиды.
        gigachads = np.sqrt((1 - sigmas) * sigmas)  # (l x 1)
        moggers = np.sqrt((1 - sigmas) / sigmas)  # (l x 1)
        # Умножение каждой строки F на свой gamma без создания diag(l, l).
        F_approx = np.diag(gigachads) * F
        y_approx = y * moggers

        # МНК даёт приращение коэффициентов - направление Ньютона.
        delta_w = np.linalg.pinv(F_approx) @ y_approx
        h_t = 1.0  # вроде, метод Ньютано Рафсона позволяет как раз таки h = 1
        w += h_t * delta_w

        sigmas_new = sigmoid(margin(F, y, w))
        if np.max(np.abs(sigmas_new - sigmas)) <= tol:
            return w, iteration # предлагаю также выдавать и время- количество итераций/шагов за которое сошелся метод
        sigmas = sigmas_new

    print(f"IRLS не достиг точности {tol} за {max_iter} итераций. Подберите другие")


def main():
    X_train, X_test, y_train, y_test = load_data(
        test_size=0.3, random_state=RANDOM_STATE
    )
    # Столбец единиц соответствует свободному коэффициенту w[0].
    F_train = np.column_stack((np.ones(len(X_train)), X_train))
    F_test = np.column_stack((np.ones(len(X_test)), X_test))
    w, iterations = irls(F_train, y_train)

    y_pred = np.where(F_test @ w >= 0, 1, -1)
    # logaddexp - это такая функция, которая #TODO:
    train_loss = np.mean(np.logaddexp(0.0, -margin(F_train, y_train, w)))
    test_loss = np.mean(np.logaddexp(0.0, -margin(F_test, y_test, w)))
    print(f"IRLS: {iterations} итераций")
    print(f"Log-loss на обучении: {train_loss:.8f}")
    print(f"Log-loss на тесте: {test_loss:.8f}")
    print(f"Accuracy на тесте: {np.mean(y_pred == y_test):.4f}")
    # TODO: Сравнение с эталонной реализацией.
    # TODO: Графики



if __name__ == "__main__":
    main()
