"""Эксперименты лабораторной работы № 4. PCA."""

from pathlib import Path

OUTPUT_DIR = Path(__file__).with_name("figures")
RANDOM_STATE = 42


def main():
    from pca import em
    from visualization import plot_reconstruction_error

    plot_reconstruction_error(em, OUTPUT_DIR / "pca_reconstruction_error.png")
    # TODO: Датасет для линейной регрессии.
    # TODO: Реализация PCA через сингулярное разложение.
    # TODO: Эффективная размерность выборки.
    # TODO: Сравнение с эталонной реализацией.
    # TODO: Вывод.

    # а теперь попробуем прямой способ без PCA но с добавлением регурялизации
    tau = 0.1
    w_a = U @ np.linalg.inv(np.diag(D) ** 2 + tau * np.eye(len(D))) @ np.diag(D) @ V.T @ y
    Ridge_regression_predict = F @ w_a

    # Аналитическое решение через псевдообратную матрицу (Очень просто).
    # weights_pinv = np.linalg.pinv(F) @ y  # F^+ @ y
    print("ЛР4: заготовка. Реализуйте пункты задания из README.md.")


if __name__ == "__main__":
    main()

