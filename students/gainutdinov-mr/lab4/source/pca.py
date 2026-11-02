"""Собственная реализация алгоритма: PCA."""
# U, D, Vh = np.linalg.svd(FT_x_F, full_matrices=True)
# в лекции V @ D @ U.T
# Вместо U @ D V.T
# Получается в нашем варианте

import numpy as np


class PCA:
    """
    n_components - m = Число конечных компонент - скорее всего задается вне
    D - сингулярные значения - считается внутри
    G - Новая матрица признаков
    Ut_transform - матрица для преобразования признаков

    """

    def fit(X):
        """
        выполняем svd

        PCA не нужены Y для fit, можно сказать обучение без учителя
        """
        F = X
        FT_x_F = F.T @ F
        self.V, self.D, U_T = np.linalg.svd(F, full_matrices=False)
        self.U = U_T.T

        # Можно потом пойти от того, что мы получили D через eig от FT @ F, но пока пускай вычислим целиком svd
        # eigs_FT_x_F = np.sort(np.linalg.eig(FT_x_F)[0].real)[::-1]
        return ...  # ???? хз

    def transform(X):
        """
        Переход в новое пространство признаков G
        """
        F = X
        self.U_effective = self.U[:, :effective_m]
        self.V_effective = self.V[:, :effective_m]

        self.L = self.D[:effective_m]  # BIG LAMBDA = G.T @ G
        self.G = F @ U_effective  # - Матрица новых признаков

        ...

    def get_model_without_targets():
        self.beta = np.diag(1 / L) @ self.V_effective.T
        self.model_pca = self.G @ self.beta # Не хватает только @ y

    def inverse_transform(Z):
        """
        Восстановление признаков обратно
        """
        self.F_restored = self.G @ self.U_effective.T
        ...

    def fit_transform(X):
        """
        Выполняем SVD
        Затем преобразование
        """
        ...

    def calculate_em(D):
        eigs_full_sum = np.sum(D**2)
        Ems = []
        for m in range(1, len(D)):
            Em = np.sum(D[m:] ** 2) / eigs_full_sum
            Ems.append(Em)




# Пускай
effective_m = 3  ### ????? Magic NUMBER???


# print(np.linalg.norm(F - V_effective @ np.diag(L) @ Uh_effective)) # ||F - V @ LAMBDA @ U.T|| -> 0??





# Аналитическое решение через псевдообратную матрицу (Очень просто).
# weights_pinv = np.linalg.pinv(F) @ y  # F^+ @ y
