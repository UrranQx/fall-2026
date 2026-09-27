"""
В рамках лабораторной работы предстоит реализовать линейный классификатор.
И обучить его методом стохастического градиентного спуска с инерцией
с L2 регуляризацией и квадратичной функцией потерь.
"""

# 1. выбрать датасет для классификации, например на [kaggle](https://www.kaggle.com/datasets?&tags=13304-Clustering);
import os

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEARNING_RATE = 0.01
FORGETTING_RATE = 0.3


# Download latest version
path = kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")

print("Path to dataset files:", path)


list_of_files = os.listdir(path)
print(f"List of files in dataset directory: {list_of_files}\n")


# print(f'Path to dataset file: {path_to_zoo}\n')
# print(pd.read_csv(train_path,sep=','))

# print()

# df_train = pd.read_csv(train_path)
# df_test = pd.read_csv(test_path)
#
# print(f"ANY NULL??: {df_train.isna().any().any()}")
# print(df_train.columns[df_train.isna().any()].tolist())
# df_train = df_train.dropna()
# print(df_train.head())
# print(df_train.shape)
#
#
# print(df_train.describe(include="all"))
# print(f"{df_train.info()}\n\n")
#
#
# print(f"ANY NULL??: {df_test.isna().any().any()}")
# print(df_test.columns[df_test.isna().any()].tolist())
#
# df_test = df_test.dropna()
# print(df_test.head())
# print(df_test.shape)
#
#
# print(df_test.describe(include="all"))
# print(f"{df_test.info()}\n\n")

# df_test.drop(["Unnamed: 0", "id"], axis=1, inplace=True)


# Объект - пассажир
# Классы - удовлетворен/ нейтрально или неудовлетворен полетом

# Цель - предсказать класс удовлетворения полетом (satisfied/ neutral or dissatisfied) на основе признаков пассажира.

# Без итогового класса - у нас 24 признка,

# Первый столбец - нам не нужен, так как он лишь обозначает номер записи, и к пассажиру никакого отношения не имеет.
# id - возможно стоит удалить, но ради эксперимента оставим.

# Признаки
# Бинарные: Gender, Customer Type, Type of Travel
# Номинальные:
# Порядковые: Class (Eco = 0, EcoPlus = 1, Business = 2), это как tier подписки
# Количественные:   Age, Flight Distance,
#
#                   Infliflight wifi service, Departure/Arrival time convenient,
#                   Ease of Online booking, Gate location, Food and drink,
#                   Online boarding, Seat comfort, Inflight entertainment,
#                   On-board service, Leg room service, Baggage handling,
#                   Checkin service, Inflight service, Cleanliness,
#
#                   Departure Delay in Minutes, Arrival Delay in Minutes


# Для Arrival Delay in Minutes было выявленно 25893 non-null вхождений - это в тесте,
# Получается, что 25976-25893 = 83 пассажира не указали время задержки прибытия.


def prepare_data(data_path, DROP_NA=True):
    # TODO: Вернуть предобработанные данные в виде pandas df
    # Если планируется применять эту функцию к тестовой и трейн данным, то лучше дополнительно проверить на утечку
    df = pd.read_csv(data_path)
    df.drop(["Unnamed: 0", "id"], axis=1, inplace=True)
    if DROP_NA:
        df.dropna(inplace=True)

    # print(df.head())

    # gender_values = [x for x in df["Gender"].unique()]
    # customer_type_values = [x for x in df["Customer Type"].unique()]
    # type_of_travel_value = [x for x in df["Type of Travel"].unique()]
    # # class_values = [x for x in df["Class"].unique()]
    # satisfaction_values = [x for x in df["satisfaction"].unique()]

    # print(f"Gender values = {gender_values}")
    # print(f"Customer Type values = {customer_type_values}")
    # print(f"Type of Travel values = {type_of_travel_value}")
    # # print(f"Class values = {class_values}")
    # print(f"satisfaction values = {satisfaction_values}")

    rule_satisfied = {
        "satisfied": 1,
        "neutral or dissatisfied": -1,
    }  # 1 and -1 потому что у нас бинарная классификация
    rule_gender = {"Male": 0, "Female": 1}
    rule_customer_type = {"Loyal Customer": 0, "disloyal Customer": 1}
    rule_type_of_travel = {"Personal Travel": 0, "Business travel": 1}
    rule_class = {"Eco": 0, "Eco Plus": 1, "Business": 2}  # Так как их >2, пропишу явно

    df.replace({"satisfaction": rule_satisfied}, inplace=True)
    df.replace({"Gender": rule_gender}, inplace=True)
    df.replace({"Customer Type": rule_customer_type}, inplace=True)
    df.replace({"Type of Travel": rule_type_of_travel}, inplace=True)
    df.replace({"Class": rule_class}, inplace=True)
    # TODO: Добавить свободный член для w0

    return df


def standardize_data(X_train, X_test):
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    # Чтобы не делить на ноль, где например постоянные признаки.
    train_std = np.where(train_std == 0, 1.0, train_std)

    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std

    return X_train_scaled, X_test_scaled, train_mean, train_std


# print(X_train.columns)
# www = np.zeros_like(X_train.columns)
# print(www)
# print(len(www))
# print(len(X_train.columns))
# print(X_train.head())
# print("Train data:\n", X_train)
# print("Test data:\n", X_test)

# print("Train data:\n", y_train)
# print("Test data:\n", y_test)

# return (X_train, y_train), (X_test, y_test)
# 2. реализовать вычисление отступа объекта (визуализировать, проанализировать);

# Для визуализации используем matplotlib, по идее нам надо в 2D разместить точки,
# затем провести линию M_i(w) = g(x_i, w)y_i, используя
# разделяющей функции g(x,w) (=0 для бинарного классификатора)
#


def calculate_margin(X, w, y):
    # Написать функцию, которая будет вычислять отступ объекта.
    # g (x, w) = sign(Sum_{1}^{len(w)} w_j * f_j(x))), но для формулы функцию sign убирают.
    # где f_j(x) - j-итая фича объекста x, а w_j - подобранный (чудом) вес для нее.

    # Отступ для i-го объекта (x_i из множества X) = g(x_i, w) * y_i
    # Веса для фич что для одного объекта, что для другого из X будут одинаковые
    # Если писать циклом:
    # M = []
    # for i in range(len(X)):
    #    # M_i = g(...) * Y[i]
    #    #
    #    # x_i = X[i]
    #    # y_i = y[i]
    #    # summ = 0
    #    # for j in range(len(W)):
    #    #     summ += w[j] * x_i[j]  ## ==> w.T @ x
    #    # M.append(summ * y_i) ## ==> summ = w.T @ X.T ==> summ = X @ w
    # return M

    return (X @ w) * y


def calculate_margin_i(features, w, target):
    # w.T = R^1 x n; features = R^n x 1;
    return (
        w.T @ features
    ) * target  # Получается число - Margin для объекта с фичами X[i], таргетом y[i].


def plot_margin(margin):
    # TODO: Отсортировать значения margin, для нормального красивого графика (или использовать scatter?)
    plt.figure(figsize=(12, 6))
    plt.plot(margin, label="Margin/Отступ")
    plt.plot(np.zeros_like(margin), label="zero level")
    plt.show()


def analyze_margin(margin):
    # Разделим результат на 3 группы качества:
    # 1) (-inf; -eps) - отвратительно
    # 2) [-eps; eps] - было близко
    # 3) (eps; +inf) - хорошо
    eps = 0.5
    confident_miss = np.sum(margin < -eps)
    close = np.sum((margin >= -eps) & (margin <= eps))
    accepted = np.sum(margin > eps)
    n_wrong = np.sum(margin <= 0)
    print(
        f"Существенный промах: {confident_miss}\n"
        f"Почти: {close}\n"
        f"Приемлимый результат: {accepted}\n"
        f"\n"
        f"Доля правильных ответов: {(len(margin) - n_wrong) / len(margin)}\n"
        f"Средний отступ: {margin.mean()}"
    )


# 3. реализовать вычисление градиента функции потерь;
# Это понятно, градиент. (Можно как написать ручками, так и использовать numpy)


def loss_function(margin, func_name="FLD"):
    # По заданию - квадратичная функция потерь - FLD
    return 0.5 * (1 - margin) ** 2


def loss_gradient(sample, weights, target):

    # M[i] = y[i] * (w.T @ x[i])
    # Loss[i] = 1/2 * (1-M[i])**2
    # =>
    # grad(L[i]) = dL[i]/dM[i] * dM[i]/dW
    # (1-M[i])(-1) * y[i] * x[i][k] - частные производные по w[k]
    # Можно вынести (1-M[i])(-1) * y[i] за скобку, и тогда можно посчитать градиент
    # grad(L[i]) = ((M[i] - 1) * y[i]) * x[i]
    # x[i] - вектор длинной с размером фич
    margin = calculate_margin(sample, weights, target)
    return ((margin - 1) * target) * sample


# 4. реализовать рекуррентную оценку функционала качества;
# Q - от слова Quality
def update_quality(old_quality, current_loss):
    return FORGETTING_RATE * current_loss + (1 - FORGETTING_RATE) * old_quality
    # return new_q_value = FORGETTING_RATE * Loss (x_i) + (1 - FORGETTING_RATE) * old_q_value


# 5. реализовать метод стохастического градиентного спуска с инерцией;


def SGD_with_momentum(
    X, y, initial_weights, max_iterations, momentum=0.9, regularization=0.1
):
    weights = initial_weights.copy()
    tolerance = 1e-3

    subset_size = min(
        100, len(X)
    )  # Инициализируем qaulity по случайному подмножеству из 100 элементов, можно больше, можно меньше. Можно поиграться
    subset_indices = np.random.choice(len(X), size=subset_size, replace=False)

    initial_margins = calculate_margin(X[subset_indices], weights, y[subset_indices])

    initial_l2_penalty, _ = l2_penalty_and_gradient(weights, regularization)

    quality = np.mean(loss_function(initial_margins)) + initial_l2_penalty

    quality_history = []  # - Чтобы потом показывать график обучения
    quality_history.append(quality)

    velocity = np.zeros_like(weights)

    # v = gamma * v + (1 - gamma) * grad(Loss(w,x[i])) # - velocity
    # w = w - LEARNING_RATE * v # weights

    for iteration in range(max_iterations):
        index = np.random.randint(
            len(X)
        )  # TODO: Сейчас реализация SGD с возвращеним, но что если надо без?
        sample = X[index]
        target = y[index]
        # Что если для этого вообще надо будет написать отдельный дата лоадер, потому что потом попросят предъявлять объекты в одном порядке, потом в другом и т.п.

        margin = calculate_margin(sample, weights, target)
        current_loss = loss_function(margin)
        gradient = loss_gradient(
            sample, weights, target
        )  # TODO: Можно потом подумать как спользовать момент нестерова
        # gradient = loss_gradient(sample, weights - LEARNING_RATE*momentum*velocity, target)  # Например это будет выглядеть вот так вот.

        # L2 регуляризация
        l2_penalty, l2_gradient = l2_penalty_and_gradient(weights, regularization)
        current_loss += l2_penalty
        gradient += l2_gradient  # w0 тоже тогда попало под регуляризацию, если оно есть. Чтобы поменять - regularization * weights

        velocity = momentum * velocity + (1 - momentum) * gradient
        weights = weights - LEARNING_RATE * velocity
        quality = update_quality(quality, current_loss)
        quality_history.append(quality)
        # if abs(quality_history[-1] - quality[-2]) < tolerance:
        #     break

    return weights, quality_history


# 6. реализовать L2 регуляризацию;
# AKA Tikhonov regularization or ridge regression
def l2_penalty_and_gradient(weights, regularization):
    # regularization - коэфф регуляризации
    #
    weights_regularized = weights.copy()
    weights_regularized[0] = (
        0  # Так как для w0 мы регуляризацию не делаем. # TODO: А может сделать?
    )

    penalty = regularization / 2 * np.sum(weights_regularized**2)
    gradient = regularization * weights_regularized

    return penalty, gradient
    # tau = reg_coeff
    # loss_modified = Loss(w, x[i]) + tau/2 * sum(w_j**2)
    # grad_loss_modified = grad_loss(w, x[i]) + tau * w
    # =>
    # w = w * (1 - LEARNING_RATE*tau) - LEARNING_RATE * grad_loss(w, x[i])
    # pass


# 7. реализовать скорейший градиентный спуск;
def optimal_step(sample):
    # Это работает только когда у нас квадратичная потеря - MSE
    squared_norm = sample @ sample

    if squared_norm == 0:
        return 0.0
    return 1.0 / squared_norm


def stochastic_steepest_gradient_descent(
    X,
    y,
    initial_weights,
    max_iterations,
):
    # TODO: все тоже самое, только LEARNING_RATE = optimal_step(sample)
    weights = initial_weights.copy()

    subset_size = min(100, len(X))
    subset_indices = np.random.choice(
        len(X),
        size=subset_size,
        replace=False,
    )

    initial_margins = calculate_margin(
        X[subset_indices],
        weights,
        y[subset_indices],
    )

    quality = np.mean(loss_function(initial_margins))
    quality_history = [quality]

    for iteration in range(max_iterations):
        index = np.random.randint(len(X))
        sample = X[index]
        target = y[index]

        margin = calculate_margin(
            sample,
            weights,
            target,
        )
        current_loss = loss_function(margin)

        gradient = loss_gradient(
            sample,
            weights,
            target,
        )

        learning_rate = optimal_step(sample)

        weights = (
            weights
            - learning_rate * gradient
        )

        quality = update_quality(
            quality,
            current_loss,
        )
        quality_history.append(quality)

    return weights, quality_history


def batch_steepest_gradient_descent(
    X,
    y,
    initial_weights,
    max_iterations,
    tolerance=1e-8,
):
    weights = initial_weights.copy()
    n_samples = len(X)

    # Для квадратичной функции гессиан постоянен,
    # поэтому достаточно вычислить его один раз.
    hessian = (X.T @ X) / n_samples

    initial_margins = calculate_margin(
        X,
        weights,
        y,
    )
    initial_quality = np.mean(
        loss_function(initial_margins)
    )

    quality_history = [initial_quality]
    step_history = []

    for iteration in range(max_iterations):
        predictions = X @ weights
        residuals = predictions - y

        gradient = (
            X.T @ residuals
        ) / n_samples

        gradient_squared_norm = gradient @ gradient

        # Если градиент практически нулевой,
        # мы уже находимся рядом с минимумом.
        if gradient_squared_norm <= tolerance**2:
            break

        hessian_times_gradient = hessian @ gradient
        denominator = gradient @ hessian_times_gradient

        # Защита от деления на ноль из-за
        # вырожденности или численных погрешностей.
        if denominator <= np.finfo(float).eps:
            break

        learning_rate = (
            gradient_squared_norm
            / denominator
        )

        weights = (
            weights
            - learning_rate * gradient
        )

        margins = calculate_margin(
            X,
            weights,
            y,
        )
        quality = np.mean(
            loss_function(margins)
        )

        quality_history.append(quality)
        step_history.append(learning_rate)

    return weights, quality_history, step_history

# 8. реализовать предъявление объектов по модулю отступа;
# То есть чаще брать объекты, на которых уверенность меньше:
# чем меньше |Mi |, тем больше вероятность взять объект;

# Это получается, надо сначала как-то прогнать модель, и получить результаты по каждому объекту
# Затем высчитать для каждого из них |Mi| (т.е. Отступ для i-го объекта)
# Как-то их ранжировать или придумать, как брать именно те объекты, на которых малейший отступ.
# Т.е. если |M_i| -> 0, то Вероятность взять i-й объект -> 1;


def init_weights(X, y):
    # Реализовать инициализацию весов через корреляцию
    # X - feature matrix; Т.е. из нее мы можем достать например столбец j-го признака
    # y - targets.
    # weights - вектор весов
    #
    #
    # Также, если Функция потерь квадратична, а признаки некоррелированы <f_k, f_j> = 0, j!=k,
    # то мы уже инициализировались итоговыми весами
    #
    y = y.to_numpy()
    weights = np.zeros(
        X.shape[1]
    )  # TODO: Сначала массив можно проинициализировать нулями, и он должен быть размером
    for j in range(len(weights)):
        f_j = X.iloc[:, j].to_numpy()  # Feature column - j
        weights[j] = (y @ f_j) / (f_j @ f_j)

    return weights


def init_weights_v2(X, y):
    # По сути, если правильно расписать линал в оригинальной функции, можно понять, что ее можно записать куда короче
    # используя уже встроенный функционал numpy
    X = np.asarray(X)
    y = np.asarray(y)

    return (X.T @ y) / np.sum(X**2, axis=0)


# 9. обучить линейный классификатор на выбранном датасете;
#    1. обучить с инициализацией весов через корреляцию;
#    2. обучить со случайной инициализацией весов через мультистарт;
#    3. обучить со случайным предъявлением и с п.8;


# 10. оценить качество классификации;


# TODO: METRICS: Precision, Accuracy, Recall, F norm; Confusion Matrix;
def precision(model, target):
    return 0


def MeanSquareError(model, target):
    return np.mean((target - model) ** 2)


# 11. сравнить лучшую реализацию с эталонной; эталлонная - из готовых библиотек


# 12. подготовить отчет.
# TODO: Написать ридми


test_path = os.path.join(path, list_of_files[0])
train_path = os.path.join(path, list_of_files[1])

train_data = prepare_data(train_path)
test_data = prepare_data(test_path)

# print("Train data:\n", train_data)
# print("Test data:\n", test_data)


X_train = train_data.drop(["satisfaction"], axis=1)
y_train = train_data["satisfaction"]

X_test = test_data.drop(["satisfaction"], axis=1)
y_test = test_data["satisfaction"]

X_train = X_train.to_numpy(dtype=float)
y_train = y_train.to_numpy(dtype=float)

X_test = X_test.to_numpy(dtype=float)
y_test = y_test.to_numpy(dtype=float)

X_train, X_test, feature_mean, feature_std = standardize_data(
    X_train,
    X_test,
)
# И еще добавим дополнительный столбец, чтобы был параметр w_0
X_train = np.column_stack(
    [
        np.ones(X_train.shape[0]),
        X_train,
    ]
)

X_test = np.column_stack(
    [
        np.ones(X_test.shape[0]),
        X_test,
    ]
)


weights = init_weights_v2(X_train, y_train)

max_iterations=3 * len(X_train)

trained_weights, quality_history = SGD_with_momentum(
    X_train,
    y_train,
    weights,
    max_iterations=max_iterations,
)

# Наискорейший sgd
steepest_weights, steepest_quality_history = (
    stochastic_steepest_gradient_descent(
        X_train,
        y_train,
        weights,
        max_iterations=max_iterations,
    )
)

# print(steepest_quality_history[-1])

batch_weights, batch_quality_history, batch_step_history = batch_steepest_gradient_descent(
    X_train,
    y_train,
    weights,
    max_iterations=100,
)

print("Iterations:", len(batch_step_history))
print("Initial Q:", batch_quality_history[0])
print("Final Q:", batch_quality_history[-1])
print("Last step:", batch_step_history[-1])

quality_differences = np.diff(batch_quality_history)

print(
    "Quality is non-increasing:",
    np.all(quality_differences <= 1e-12),
)
# # Пустой тест без обучений, просто инициализированных весов.
# random_index = np.random.randint(0,len(X_train))
# ri = random_index
# sample_0 = X_train.iloc[ri].to_numpy()
# target_0 = y_train.iloc[ri]

# print(f"error = {calculate_margin_i(sample_0, weights, target_0)}")
# print(f"sample = {sample_0}\n"
#       f"target = {target_0}")
# # l x 1 ; 1xl
# output = np.sign(weights.T @ sample_0)
# print(f"output = {output}")
# # print(MeanSquareError())
