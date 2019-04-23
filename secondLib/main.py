from functools import reduce

import numpy as np
from scipy.stats.distributions import chi2
from scipy.stats import t


def parse_file(file_name):
    first_attribute = []
    second_attribute = []
    with open(file_name, 'r') as file:
        for line in file:
            parsed_data = line
            futureInt = ""
            for i in parsed_data:
                if i == '\t':
                    first_attribute.append(float(futureInt))
                    futureInt = ""
                elif i == '\n':
                    second_attribute.append(float(futureInt))
                    futureInt = ""
                else:
                    futureInt += i

    return first_attribute, second_attribute


# ожидаемое значение
def expected_value(variable):
    return reduce((lambda x, y: x + y), variable) / len(variable)


# выборочная дисперсия
def sample_variance(random_variable):
    return expected_value(np.square(random_variable)) - pow(expected_value(random_variable), 2)


# вычисление
def compute_chi2(random_variable, alpha):
    coefficient = 1 if alpha > 0.5 else -1
    denominator = (1 - alpha) if coefficient == 1 else alpha
    d = coefficient * (2.0637 * (np.log(1 / denominator) - 0.16) ** 0.4274 - 1.5774)
    A = d * np.sqrt(2)
    B = 2 / 3 * (d ** 2 - 1)
    C = d * (d ** 2 - 7) / 9 * np.sqrt(2)
    D = -(6 * d ** 4 + 14 * d ** 2 - 32) / 405
    E = d * (9 * d ** 4 + 256 * d ** 2 - 433) / (4860 * np.sqrt(2))
    n = len(random_variable) - 1

    return n + A * np.sqrt(n) + B + C / np.sqrt(n) + D / n + E / (n * np.sqrt(n))


# вычислить переменную границу интервала
def compute_var_interval_border(random_variable, alpha):
    length = len(random_variable)
    quantise_high = compute_chi2(random_variable, alpha / 2)
    quantise_low = chi2.ppf(1 - alpha / 2, df=(length - 1))
    variance = unbiased_sample_variance(random_variable)

    low_border = variance * (length - 1) / quantise_low
    high_border = variance * (length - 1) / quantise_high

    return low_border, high_border


# несмещенная выборочная дисперсия
def unbiased_sample_variance(random_variable):
    n = len(random_variable)

    return sample_variance(random_variable) * n / (n - 1)


# ожидаемое значение доверительного интервала
def expected_value_confidence_interval(random_variable, alpha):
    coefficient = np.abs(t.ppf(alpha / 2, df=(len(random_variable) - 1)))#Нахождение коэфа довеия

    return coefficient * np.sqrt(unbiased_sample_variance(random_variable) / len(random_variable))


# ожидаемые значения равные известной переменной
def are_expected_values_equal_known_var(random_variable1, random_variable2, erf_coeff):
    expected_value1 = expected_value(random_variable1)
    expected_value2 = expected_value(random_variable2)
    variance1 = unbiased_sample_variance(random_variable1)
    variance2 = unbiased_sample_variance(random_variable2)
    length1 = len(random_variable1)
    length2 = len(random_variable2)

    return erf_coeff > np.abs(expected_value1 - expected_value2) / np.sqrt(
        (variance1 / length1) + (variance2 / length2))


# ожидаемые значения равные неизвестной переменной
def are_expected_values_equal_unknown_var(random_variable1, random_variable2, alpha):
    expected_value1 = expected_value(random_variable1)
    expected_value2 = expected_value(random_variable2)
    unbiased_var1 = unbiased_sample_variance(random_variable1)
    unbiased_var2 = unbiased_sample_variance(random_variable2)
    length1 = len(random_variable1)
    length2 = len(random_variable2)

    estimate = np.abs(expected_value1 - expected_value2) / \
               (np.sqrt((length1 - 1) * unbiased_var1 + (length2 - 1) * unbiased_var2)) * \
               np.sqrt(((length1 * length2) * (length1 + length2 - 2)) / (length1 + length2))

    student_caff = np.abs(t.ppf(alpha / 2, df=(length1 + length2 - 2)))

    return student_caff > estimate


def main():
    msg, gvh = parse_file('input.txt')
    method_Mac = list(map(float, msg))
    method_Heine = list(map(float, gvh))

    alpha = 0.05

    expected_method_Mac = expected_value(method_Mac)
    expected_method_Heine = expected_value(method_Heine)

    variance_Method_Mac = sample_variance(method_Mac)
    variance_Method_Heine = sample_variance(method_Heine)

    unbiased_sample_variance_method_Mac = unbiased_sample_variance(method_Mac)
    unbiased_sample_variance_method_Hene = unbiased_sample_variance(method_Heine)

    deviation_Method_Mac = np.sqrt(sample_variance(method_Mac))
    deviation_Method_Heine = np.sqrt(sample_variance(method_Heine))

    print('Ожидаемое значение метода Mac = ', expected_method_Mac)
    print('Ожидаемое значение метода Heine = ', expected_method_Heine)

    print('Смещенная дисперсия средней Mac = ', variance_Method_Mac)
    print('Cмещенная средней Heine = ', variance_Method_Heine)

    print('Несмещенная дисперсия средней Mac = ', unbiased_sample_variance_method_Mac)
    print('Несмещенная дисперсия средней Heine = ', unbiased_sample_variance_method_Hene)

    print('Стандартное отклонение средней Mac = ', deviation_Method_Mac)
    print('Стандартное отклонение средней Heine = ', deviation_Method_Heine)

    # Expected values' confidence intervals computations

    print('Доверительный интервал средней ожидаемой Mac {} < E(x) < {}'.format(
        expected_method_Mac - expected_value_confidence_interval(method_Mac, alpha),
        expected_method_Mac + expected_value_confidence_interval(method_Mac, alpha)
    ))

    print('Доверительный интервал средней ожидаемой Heine: {} < E(x) < {}'.format(
        expected_method_Heine - expected_value_confidence_interval(method_Heine, alpha),
        expected_method_Heine + expected_value_confidence_interval(method_Heine, alpha)
    ))

    # Variances' confidence intervals computations

    low_Mac_border, high_Mac_border = compute_var_interval_border(method_Mac, alpha)
    print('Доверительный интервал средней дисперсии Mac: {} < sigma^2 < {}'.format(
        low_Mac_border,
        high_Mac_border
    ))

    low_Heine_border, high_Heine_border = compute_var_interval_border(method_Heine, alpha)
    print('Доверительный интервал средней дисперсии ветра Heine: {} < sigma^2 < {}'.format(
        low_Heine_border,
        high_Heine_border
    ))

    # Hypothesises' checks

    # 1.65 - Laplace's function's arg with alpha equal to 0.05
    print('Является ли истинной гипотеза, что E(Mac) = E(Heine) с известной дисперсией:',
          are_expected_values_equal_known_var(method_Mac, method_Heine, 1.65))

    print('Является ли истинной гипотеза, что E(Mac) = E(Heine) с не известной дисперсией:',
          are_expected_values_equal_unknown_var(method_Mac, method_Heine, alpha))


if __name__ == '__main__':
    main()
