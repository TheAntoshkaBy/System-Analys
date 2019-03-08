import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter

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
                    futureInt+=i

    return first_attribute,second_attribute

def normalize_weight_map(weight_map, data_size):
    for key, value in weight_map.items():
        weight_map[key] = value / data_size
    return weight_map

def variance(random_variable):
    return expected_value(np.square(random_variable)) - pow(expected_value(random_variable), 2)

def expected_value(random_variable):
    # computing weights of random variable
    random_variable_weights_map = normalize_weight_map(Counter(random_variable), len(random_variable))

    # decompose values and their weights
    random_variable_values = list(random_variable_weights_map.keys())
    random_variable_weights = list(random_variable_weights_map.values())

    return np.average(random_variable_values, weights=random_variable_weights)


def main():
    first_attribute, second_attribute = parse_file('input.txt')

    #first_attribute = list(map(float,first_attribute))
    #second_attribute = list(map(float,second_attribute))
    #_, arg2, arg3 = tuple

    # converting data to 2-dimensional arrays in order to work with them
    # because LinearRegression() works with tables represented by matrices
    matr_first_attribute = np.array(list(zip(first_attribute)))
    matr_sec_attribute   = np.array(list(zip(second_attribute)))

    # setting up regression model
    regression = LinearRegression()
    # Train the model, using given sets
    regression.fit(matr_first_attribute,matr_sec_attribute)

    # Make predictions using the testing set
    prediction_first_arg = regression.predict(matr_sec_attribute)

    print('Expected value of average wind speed = ', expected_value(first_attribute))
    print('Expected value of average temperature = ', expected_value(second_attribute))

    print('Variance of average wind speed = ', variance(first_attribute))
    print('Variance of average temperature = ', variance(second_attribute))

    print('NUMPY Variance of average wind speed = ', np.var(first_attribute))
    print('NUMPY Variance of average temperature = ', np.var(second_attribute))

    print('Standard deviation of average wind speed = ', np.sqrt(variance(first_attribute)))
    print('Standard deviation of average temperature = ', np.sqrt(variance(second_attribute)))

    print('Correlation coefficient = ', np.corrcoef(first_attribute, second_attribute)[0, 1])


    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.set_xlabel('Average wind speed')
    ax.set_ylabel('Average temperature')
    plt.plot(first_attribute, second_attribute, color='b', marker='2', linestyle='')

    # the way to draw regression model manually
    # plt.plot(wind_speed, regression.intercept_ + regression.coef_[0] * wind_speed, color='r')
    # or with the help of built-in prediction mechanism
    plt.plot(matr_sec_attribute,prediction_first_arg, color='r')

    plt.show()

if __name__ == '__main__':
    main()