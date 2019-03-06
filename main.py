import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter

def parse_file(file_name):
    attribute = []
    with open(file_name, 'r') as file:
        for line in file:
            parsed_data = line.split('.')
            print(parsed_data)


    return attribute



def main():
    attribute = parse_file('input.txt')


if __name__ == '__main__':
    main()