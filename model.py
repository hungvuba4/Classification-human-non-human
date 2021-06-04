import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from layer import Layer
from neural_network import NeuralNetwork
from accuracy import get_accu
from sys import path
path.append('..')


def main():
    path_human = 'C:/Users/LENOVO/Desktop/AI/Computer vision/AI + CV K62 63/Task7/data/human'
    part_non_human = 'C:/Users/LENOVO/Desktop/AI/Computer vision/AI + CV K62 63/Task7/data/non_human'
    ones_human = np.ones((64 * 64, 1)).reshape(1, -1)
    print(ones_human)
    for file in os.listdir(path_human):
        path_img_human = path_human + '/' + file
        img_human = cv.imread(path_img_human, 0)
        img_human = cv.resize(img_human, (64, 64)).reshape(1, -1)
        ones_human = np.vstack((ones_human, img_human))

    data_human = np.delete(ones_human, 0, 0)
    N, d = data_human.shape
    print(N, d)
    ones_non_human = np.ones((64 * 64, 1)).reshape(1, -1)
    for file in os.listdir(part_non_human):
        path_img_non_human = part_non_human + '/' + file
        img_non_human = cv.imread(path_img_non_human, 0)
        img_non_human = cv.resize(img_non_human, (64, 64)).reshape(1, -1)
        ones_non_human = np.vstack((ones_non_human, img_non_human))
    data_non_human = np.delete(ones_non_human, 0, 0)

    data = np.vstack((data_human, data_non_human))

    label_ones = np.ones((120, 1))
    label_zeros = np.zeros((120, 1))
    label = np.vstack((label_ones, label_zeros))

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    model = NeuralNetwork(X_train, y_train)
    model.add_layer(Layer(54, act_func='relu'))
    model.add_layer(Layer(24, act_func='sigmoid'))
    model.fit(it=444, lr=0.0004)

    y_train_pred = model.pred(X_train)
    y_test_pred = model.pred(X_test)
    # print(y_train_pred)
    # print(y_test_pred)
    # print(f'Train accuracy: {accuracy_score(y_train, y_train_pred) * 100}%')
    print(f'Train accuracy: {round(get_accu(y_train, y_train_pred))}%')
    # print(f'Test accuracy: {accuracy_score(y_test, y_test_pred) * 100}%')
    print(f'Test accuracy: {round(get_accu(y_test, y_test_pred))}%')


if __name__ == '__main__':
    main()
