import numpy as np
import os
import random
import argparse
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import datetime


def get_train_data(file_directory):
    with open(file_directory, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        label = []
        for row in reader:
            sex_dict = {'male': 0, 'female': 1}
            passenger_id = float(row[0])
            is_survive = int(row[1])
            p_class = float(row[2])
            sex = sex_dict[row[4]]
            if row[5] == '':
                age = -1
            else:
                age = float(row[5])
            sibling = float(row[6])
            parent = float(row[7])
            if row[9] == '':
                fare = -1
            else:
                fare = float(row[9])
            embarked_dict = {'C': 0, 'Q': 1, 'S': 2, '': 3}
            embarked = embarked_dict[row[11]]
            new_row = [passenger_id, p_class, sex, age, sibling, parent, fare, embarked]
            data.append(new_row)
            label.append(is_survive)
    # combined = list(zip(data, label))
    # random.shuffle(combined)
    # data[:], label[:] = zip(*combined)
    return np.array(data), tf.keras.utils.to_categorical(label,2)


def get_test_data(file_directory):
    with open(file_directory, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            sex_dict = {'male': 0, 'female': 1}
            passenger_id = float(row[0])
            p_class = float(row[1])
            sex = sex_dict[row[3]]
            if row[4] == '':
                age = -1
            else:
                age = float(row[4])
            sibling = float(row[5])
            parent = float(row[6])
            if row[8] == '':
                fare = -1
            else:
                fare = float(row[8])
            embarked_dict = {'C': 0, 'Q': 1, 'S': 2, '': 3}
            embarked = embarked_dict[row[10]]
            new_row = [passenger_id, p_class, sex, age, sibling, parent, fare, embarked]
            data.append(new_row)
    return np.array(data)


def train(data, label, data_test, label_test):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu, input_dim=8))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    sgd = tf.keras.optimizers.Adam(lr=0.001)
    logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    train_history = model.fit(data, label, batch_size=32, nb_epoch=100, verbose=1, validation_data=(data_test, label_test), callbacks=[tensorboard_callback])
    print("Average train loss: ", np.average(train_history.history['loss']))
    print("Average train accuracy: ", np.average(train_history.history['acc']))
    return model

TODO: Create gitignore, check tensorboard
if __name__ == "__main__":
    print(os.path.abspath("./data/train.csv"))
    train_data, train_label = get_train_data("./train.csv")
    test_data = get_test_data("./test.csv")
    print(np.shape(train_data))
    print(np.shape(train_label))
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2)
    std_scale = preprocessing.StandardScaler().fit(X_train)
    X_train_norm = std_scale.transform(X_train)
    X_test_norm = std_scale.transform(X_test)

    model = train(X_train_norm, y_train, X_test_norm, y_test)
    score = model.evaluate(X_test_norm, y_test, verbose=0)
    result = model.predict(X_test_norm)

    # for r, l in zip(result, y_test):
    #     print("Label: %s, Prediction: %s with confidence %s" % (np.argmax(l),np.argmax(r), np.max(r)))


