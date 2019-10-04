import numpy as np
import os
import random
import argparse
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from preprocess import main
import datetime


def get_train_data(file_directory):
    with open(file_directory, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        label = []
        for row in reader:
            sex_dict = {'male': 0, 'female': 1}
            passenger_id = int(row[0])
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
            new_row = [p_class, sex, age, sibling, parent, fare, embarked]
            data.append(new_row)
            label.append(is_survive)
    # combined = list(zip(data, label))
    # random.shuffle(combined)
    # data[:], label[:] = zip(*combined)
    return np.array(data), label


def get_test_data(file_directory):
    with open(file_directory, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        id = []
        for row in reader:
            sex_dict = {'male': 0, 'female': 1}
            passenger_id = int(row[0])
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
            new_row = [p_class, sex, age, sibling, parent, fare, embarked]
            data.append(new_row)
            id.append(passenger_id)
    return np.array(data), id


def nn_train(data, label, data_test, label_test):
    lr = 0.001
    bs = 32
    epoch = 100

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(15, activation=tf.nn.relu, input_dim=np.shape(data)[1]))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    sgd = tf.keras.optimizers.Adam(lr=lr)
    name = "newver1-%s-lr%s-bs%s-epoch%s" % (datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), lr, bs, epoch)
    logdir = "logs/scalars/" + name
    print(os.path.abspath(logdir))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    train_history = model.fit(data, label, batch_size=bs, nb_epoch=epoch, verbose=1,
                              validation_data=(data_test, label_test), callbacks=[tensorboard_callback])
    print("Average train loss: ", np.average(train_history.history['loss']))
    print("Average train accuracy: ", np.average(train_history.history['acc']))
    return model, name


def get_raw_data():
    print(os.path.abspath("./data/train.csv"))
    train_data, train_label = get_train_data("./data/train.csv")
    test_data, test_id = get_test_data("./data/test.csv")
    print(np.shape(train_data))
    print(np.shape(train_label))
    return train_data, train_label, test_data, test_id


if __name__ == "__main__":
    train_data, train_label, test_data, test_id = main()
    train_label = tf.keras.utils.to_categorical(train_label, 2)
    X_train, X_eval, y_train, y_eval = train_test_split(train_data, train_label, test_size=0.2)
    std_scale = preprocessing.StandardScaler().fit(X_train)
    X_train_norm = std_scale.transform(X_train)
    X_eval_norm = std_scale.transform(X_eval)
    X_test_norm = std_scale.transform(test_data)

    model, name = nn_train(X_train_norm, y_train, X_eval_norm, y_eval)
    # score = model.evaluate(X_test_norm, y_test, verbose=0)
    result = model.predict(X_test_norm)
    print(np.shape(result))
    print(np.argmax(result, 1).tolist())
    # for r, l in zip(result, y_test):
    #     print("Label: %s, Prediction: %s with confidence %s" % (np.argmax(l),np.argmax(r), np.max(r)))

    with open("./data/final_result" + name + ".csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=["PassengerId", "Survived"])
        writer.writeheader()

        for i, r in zip(test_id, np.argmax(result, 1).tolist()):
            writer.writerow({'PassengerId': i, "Survived": r})


