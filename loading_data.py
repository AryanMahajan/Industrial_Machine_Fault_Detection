import numpy as np
import os
import cv2
from tqdm import tqdm
import random

CATEGORIES= ["def_front", "ok_front"]

training_data = []
test_data = []

IMG_SIZE = 50

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(r"data\train",category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
    random.shuffle(training_data)

    X_train = []
    y_train = []

    for features,label in training_data:
        X_train.append(features)
        y_train.append(label)

    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = np.array(y_train)

    return X_train, y_train


def create_test_data():
    print("CREATING TEST DATA")
    for category in CATEGORIES:
        path = os.path.join(r"data\test",category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])  # add this to our test_data
            except Exception as e:
                pass
    
    random.shuffle(test_data)
            
    X_test = []
    y_test = []

    for features,label in test_data:
        X_test.append(features)
        y_test.append(label)

    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = np.array(y_test)

    return X_test,y_test