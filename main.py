from training_model import training_model
from loading_data import create_test_data

import tensorflow as tf
import os
from datetime import datetime, date
import time
import csv

#Checking if the model exists or not
if os.path.exists("model.keras"):
    print("Model exists")
    model = tf.keras.models.load_model("model.keras")
else:
    print("Training Model")
    training_model()
    model = tf.keras.models.load_model("model.keras")

#importing test data
x_test,y_test = create_test_data()

#predictions

#normalizing x_test
x_test = x_test.reshape(-1, 50, 50, 1)

# Printing th accuracy of the model with built in function of tensorflow
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Letting the model to predict on Test data
predictions = model.predict(x_test)

model_prediction = []

#appending the predictions into a new list
for prediction in predictions:
    model_prediction.append(round(prediction[0]))

# Making our own functino to find the accuracy of the model which gives same output to the 5th to 6th decimal place as the built in function
def calculate_model_accuracy(model_prediction,y_test):
    correct = 0
    for i in range (0, len(model_prediction)):
        if model_prediction[i] == y_test[i]:
            correct += 1

    return (correct/(len(model_prediction)))*100



print(f"Accuracy of this model: {calculate_model_accuracy(model_prediction, y_test)}%")

def create_logs(model_prediction, y_test):
    now = datetime.now()
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    current_date = date.today()
    # Create a folder to store the logs
    if not os.path.exists(f'logs\{str(current_date)}_{str(current_time).replace(":","-")}'):
        os.makedirs(f'logs\{str(current_date)}_{str(current_time).replace(":","-")}')
        pushing_in_logs(model_prediction,y_test,f'logs\{str(current_date)}_{str(current_time).replace(":","-")}')
    else:
        pushing_in_logs(model_prediction,y_test,f'logs\{str(current_date)}_{str(current_time).replace(":","-")}')


def pushing_in_logs(model_prediciton, y_test, path):
    model_prediciton_path = f'{path}\model_prediction.csv'
    y_test_path = f'{path}\y_test.csv'
    accuracy_path = f'{path}\{"accuracy.csv"}'

    with open(model_prediciton_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(model_prediciton)

    with open(y_test_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(y_test)
        
    with open(accuracy_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([calculate_model_accuracy(model_prediciton, y_test)])

print("CREATING LOGS!!!")
create_logs(model_prediction, y_test)
print("EXITING!!!!")