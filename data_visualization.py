from main import history, saving_history

import pickle
import matplotlib.pyplot as plt
import os

#checking history is saved
if os.path.exists('history.pkl'):
    history = pickle.load(open('history.pkl',"rb"))
else:
    saving_history(history=history)
    history = pickle.load(open('history.pkl',"rb"))

#Accuracy Visualization

