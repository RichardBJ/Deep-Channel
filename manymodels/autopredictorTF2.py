# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:53:49 2019

@author: project led by RBJ, metrics by ncelik34, further help from STM Ball
"""
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

# Importing the Keras libraries and packages
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, cohen_kappa_score
import math

batch_size = 256

def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def runner(model=None, datafile=None):
	#df30= pd.read_csv(f"C:\\Users\\Richard\\Documents\\GitHub\\Deep-Channel\\Random datasets\\5 channels\\outfinaltest328.csv", header=None)
	df30=pd.read_csv(datafile, header=None)
	dataset = df30.values
	dataset = dataset.astype('float64')
	timep = dataset[:, 0]
	maxer = np.amax(dataset[:, 2])
	print(f"Max states = {maxer}")
	maxeri = maxer.astype('int')
	maxchannels = maxeri
	idataset = dataset[:, 2]
	idataset = idataset.astype(int)

	scaler = MinMaxScaler(feature_range=(0, 1))
	temp = scaler.fit_transform(dataset[:,1].reshape(-1,1))
	dataset[:,0]=temp.reshape(-1,)
	train_size = int(len(dataset))

	in_train = dataset[:, 0]
	target_train = idataset
	in_train = in_train.reshape(len(in_train), 1, 1, 1)

	loaded_model = load_model(model, custom_objects={
							  'mcor': mcor, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc})

	temp=scaler.inverse_transform(dataset[:,1].reshape(-1,1))
	dataset[:,0]=temp.reshape(-1,)
	c = loaded_model.predict(in_train, batch_size=batch_size, verbose=True)
	c=np.argmax(c, axis=-1)
	c=c.reshape(-1,1)
	"""lenny = 2000
	ulenny = 5000"""
	lenny=0
	ulenny=5000
	plot=False
	if plot==True:
		loaded_model.summary()
		plt.figure(figsize=(30, 6))
		plt.subplot(3, 1, 1)
		plt.plot(dataset[lenny:ulenny, 0], color='blue', label="the raw data")
		plt.title("The raw test")
		plt.ylabel('current')

		plt.subplot(3, 1, 2)
		plt.plot(target_train[lenny:ulenny], color='black',
				 label="ground truth")
		plt.ylabel('state')

		plt.subplot(3, 1, 3)
		plt.plot(c[lenny:ulenny], color='red', label="confidence")

		plt.xlabel('timepoint')
		plt.ylabel('p')

		plt.show()

	return cohen_kappa_score(idataset,c)

def main():
	path="c:\\users\\Richard\\Documents\\GitHub\\Deep-Channel\\alldata\\"
	all=glob.glob(path + '**\*',recursive=True)
	files=[]
	for file in all:
		if ".csv" in file:
			if all ("half", "SKM", "biological") not in file:
				files.append(file)
	for file in files:
		print(files)
	exit()
	
	models=glob.glob('*.H5')
	with open("failedexamples.txt","a") as foutput:
		for file in files:
			print(file)
			for model in models:
				output=round(runner(model=model, datafile=file),5)
				print(f"Model {model} gives Kappa = {output}")
				foutput.write(f"\nFile {file}\n   with model {model} gives Kappa = {output}")
				foutput.flush()
		

if __name__ == "__main__":
    c=main()
