# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:53:49 2019

@author: ncelik34
"""
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import os
import sys

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
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
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

def binarize(probs):
	print("binarizing the data to just two states")
	maxstates=10
	limit2=True
	
	states=list(np.argmax(probs,axis=1))
	
	freqs=[]
	for i in range(maxstates):
		freqs.append(states.count(i))
	level0=np.argmax(freqs)
	
	freqs[level0]=0
	level1=np.argmax(freqs)
	
	if level0>level1:
		tmp=level0
		level0=level1
		level1=tmp
	
	binarized=[]
	
	for row in probs:
		if np.argmax(row)<=level0:
			binarized.append(0)                    
		else:
			binarized.append(1)
	return binarized


def main(dataset, mymodel, limit2):
	dataset = np.asarray(dataset).astype('float64')
	timep = dataset[:, 0]
	"""
	maxer = np.amax(dataset[:, 2])
	print(maxer)
	maxeri = maxer.astype('int')
	maxchannels = maxeri
	idataset = dataset[:, 2]
	idataset = idataset.astype(int)"""

	scaler = MinMaxScaler(feature_range=(0, 1))
	temp = scaler.fit_transform(dataset[:,1].reshape(-1,1))
	dataset[:,1]=temp.reshape(-1,)
	train_size = int(len(dataset))

	in_train = dataset[:,1]
	"""target_train = idataset"""
	in_train = in_train.reshape(len(in_train), 1, 1, 1)

	loaded_model = load_model(mymodel, custom_objects={
							  'mcor': mcor, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc})

	"""for debugging, save in a different form to see if this works better with ML"""
	loaded_model.save('MLmodel/my_model')
	temp=scaler.inverse_transform(dataset[:,1].reshape(-1,1))
	dataset[:,1]=temp.reshape(-1,)
	c = loaded_model.predict(in_train, batch_size=batch_size, verbose=True)
	if limit2==True:
		c=np.asarray(binarize(c)).reshape(-1,1)
	else:
		c=np.argmax(c, axis=-1)
		c=c.reshape(-1,1)
	print (f"dataset shape = {dataset.shape}")
	print (f"c shape = {c.shape}")
	output=np.concatenate((dataset[:,0:2],c),axis=1)
	
	return output

if __name__ == "__main__":
	c=main(data, mymodel, limit2)
