# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:10:07 2019

@author: ncelik34
"""


# Importing the libraries
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda,floatX=float32"
#
## Now check
from theano import function, config, shared, tensor
#import theano
#from theano.tensor.shared_randomstreams import RandomStreams
import numpy
import time
import random
#import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten, Reshape, Activation
from keras.layers import LSTM
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import LearningRateScheduler
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


#metrics accuracy
from keras import backend as K
def mcor(y_true, y_pred):
     #matthews_correlation
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



import tensorflow as tf

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def make_roc(yy_res,val_target,predict_val):
    
    #roc curve plotting for multiple
    from sklearn.preprocessing import label_binarize
    y = label_binarize(yy_res, classes=[0, 1, 2, 3, 4, 5])
    #y_predict = label_binarize(target_test, classes=[0, 1, 2, 3, 4, 5])
    y_predict = label_binarize(val_target, classes=[0, 1, 2, 3, 4, 5])
    n_classesi = y.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    from sklearn.metrics import roc_curve, auc
    for i in range(n_classesi):
        #fpr[i], tpr[i], thre = roc_curve(y_predict[:, i], predict[:, i])
        fpr[i], tpr[i], thre = roc_curve(y_predict[:, i], predict_val[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    #####
    from itertools import cycle
    plt.figure(2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','black','yellow'])
    for i, color in zip(range(n_classesi), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    #plt.semilogy([0, 1], [0, 1], 'k--', lw=lw)
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Zooom in View: Some extension of ROC to multi-class')
    plt.legend(loc="lower right")
    plt.show()



'''
############# SET UP RUN HERE ####################
'''

batch_size=256
folder= r"P:/Code/MLG channels-Lowery/CED Tests/260219/testtest"


os.chdir(folder)
CurrDir = os.getcwd()
files = os.listdir(folder)


#df29 = pd.read_csv('outshot_5_train3.csv',header=None)
df29 = pd.read_csv('outshot_val_MLG_noise2.csv',header=None)
#df29 = df29.iloc[1:]
#df = df.astype('float64')
data11 = df29.values
dataset=df29.values
dataset = dataset.astype('float64')
timep=np.zeros([len(dataset),])
timep=dataset[:,0]
#maxchannels=10
maxer=np.amax(dataset[:,2])
print (maxer)
maxeri=maxer.astype('int')
maxchannels=maxeri
idataset=np.zeros([len(dataset),],dtype=int)
idataset=dataset[:,2]
idataset=idataset.astype(int)
#categorical_labels = to_categorical(idataset, num_classes=maxchannels+1)
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)
#scaler = preprocessing.StandardScaler()

#dataset = scaler.fit_transform(dataset)


#train and test set split and reshape:
train_size = int(len(dataset) * 0.80) 
modder=math.floor(train_size/batch_size)
train_size =int(modder*batch_size)
test_size = int(len(dataset) - train_size)
modder=math.floor(test_size/batch_size)
test_size =int(modder*batch_size)

print ('training set= ',train_size)
print('test set =', test_size)
print ('total length', test_size+train_size)

in_train, in_test = dataset[0:train_size,1], dataset[train_size:train_size+test_size,1]
#target_train, target_test = categorical_labels[0:train_size,:], categorical_labels[train_size:train_size+test_size,:]
in_train = in_train.reshape(len(in_train),1,1,1)
in_test = in_test.reshape(len(in_test), 1,1,1)
#state=np.argmax(target_test,axis=-1)
#print(state[0:10])                               


x_train = dataset[:,1]
y_train = idataset[:]
x_train=x_train.reshape((len(x_train),1))
y_train=y_train.reshape((len(y_train),1))

#number of classes in the test set:
#intest_det=idataset[train_size:train_size+test_size]



from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='auto', random_state=42, ratio = 'auto')
X_res, Y_res = sm.fit_sample(x_train, y_train.ravel())


#xx_res=np.random.shuffle(X_res)
from sklearn.utils import shuffle
yy_res=Y_res.reshape((len(Y_res),1))
xx_res, yy_res = shuffle(X_res, yy_res)

#print(xx_res[:40,0])   
#print(yy_res[:40,0])       


trainy_size = int(len(xx_res) * 0.80) 
modder=math.floor(trainy_size/batch_size)
trainy_size =int(modder*batch_size)
testy_size = int(len(xx_res) - trainy_size)
modder=math.floor(testy_size/batch_size)
testy_size =int(modder*batch_size)

print ('training set= ',trainy_size)
print('test set =', testy_size)
print ('total length', testy_size+trainy_size)

#categorical_labels = to_categorical(Y_res, num_classes=maxchannels+1)

in_train, in_test = xx_res[0:trainy_size,0], xx_res[trainy_size:trainy_size+testy_size,0]
#target_train, target_test = categorical_labels[0:trainy_size,:], categorical_labels[trainy_size:trainy_size+testy_size,:]
target_train, target_test = yy_res[0:trainy_size,0], yy_res[trainy_size:trainy_size+testy_size,0]
in_train = in_train.reshape(len(in_train),1,1,1)
in_test = in_test.reshape(len(in_test), 1,1,1)


#validation set!!
df31 = pd.read_csv('outfinaltest17_noise.csv',header=None)
#df29 = df29.iloc[1:]
#df = df.astype('float64')
data_val = df31.values
data_val = data_val.astype('float64')

idataset2=np.zeros([len(data_val),],dtype=int)
idataset2=data_val[:,2]
idataset2=idataset2.astype(int)
#scaler2 = preprocessing.StandardScaler()
#data_val = scaler2.fit_transform(data_val)

val_set = data_val[:,1]
val_set = val_set.reshape(len(val_set),1,1,1)
val_target = data_val[:,2]

#state=np.argmax(target_test,axis=-1)
#print(state[0:10])                               
print(target_test[0:10])  


#model starts..

newmodel = Sequential()
timestep=1
input_dim=1
newmodel.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, timestep, input_dim)))
newmodel.add(TimeDistributed(MaxPooling1D(pool_size=1)))
newmodel.add(TimeDistributed(Flatten()))

newmodel.add(LSTM(256, activation='relu', return_sequences=True))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.2))

newmodel.add(LSTM(256, activation='relu', return_sequences=True))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.2))

newmodel.add(LSTM(256, activation='relu'))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.2))

newmodel.add(Dense(maxchannels+1))
newmodel.add(Activation('softmax'))


newmodel.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False), metrics=['accuracy', precision, recall, f1])
#newmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
def step_decay(epoch):
    initial_lrate=0.001
    drop=0.001
    epochs_drop=3.0
    lrate=initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

lrate=LearningRateScheduler(step_decay)    


epochers=2
history=newmodel.fit(x=in_train,y=target_train, initial_epoch=0, epochs=epochers, batch_size=batch_size, callbacks=[lrate], verbose=1, shuffle=False,validation_data=(in_test, target_test))
#predict = model.predict(in_test, batch_size=batch_size)


#prediction for test set
predict = newmodel.predict(in_test, batch_size=batch_size)

#prediction for val set
predict_val = newmodel.predict(val_set, batch_size=batch_size)


#state=np.argmax(target_test,axis=-1)
class_predict=np.argmax(predict,axis=-1)
class_predict_val=np.argmax(predict_val,axis=-1)
#print(state[:20])
print(target_test[:20])
print(class_predict[:20])
print(class_predict_val[:20])


from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(target_test, class_predict)
cm_val = confusion_matrix(idataset2, class_predict_val)

rnd=1
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig(str(rnd)+'acc.png')
plt.show()

#summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(str(rnd)+'loss.png')
plt.show()


plotlen=test_size
lenny=2000
#target_test = dataset[train_size:len(dataset),3]
#target_test = target_test.reshape(plotlen, 1)
plt.figure(figsize=(30,6))
plt.subplot(2,1,1)
#temp=scaler.inverse_transform(dataset)
#plt.plot (temp[train_size:len(dataset),1], color='blue', label="some raw data")
plt.plot (xx_res[trainy_size:trainy_size+lenny,0], color='blue', label="some raw data")
plt.title("The raw test")

plt.subplot(2,1,2)
#plt.plot(target_test.reshape(plotlen,1)*maxchannels, color='black', label="the actual idealisation")
plt.plot(target_test[:lenny], color='black', label="the actual idealisation")
#plt.plot(spredict, color='red', label="predicted idealisation")

line,=plt.plot(class_predict[:lenny], color='red', label="predicted idealisation")
plt.setp(line, linestyle='--')
plt.xlabel('timepoint')
plt.ylabel('current')
#plt.savefig(str(rnd)+'data.png')
plt.legend()
plt.show()


#newmodel.save('nmn_oversampled_deepchanel6_5.h5')

make_roc(yy_res,val_target,predict_val)

#roc curve plotting for single
#from sklearn.preprocessing import label_binarize
#y = label_binarize(yy_res, classes=[0, 1,2])
#y_predict = label_binarize(target_test, classes=[0, 1,2])
#n_classesi = y.shape[1]
#
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#from sklearn.metrics import roc_curve, auc
#for i in range(n_classesi-1):
#    fpr[i], tpr[i], thre = roc_curve(y_predict[:, i], predict[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])



