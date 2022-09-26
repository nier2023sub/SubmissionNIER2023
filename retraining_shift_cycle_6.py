from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten,Dropout, MaxPooling2D, AveragePooling2D, Activation
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K

from numpy import genfromtxt
import sys

#set A as true to train and execute CNN A
#set A as false and B as true to train and execute CNN B
#set A as false and B as false to train and execute CNN C
A = False
B = False

rep = sys.argv[1]

train = 1000
val = 500
test = 1000
#test = 70000 - (train+val)

rand_seed = int(sys.argv[2])

path = "Cycle6/rep"+rep

print('Downloading MNIST…')
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
print('Done!')

x = np.concatenate((Xtrain, Xtest))
y = np.concatenate((ytrain, ytest))

Xtrain, Xtemp, ytrain, ytemp = train_test_split(x, y, train_size=train, random_state=rand_seed)

Xval, Xtest, yval, ytest = train_test_split(Xtemp, ytemp, train_size=val, random_state=rand_seed)

Xtest, X_remaining, ytest, y_remaining = train_test_split(Xtest, ytest, train_size=test, random_state=rand_seed)
Xtest, X_remaining, ytest, y_remaining = train_test_split(X_remaining, y_remaining, train_size=test, random_state=rand_seed)
Xtest, X_remaining, ytest, y_remaining = train_test_split(X_remaining, y_remaining, train_size=test, random_state=rand_seed)
Xtest, X_remaining, ytest, y_remaining = train_test_split(X_remaining, y_remaining, train_size=test, random_state=rand_seed)

tot_train_examples = train
tot_test_examples = val
width=28
height=28
channels = 1
f_size1 = 32 
f_size2= 16  

Xtrain_reshaped = Xtrain.reshape(tot_train_examples,width,height,channels)
Xval_reshaped = Xval.reshape(tot_test_examples,width,height,channels)

# #### Inversione labels ####
index1 = np.where(ytest == 2)
index2 = np.where(ytest == 7)
# index1 = np.where(ytrain == 5)
# index2 = np.where(ytrain == 6)


index1 = index1[0][0: int(index1[0].shape[0])]
ytest[index1] = 7
# ytrain[index1] = 6
index2 = index2[0][0: int(index2[0].shape[0])]
ytest[index2] = 2
# ytrain[index2] = 5
# ################################

print('New shape ',Xtrain_reshaped[0].shape)

y_train_cat = to_categorical(ytrain)
y_val_cat = to_categorical(yval)
print(y_train_cat[0])

indexes_df = genfromtxt("Cycle4/rep"+rep+"/sel_indexes.csv",delimiter=',',dtype=int)

print(len(indexes_df))

Xtrain = np.concatenate((Xtrain, Xval))
ytrain = np.concatenate((ytrain, yval))

print(Xtest.shape)
first=True


Xtrain = np.concatenate((Xtrain, Xtest[indexes_df]))
ytrain = np.concatenate((ytrain, ytest[indexes_df]))

tot_train_examples = tot_train_examples + 500

Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, train_size=tot_train_examples, random_state=rand_seed)

Xtrain_reshaped = Xtrain.reshape(tot_train_examples,width,height,channels)
Xval_reshaped = Xval.reshape(tot_test_examples,width,height,channels)

y_train_cat = to_categorical(ytrain)
y_val_cat = to_categorical(yval)

# print(Xtrain.shape)
# print(y_train_cat.shape)

# print("NON VA BENE!!! NON MI TROVO CON GLI OUTPUT")
# print(indexes_df)
# print(Xtest[indexes_df].shape)
# print(ytest[indexes_df].shape)

# print(ytest[indexes_df])

#Cicli post prima mutazione
#Cycle 5
Xtest, X_remaining, ytest, y_remaining = train_test_split(X_remaining, y_remaining, train_size=test, random_state=rand_seed)

# #### Inversione labels ####
index1 = np.where(ytest == 2)
index2 = np.where(ytest == 7)
# index1 = np.where(ytrain == 5)
# index2 = np.where(ytrain == 6)


index1 = index1[0][0: int(index1[0].shape[0])]
ytest[index1] = 7
# ytrain[index1] = 6
index2 = index2[0][0: int(index2[0].shape[0])]
ytest[index2] = 2
# ytrain[index2] = 5
# ################################

indexes_df = genfromtxt("Cycle5/rep"+rep+"/sel_indexes.csv",delimiter=',',dtype=int)

print(len(indexes_df))

Xtrain = np.concatenate((Xtrain, Xval))
ytrain = np.concatenate((ytrain, yval))

print(Xtest.shape)
first=True


Xtrain = np.concatenate((Xtrain, Xtest[indexes_df]))
ytrain = np.concatenate((ytrain, ytest[indexes_df]))

tot_train_examples = tot_train_examples +500

Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, train_size=tot_train_examples, random_state=rand_seed)

Xtrain_reshaped = Xtrain.reshape(tot_train_examples,width,height,channels)
Xval_reshaped = Xval.reshape(tot_test_examples,width,height,channels)

y_train_cat = to_categorical(ytrain)
y_val_cat = to_categorical(yval)

# print(Xtrain.shape)
# print(y_train_cat.shape)

#Cycle 6
Xtest, X_remaining, ytest, y_remaining = train_test_split(X_remaining, y_remaining, train_size=test, random_state=rand_seed)

# #### Inversione labels ####
index1 = np.where(ytest == 2)
index2 = np.where(ytest == 7)
# index1 = np.where(ytrain == 5)
# index2 = np.where(ytrain == 6)


index1 = index1[0][0: int(index1[0].shape[0])]
ytest[index1] = 7
# ytrain[index1] = 6
index2 = index2[0][0: int(index2[0].shape[0])]
ytest[index2] = 2
# ytrain[index2] = 5
# ################################


# model = Sequential()

# model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,1), padding="same"))
# model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
# model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
# model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
# model.add(Flatten())
# model.add(Dense(84, activation='tanh'))
# model.add(Dense(10, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=["accuracy"])

# history = model.fit(Xtrain_reshaped, y_train_cat, validation_data=(Xval_reshaped, y_val_cat), epochs=12,batch_size=128,shuffle=True)

# model.save(path+"/modelC_loop_mutation.h5")

model = load_model(path+"/modelC_loop_mutation.h5")

Xtest_reshaped = Xtest.reshape(test,width,height,channels)
y_test_cat = to_categorical(ytest)

print(model.evaluate(Xval_reshaped,y_val_cat))
print(model.evaluate(Xtest_reshaped,y_test_cat))

classification_v = model.predict_classes(Xval_reshaped)
#print(classification_v)

score_final_v = model.predict(Xval_reshaped)

classification = model.predict_classes(Xtest_reshaped)
#print(classification)

score_final = model.predict(Xtest_reshaped)

trainingset=pd.DataFrame()

for i in range(0, tot_train_examples):
    ma = np.matrix(Xtrain_reshaped[i])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array=pd.DataFrame(ar)
    array.insert(784,'label',ytrain[i])
    trainingset=trainingset.append(array)

trainingset.to_csv(path+'/training.csv', index = False, header = True)
print("training.csv completed")

#validation set csv printing
validationset=pd.DataFrame()

for i in range(0, val):
    ma = np.matrix(Xval_reshaped[i])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array=pd.DataFrame(ar)
    array.insert(784,'label',yval[i])
    array.insert(785,'SUT',classification_v[i])
    array.insert(786,'PredictedLabel0',score_final_v[i][0])
    array.insert(787,'PredictedLabel1',score_final_v[i][1])
    array.insert(788,'PredictedLabel2',score_final_v[i][2])
    array.insert(789,'PredictedLabel3',score_final_v[i][3])
    array.insert(790,'PredictedLabel4',score_final_v[i][4])
    array.insert(791,'PredictedLabel5',score_final_v[i][5])
    array.insert(792,'PredictedLabel6',score_final_v[i][6])
    array.insert(793,'PredictedLabel7',score_final_v[i][7])
    array.insert(794,'PredictedLabel8',score_final_v[i][8])
    array.insert(795,'PredictedLabel9',score_final_v[i][9])
    validationset=validationset.append(array)

validationset.to_csv(path+'/validation.csv', index = False, header = True)

print("validation.csv completed")

#test set csv printing
testset=pd.DataFrame()

for i in range(0, test):
    #print(Xtest_reshaped[i].shape)
    ma = np.matrix(Xtest_reshaped[i])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    # print(res2)
    # print(ytest[0])
    array=pd.DataFrame(ar)
    array.insert(784,'label',ytest[i])
    array.insert(785,'SUT',classification[i])
    array.insert(786,'PredictedLabel0',score_final[i][0])
    array.insert(787,'PredictedLabel1',score_final[i][1])
    array.insert(788,'PredictedLabel2',score_final[i][2])
    array.insert(789,'PredictedLabel3',score_final[i][3])
    array.insert(790,'PredictedLabel4',score_final[i][4])
    array.insert(791,'PredictedLabel5',score_final[i][5])
    array.insert(792,'PredictedLabel6',score_final[i][6])
    array.insert(793,'PredictedLabel7',score_final[i][7])
    array.insert(794,'PredictedLabel8',score_final[i][8])
    array.insert(795,'PredictedLabel9',score_final[i][9])
    
    if(ytest[i] < 5):
        array.insert(796,'EP', 0)
    else:
        array.insert(796,'EP', 1)

    if(ytest[i] in (0, 3, 6, 8, 9)):
        array.insert(797,'FP', 0)
    elif(ytest[i] in (1, 4, 7)):
        array.insert(797,'FP', 1)
    else:
        array.insert(797,'FP', 2)

    testset=testset.append(array)

testset.to_csv(path+'/test.csv', index = False, header = True)
print("test.csv completed")