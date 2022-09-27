from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten,Dropout, MaxPooling2D, AveragePooling2D, Activation
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K
import sys

rep = sys.argv[1]

path = "Cycle1/rep"+rep

train = 1000
val = 500
test = 1000
#test = 70000 - (train+val)

rand_seed = int(sys.argv[2])

(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()

x = np.concatenate((Xtrain, Xtest))
y = np.concatenate((ytrain, ytest))

Xtrain, Xtemp, ytrain, ytemp = train_test_split(x, y, train_size=train, random_state=rand_seed)

Xval, Xtest, yval, ytest = train_test_split(Xtemp, ytemp, train_size=val, random_state=rand_seed)

#Cycle 1
Xtest, X_remaining, ytest, y_remaining = train_test_split(Xtest, ytest, train_size=test, random_state=rand_seed)
#Xtest, X_remaining, ytest, y_remaining = train_test_split(X_remaining, y_remaining, train_size=test, random_state=rand_seed)
# Xtest, X_remaining, ytest, y_remaining = train_test_split(X_remaining, y_remaining, train_size=test, random_state=rand_seed)
# Xtest, X_remaining, ytest, y_remaining = train_test_split(X_remaining, y_remaining, train_size=test, random_state=rand_seed)
# Xtest, X_remaining, ytest, y_remaining = train_test_split(X_remaining, y_remaining, train_size=test, random_state=rand_seed)
# Xtest, X_remaining, ytest, y_remaining = train_test_split(X_remaining, y_remaining, train_size=test, random_state=rand_seed)

tot_train_examples = train
tot_test_examples = val
width=28
height=28
channels = 1
f_size1 = 32 
f_size2= 16  

Xtrain_reshaped = Xtrain.reshape(tot_train_examples,width,height,channels)
Xval_reshaped = Xval.reshape(tot_test_examples,width,height,channels)

# # #### Inversione labels ####
# index1 = np.where(ytest == 2)
# index2 = np.where(ytest == 7)
# # index1 = np.where(ytrain == 5)
# # index2 = np.where(ytrain == 6)


# index1 = index1[0][0: int(index1[0].shape[0])]
# ytest[index1] = 7
# # ytrain[index1] = 6
# index2 = index2[0][0: int(index2[0].shape[0])]
# ytest[index2] = 2
# # ytrain[index2] = 5
# # ################################

print('New shape ',Xtrain_reshaped[0].shape)

y_train_cat = to_categorical(ytrain)
y_val_cat = to_categorical(yval)
print(y_train_cat[0])

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
# history = model.fit(Xtrain_reshaped, y_train_cat, validation_data=(Xval_reshaped, y_val_cat), epochs=10,batch_size=128,shuffle=True)
# model.save(path+"/modelC_loop.h5")

model = load_model("Cycle1/rep"+rep+"/modelC_loop.h5")

Xtest_reshaped = Xtest.reshape(test,width,height,channels)
y_test_cat = to_categorical(ytest)

#print(Xtest_reshaped.shape)

classification_v = model.predict_classes(Xval_reshaped)
#print(classification_v)

score_final_v = model.predict(Xval_reshaped)

classification = model.predict_classes(Xtest_reshaped)
#print(classification)

print(model.evaluate(Xval_reshaped,y_val_cat))
print(model.evaluate(Xtest_reshaped,y_test_cat))



score_final = model.predict(Xtest_reshaped)

# print(score_final)

get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[3].output])
layer_output = get_3rd_layer_output(np.expand_dims(Xtest_reshaped[0], axis=0))[0]

#print(layer_output.shape)

#training set csv printing
trainingset=pd.DataFrame()

for i in range(0, train):
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