from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dropout

from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load  dataset
dataset = pd.read_csv('/home_l/francovm/Projects/SSE/data/processed/input_data.csv', sep='\t', encoding='utf-8' ,index_col=0)

# split into input (X) and output (Y) variables

train_X = dataset.drop(columns=['Events'])

#one-hot encode target column
# train_Y = to_categorical(dataset.Events)

#Non categorical data
train_Y = dataset['Events'].values

#get number of columns in training data
n_cols = train_X.shape[1]


# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.33, random_state=seed)

model_2 = Sequential()

#add layers to model
model_2.add(Dense(512, activation='relu', input_shape=(n_cols,)))
model_2.add(Dropout(0.2))
model_2.add(Dense(512, activation='relu'))
model_2.add(Dropout(0.2))
model_2.add(Dense(1, activation='sigmoid'))

# Compile model
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10)

# Fit the model

history = model_2.fit(X_train,y_train,
                      epochs=100,
                      validation_split=0.3,
                      batch_size=64,
                      callbacks=[early_stopping_monitor])


## Plot the model
# from keras.utils import plot_model
# plot_model(model_2, to_file='model.png')

# evaluate the model
# scores = model_2.evaluate(train_X,train_Y)
# print((scores[1]*100))



# list all data in history
print(history.history.keys())

# # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('results1.png')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#
#
#
plt.savefig('results2.png')
plt.show()
#
