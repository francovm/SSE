from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy
import pandas as pd

# fix random seed for reproducibility

numpy.random.seed(7)

# load  dataset

dataset = pd.read_csv('input_data.csv', sep='\t', encoding='utf-8',index_col=0)


# split into input (X) and output (Y) variables

train_X = dataset.drop(columns=['Events'])

#one-hot encode target column
train_Y = to_categorical(dataset.Events)


#get number of columns in training data
n_cols = train_X.shape[1]



# create model
# model = Sequential()

model_2 = Sequential()

# add layers to model

# model.add(Dense(12, input_dim=7, activation='relu'))
#
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu')) # extra layer
# model.add(Dense(10, activation='relu')) # extra layer
# model.add(Dense(10, activation='relu')) # extra layer
# model.add(Dense(2, activation='sigmoid'))

#add layers to model
model_2.add(Dense(250, activation='relu', input_shape=(n_cols,)))
model_2.add(Dense(250, activation='relu'))
model_2.add(Dense(250, activation='relu'))
model_2.add(Dense(2, activation='softmax'))

# Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10)

# Fit the model
# history = model.fit(train_X,train_Y, batch_size=100, epochs=10)

history = model_2.fit(train_X,train_Y, epochs=20, validation_split=0.2, callbacks=[early_stopping_monitor])

# model_2.fit(train_X,train_Y, epochs=10, validation_split=0.2, callbacks=[early_stopping_monitor])


## Plot the model
# from keras.utils import plot_model
# plot_model(model_2, to_file='model.png')

# evaluate the model
# scores = model_2.evaluate(train_X,train_Y)
# print((scores[1]*100))



# _, accuracy = model_2.evaluate(train_X,train_Y)
# print('Accuracy: %.2f' % (accuracy*100))
#
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
