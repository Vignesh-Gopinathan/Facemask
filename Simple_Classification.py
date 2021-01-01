# Author : Vignesh Gopinathan
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Train data Set Generation.
train_target = []
train_samples = []

for i in range(500):
  train_samples.append(randint(1,30))
  train_target.append(1) # Young people affected by corona.
  train_samples.append(randint(31,100))
  train_target.append(0) # Old people not affeced by corona.
for i in range(10000):
  train_samples.append(randint(1,30))
  train_target.append(0) # Young people not affected by corona.
  train_samples.append(randint(31,100))
  train_target.append(1) # Old people affeced by corona.

train_target = np.array(train_target)
train_samples = np.array(train_samples)
train_target, train_samples = shuffle(train_target, train_samples)
scaler = MinMaxScaler()
train_samples = scaler.fit_transform(train_samples.reshape(-1,1)) # Rescaling data between 0 and 1.

# Model construction.
model = Sequential([ # Dense means fully connected network.
                    Dense(units = 16, input_shape=(1,), activation='relu'), # First hidden layer or second layer; unit is the neurons
                    Dense(units = 32, activation='relu'),
                    Dense(units = 2, activation = 'softmax') # Output layer
                   ])
model.summary()

# Model compilation.
model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Model Training.
#If validation split option is used, then the validation set is created from training set before shuffling the training. So the training data has to be shuffled before fitting.
model.fit(x = train_samples, y = train_target, validation_split = 0.1, batch_size = 5, epochs = 30, verbose = 2) # Batch is subsample size, epochs is training cycle, verbose is output level(0,1,2)

# Test data Set Generation.
test_target = []
test_samples = []

for i in range(500):
  test_samples.append(randint(1,30))
  test_target.append(1) # Young people affected by corona.
  test_samples.append(randint(31,100))
  test_target.append(0) # Old people not affeced by corona.
for i in range(10000):
  test_samples.append(randint(1,30))
  test_target.append(0) # Young people not affected by corona.
  test_samples.append(randint(31,100))
  test_target.append(1) # Old people affeced by corona.

test_target = np.array(test_target)
test_samples = np.array(test_samples)
test_target, test_samples = shuffle(test_target, test_samples)
scaler = MinMaxScaler()
test_samples = scaler.fit_transform(test_samples.reshape(-1,1)) # Rescaling data between 0 and 1.

#Predictions.
predict = model.predict(x = test_samples, batch_size = 5, verbose = 0) 
rounded_predict = np.argmax(predict, axis = -1)

# Plot function for confusion matrix taken from sklearn site.
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(y_true=test_target, y_pred=rounded_predict)
cm_labels = ['does not have corona', 'has corona']
plot_confusion_matrix(cm = cm,classes = cm_labels, title = 'Confusion Matrix')

