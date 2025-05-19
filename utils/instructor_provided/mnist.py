## Train a basic network on classic MNIST digits dataset
# See packt.link/dltf Chapter 1 for more examples
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# for reproducibility
np.random.seed(523)

# hyperparameters for network and training 
EPOCHS = 100
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10   # number of outputs = number of digits
N_HIDDEN = 128
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION

# loading MNIST dataset
# verify
# the split between train and test is 60,000, and 10,000 respectly 
# one-hot is automatically applied
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


#normalize in [0,1]
X_train, X_test = X_train / 255.0, X_test / 255.0
#X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
#
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')

model = tf.keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(RESHAPED,)))
model.add(keras.layers.Dense(NB_CLASSES,   		
   		name='dense_layer', activation='softmax'))

# summary of the model
model.summary()

# compiling the model
model.compile(optimizer='SGD', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the model and saving history
history = model.fit(X_train, Y_train,
		batch_size=BATCH_SIZE, epochs=EPOCHS,
		verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# retrieve training history
epochs = history.epoch
training_loss = history.history['loss']
training_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# evalutae the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)

# making prediction
predictions = model.predict(X_test)

# Create plots comparing training and validation metrics
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, training_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('mnist_training_metrics_seed_42.png')
plt.show()