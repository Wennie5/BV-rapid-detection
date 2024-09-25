import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# Load dataset and split features and labels
df1 = pd.read_csv('D:/Desktop/BV/BV-SERS data/阴性阳性_norm.csv', header=None)
X = np.expand_dims(df1.values[:, 0:1161].astype(float), axis=2)
Y = df1.values[:, -1]

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None)


# Define the model structure
def baseline_model():
    model = Sequential()
    model.add(Conv1D(8, 3, input_shape=(1161, 1)))
    model.add(Conv1D(16, 5, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Train the model
estimator = KerasClassifier(build_fn=baseline_model, epochs=80, batch_size=15, verbose=1, validation_split=0.2)
history = estimator.fit(X_train, Y_train)


# Plot learning curve
def plot_learning_curve(history):
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


# Call the function to plot learning curve
plot_learning_curve(history)


