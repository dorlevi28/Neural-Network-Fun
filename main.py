import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential
import matplotlib.image as mpimg


def print_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(True)
    plt.show()


def get_data_and_labels(data,labels,num_of_dir):
    # get data path
    for i in range(num_of_dir):
        path = os.path.join(os.getcwd(), 'train', str(i))
        images = os.listdir(path)
        for a in images:
            try:
                image = Image.open(path + '\\' + a)
                image = image.resize((30, 30))
                image = np.array(image)
                data.append(image)
                # if we are classifying a stop sign, give it a label 1
                if (i == 14):
                    labels.append(1)
                else:
                    labels.append(0)
            except:
                print("Error loading image")
    return data,labels

def get_model(X_train):
    # Building the model, using convenutinal neural network for image classification
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1, activation='sigmoid'))  # last layer of binary classification
    return model

def show_accuarcy(history):
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def excuecte():

    data=[] #list of np arrays, each representing one photo image of the train set
    labels=[]
    num_of_dir=43
    data,labels=get_data_and_labels(data,labels,num_of_dir)
    #Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    #split data to train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    #normalize values for better results
    X_train=X_train/900
    X_test=X_test/900

    #set the model
    model= get_model(X_train)

    #set hyper parametres
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #train the model
    history = model.fit(X_train, Y_train, batch_size=64, epochs=15, validation_data=(X_test, Y_test))

    #plotting graphs for accuracy
    show_accuarcy(history)

    #testing accuracy on test dataset
    y_test = pd.read_csv('Test.csv')
    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data=[]
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))
    X_test = np.array(data)
    X_test=X_test/900
    #predict on test set
    pred = model.predict_classes(X_test)
    print(accuracy_score(labels, pred))


if __name__=="__main__":
    excuecte()







