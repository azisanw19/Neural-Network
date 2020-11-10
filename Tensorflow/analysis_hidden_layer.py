import tensorflow  as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def tensorflow(data, n_hidden_layer):
    (train_images, train_labels), (test_images, test_labels) = data.load_data()


    train_images = train_images/255.0
    test_images = test_images/255.0

    layer = []
    layer.append(keras.layers.Flatten(input_shape=(28,28)))
    for _ in range(n_hidden_layer):
        layer.append(keras.layers.Dense(128, activation='relu'))
    layer.append(keras.layers.Dense(10, activation='softmax'))

    model = keras.Sequential(layer)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('number of hidden layer', len(layer)-2)

    train_hasil = model.fit(train_images, train_labels, epochs=5)

    print('\n\nTested')

    test_hasil = model.evaluate(test_images, test_labels)

    return (train_hasil.history['accuracy'][-1], test_hasil[1])

if __name__ == "__main__":
    # get image data (28x28 pixel)
    data = keras.datasets.fashion_mnist
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_acc = []
    test_acc = []
    length = range(1, 20)

    for i in length:
        r, e = tensorflow(data, i)
        train_acc.append(r)
        test_acc.append(e)

    plt.plot(length, train_acc, length, test_acc)
    plt.title('Analysis number of hidden layer')
    plt.xlabel('Number of hidden layer')
    plt.ylabel('Accuracy')
    plt.show()