import tensorflow  as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def tensorflow(data, learning_rate=0.01):
    (train_images, train_labels), (test_images, test_labels) = data.load_data()


    train_images = train_images/255.0
    test_images = test_images/255.0

    layer = []
    layer.append(keras.layers.Flatten(input_shape=(28,28)))
    layer.append(keras.layers.Dense(128, activation='relu'))
    layer.append(keras.layers.Dense(10, activation='softmax'))

    model = keras.Sequential(layer)

    l_rate = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=l_rate, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('learning rate: ', learning_rate)

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
    length = []
    
    l_rate = 0
    for i in range(1, 100):
        r, e = tensorflow(data, l_rate)
        train_acc.append(r)
        test_acc.append(e)
        length.append(l_rate)
        l_rate += 0.01
    

    plt.plot(length, train_acc, length, test_acc)
    plt.title('Analysis learning rate')
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.show()