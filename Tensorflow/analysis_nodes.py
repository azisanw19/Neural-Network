import tensorflow  as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def tensorflow(data, n_nodes):
    (train_images, train_labels), (test_images, test_labels) = data.load_data()


    train_images = train_images/255.0
    test_images = test_images/255.0

    layer = []
    layer.append(keras.layers.Flatten(input_shape=(28,28)))
    layer.append(keras.layers.Dense(n_nodes, activation='relu'))
    layer.append(keras.layers.Dense(10, activation='softmax'))

    model = keras.Sequential(layer)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('number of nodes', n_nodes)

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
    node = 100
    for i in range(20):
        r, e = tensorflow(data, node)
        train_acc.append(r)
        test_acc.append(e)
        length.append(node)
        node += 100
        

        
    plt.plot(length, train_acc, length, test_acc)
    plt.title('Analysis number of nodes')
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy')
    plt.show()