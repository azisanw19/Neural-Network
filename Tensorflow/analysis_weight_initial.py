import tensorflow  as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def tensorflow(data, type_init):
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    initial = None

    train_images = train_images/255.0
    test_images = test_images/255.0

    if type_init == 'random_normal':
        initial = keras.initializers.RandomNormal(mean=0.5)
    elif type_init == 'random_uniform':
        initial = keras.initializers.RandomUniform(minval=0., maxval=1.0)
    elif type_init == 'zeros':
        initial = keras.initializers.zeros()
    elif type_init == 'ones':
        initial = keras.initializers.ones()

    layer = []
    layer.append(keras.layers.Flatten(input_shape=(28,28)))
    
    layer.append(keras.layers.Dense(128, activation='relu', kernel_initializer=initial, bias_initializer=initial))
    layer.append(keras.layers.Dense(10, activation='softmax', kernel_initializer=initial, bias_initializer=initial))

    model = keras.Sequential(layer)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('type weight initialization: ', type_init)

    train_hasil = model.fit(train_images, train_labels, epochs=5)

    print('\n\nTested')

    test_hasil = model.evaluate(test_images, test_labels)

    return (train_hasil.history['accuracy'][-1], test_hasil[1])

if __name__ == "__main__":
    # get image data (28x28 pixel)
    data = keras.datasets.fashion_mnist
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    tensorflow(data, 'radnom_normal')
    tensorflow(data, 'random_uniform')
    tensorflow(data, 'zeros')
    tensorflow(data, 'ones')