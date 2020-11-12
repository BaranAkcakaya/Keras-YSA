# -*- coding: utf-8 -*-
# MNIST VERİ KÜMESİNİ KULLANARAK YAPAY SİNİR AĞLARI İLE RAKAM SINIFLANDIRMA
## MNIST VERİ KÜMESİNİ YÜKLEMEK
"""

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.shape

len(train_images)

train_labels

test_images.shape

test_labels

"""## Yapay Sinir Ağı Mimarisi"""

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

"""## YSA Modelinin Derlenmesi"""

network.compile(optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

"""## Girişlerin ve etiketlerin hazırlanması"""

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

"""## YSA Modelinin Eğitilmesi"""

network.fit(train_images, train_labels, epochs = 5, batch_size = 128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_loss:', test_loss)

print('test_acc:', test_acc)

