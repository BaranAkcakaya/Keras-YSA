"""### Kurulum ve Kontroller"""

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

"""**Drive da dosya konumlandırmayı yapma işlemleri**"""

!mkdir -p drive
!google-drive-ocamlfuse drive
!ls

!ls drive

import os 
os.chdir("/content/drive/My Drive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/")
!pwd

!ls

!pip install -q keras

"""### Uygulama Başlangıç"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras. layers import Conv2D, MaxPooling2D
from keras import backend as K

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""### Veriden örnekleri görselleştirme"""

plt.figure(figsize=(14,14))
x, y = 10, 4 
for i in range(40):  
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i])
plt.show()

batch_size = 128 
num_classes = 10 
epochs = 6 # 12 epoch önerilir

img_rows, img_cols = 28, 28


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""## MODEL OLUŞTURMA"""

model = Sequential()

"""**Katlanların oluşturulması**"""

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

 
model.add(Conv2D(64, (3, 3), activation='relu'))


model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Dropout(0.25))


model.add(Flatten())


model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))


model.add(Dense(num_classes, activation='softmax'))

"""Modell Görselleştirme"""

model.summary()

model.compile(loss = keras.losses.categorical_crossentropy,
             optimizer = keras.optimizers.Adadelta(),
             metrics = ['accuracy'])

"""### Eğitim İşlemleri"""

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save('save_models/mnist_model.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

"""**Rastgele değer için test işlemi**"""

model_test = model.save('save_models/mnist_model.h5')

test_image = x_test[32]
y_test[32]

plt.imshow(test_image.reshape(28,28))

test_data = x_test[32].reshape(1,28,28,1)
pre=model_test.predict(test_data, batch_size=1)