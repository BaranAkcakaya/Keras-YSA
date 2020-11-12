from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras. layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow  as tf
from keras.layers import *
from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt

"""## Veri Setinin İndirilmesi
 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""### Yapılandırma Ayarları
Küme boyutu, sınıf sayısı, eğitim epoch sayısı gibi parametreleri tüm optimizasyon denemeleri için aynı şekilde ayarlıyoruz!
"""

batch_size = 128 # Küme Boyutu
num_classes = 10 # Sınıf Sayısı
epochs = 20 # Eğitimin epoch sayısı
w_l2 = 1e-5 # Başlangıç

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# sınıf vektörlerini ikili sınıf matrislerine dönüştürmek
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

from keras import optimizers

"""### Tüm optimizasyon yöntemlerini eğiteceğimiz evrişimli sinir ağı modelinin oluşturulması"""

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),  kernel_regularizer=regularizers.l2(w_l2),
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3),  kernel_regularizer=regularizers.l2(w_l2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, kernel_regularizer=regularizers.l2(w_l2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

"""### STOKASTİK GRADYAN/BAYIR İNİŞ OPTİMİZASYONU

Tüm örneklerden geçmek yerine, Stokastik Degrade İniş (SGD), her bir örnekte $(x ^ i, y ^ i)$  parametrelerin güncellenmesini gerçekleştirir. Bu nedenle, öğrenme her örnekte gerçekleşir:

$w = w− α∇wJ(x^i,y^i;w,b)$

```
for i in range(num_epochs):
    np.random.shuffle(data)
    for example in data:
        grad = compute_gradient(example, params)
        params = params - learning_rate * grad
```
"""

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.summary()

"""### Modelin Eğitilm ve Test Sonuçları"""

hist_SGD=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[reduce_lr])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""### ADAM OPTİMİZASYONU
***Adam veya adaptif momentum AdaDelta’ya benzer bir algoritmadır. AdaDelta’dan farklı olarak parametrelerin her birinin öğrenme oranlarının yanısıra momentum değişikliklerini de önbellekte (cache) saklar; yani RMSprop ve momentumu birleştirir.***
"""

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])



model.summary()

"""### Modelin Eğitilm ve Test Sonuçları"""

hist_ADAM=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[reduce_lr])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""### RMSprop OPTİMİZASYONU

***RMSprop ve benzeri olan AdaDelta, AdaGrad’ın bu sorununu çözerek bu hızlı düşüşü önler.***
"""

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
model.summary()

hist_RMSprob=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[reduce_lr])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""### ADAGRAD OPTİMİZASYONU

***AdaGrad seyrek parametreler için büyük güncellemeler yaparken sık parametreler için daha küçük güncellemeler yapar. Bu nedenle NLP ve resim tanıma gibi seyrek veriler için daha uygundur.***
AdaGrad’da her parametrenin kendi öğrenme hızı vardır ve algoritmanın özelliklerine bağlı olarak öğrenme oranı giderek azalmaktadır. Bu nedenle öğreneme oranı giderek azalır ve zamanın bir noktasında sistem öğrenmeyi bırakır. Bu AdaGrad’ın en büyük dez avantajıdır.
"""

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adagrad(),
              metrics=['accuracy'])
model.summary()

hist_adagrad=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[reduce_lr])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""### ADADELTA OPTİMİZASYONU
***AdaDelta, AdaGrad’ın bu sorununu çözerek bu hızlı düşüşü önler.***
"""

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

hist_adadelta=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[reduce_lr])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""## Kaydedilen sonuçların çizilmesi için plot fonksiyonunun tanımlanması"""

def plot_history(hists, attribute='val_loss', axis=(-1,21,0.85,0.94), loc='lower right'):
    ylabel = {'oss': 'loss', 'acc': 'accuracy'}
    title = {'val_loss': 'valid. loss', 'loss': 'trn. loss', 'val_acc': 'valid. accuracy', 'acc': 'trn. accuracy'}
    num_hists = len(hists)
    
    plt.figure(figsize=(12, 8))  
    plt.axis(axis)
    for i in range(num_hists):
        plt.plot(hists[i].history[attribute])
    plt.title(title[attribute])  
    plt.ylabel(ylabel[attribute[-3:]])  
    plt.xlabel('epoch')  
    plt.legend(['ADAM', 'SGD', 'RMSprob', 'adadelta', 'adagrad'], loc=loc)  

    plt.show()

hists = [hist_ADAM, hist_SGD, hist_RMSprob, hist_adadelta, hist_adagrad]

"""## SONUÇLARIN KARŞILAŞTIRILMASI"""

plot_history(hists, attribute='acc', axis=(-1,21,0.985,1.0), loc='lower right')

plot_history(hists, attribute='loss', axis=(-1,21,0.009,0.07), loc='upper right')

"""# SONUÇ
Kullandığımız optimizasyon algoritmasına göre epoch süresince eğitim hatasının nasıl değiştiğini çizdirdiğimiz grafiklerden inceleyebiliyoruz. Hatta öyle ki örneğin derin öğrenme modellerindençok sık kullanılan Stokastik Gradyan/Bayır İniş algoritması için değişmeyen diğer tüm parametreler durumunda bu problem için **20 Epoch** tan daha önce de eğitimi sonlandırabiliriz. 

![Happy Dance](https://gfx-bloggar.aftonbladet-cdn.se/wp-content/blogs.dir/428/files/2017/07/tenor-21.gif) 

*Burn After Reading Filminden :)*


### Test Yitim Değerlerini de aşağıdaki tablo ile inceleyeyim:


![TEST SONUÇLARI](https://i.hizliresim.com/alRa2B.png)




---

## KAYNAKLAR
---
* [Kaynak 1]( http://sebastianruder.com/optimizing-gradient-descent/) Optimizing Gradient Descent-Dean, J., Corrado, G. S., Monga, R., Chen, K., Devin, M., Le, Q. V, … Ng, A. Y. (2012). Large Scale Distributed Deep Networks. NIPS 2012: Neural Information Processing Systems.

* [Kaynak 2](http://proceedings.mlr.press/v37/ioffe15.pdf) Ioffe, S., & Szegedy, C. (2015). Batch Normalization : Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv Preprint arXiv:1502.03167v3.

* [Kaynak 3](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf) Qian, N. (1999). On the momentum term in gradient descent learning algorithms. Neural Networks : The Official Journal of the International Neural Network Society, 12(1), 145–151.

* [Kaynak 4](https://arxiv.org/pdf/1412.6980.pdf) Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations
Zaremba, W., & Sutskever, I. (2014). Learning to Execute, 1–25.

* [Kaynak 5](https://arxiv.org/pdf/1412.6651.pdf) Zhang, S., Choromanska, A., & LeCun, Y. (2015). Deep learning with Elastic Averaging SGD. Neural Information Processing Systems Conference (NIPS 2015).

* [Kaynak 6](https://ieeexplore.ieee.org/document/253713) Darken, C., Chang, J., & Moody, J. (1992). Learning rate schedules for faster stochastic gradient search. Neural Networks for Signal Processing II Proceedings of the 1992 IEEE Workshop, (September). 

* [Kaynak 7](https://keras.io/optimizers/) Usage of optimizers, Keras Docs

* [Kaynak 8](https://www.coursera.org/specializations/deep-learning) Coursera- deeplearning.ai, Deep Learning Specialization

* [Kaynak 9](https://imaddabbura.github.io/post/gradient_descent_algorithms/) Gradient Descent Algorithm and Its Variants

* [Kaynak 10](https://medium.com/@jrodthoughts/improving-deep-learning-algorithms-optimization-vs-regularization-f9b6e86fee8c) Improving Deep Learning Algorithms: Optimization vs. Regularization

* [Kaynak 11](https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f) Types of Optimization Algorithms used in Neural Networks and Ways to Optimize Gradient Descent

* [Kaynak 12](http://cs231n.github.io/optimization-1/) CS231n - Convolutional Neural Networks for Visual Recognition

* [Kaynak 13](https://medium.com/deep-learning-turkiye/derin-ogrenme-uygulamalarinda-en-sik-kullanilan-hiper-parametreler-ece8e9125c4) Derin Öğrenme Uygulamalarında En Sık kullanılan Hiper-parametreler
"""