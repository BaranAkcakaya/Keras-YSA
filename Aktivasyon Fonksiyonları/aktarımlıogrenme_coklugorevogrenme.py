from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from io import BytesIO
import os
import requests

model = ResNet50(weights="imagenet")

layers = dict([(layer.name, layer.output) for layer in model.layers])
model.summary()

# MODELDEKİ TOPLAM PARAMETRE SAYISINI EKRANA YAZDIR
model.count_params()

def prepare_image(image, target):
	# giriş görüntüsünü yeniden boyutlandırma ve ön işlemerin yapılması
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# işlenmiş görüntüyü alma
	return image

"""### İnternet Kaynaklı bir görüntünün URL'sini kopyalayarak test işlemini yapabilirsiniz. Binlerce kategorinin olduğu ImageNet veri seti sayesinde bir çok sınıfı rahatlıkla kategorize edilebildiğini göreceksiniz."""

#@title Görüntünün URL'sini Yapıştırın { vertical-output: true }
ImageURL = "https://3.bp.blogspot.com/-u2EcSH2R3aM/VM69jPZvvOI/AAAAAAAAYzk/xmjSdaDD06o/s1600/mercan_resif.jpg" #@param {type:"string"}

#ImageURL = "https://i.cnnturk.com/ps/cnnturk/75/650x0/57ad7dd9a781b6264026292d.jpg"
response = requests.get(ImageURL)
image = Image.open(BytesIO(response.content))
image

"""**Eğer Dosyadan Resim Okumak isterseniz**"""

# root = 'drive/My Drive/'
# image_path = root+ 'Olips.png'
# image = Image.open(image_path)
# image = image.resize((224, 224))
# image
# Görüntüyü diziye çevir
# x = np.asarray(image, dtype='float32')
# Dizi listesine çevir
# x = np.expand_dims(x, axis=0)
# Giriş görüntüsünü eğitim setine uygun şekilde ön işlemleri yap 
# x = preprocess_input(x)
#preds = model.predict(x)
#print('Predicted:', decode_predictions(preds, top=3)[0])
#print(decode_predictions(preds, top=1)[0][0][1])

"""### İnternetten Aldığınız Verinin Ön İşlemlerinin Yapılması Yeniden Boyutlandırılması ve Olası ilk 5 Tahmin ve Tahmin Oranlarının Ekrana Yazdırılması

Örnekteki görsel için %91.9 olasılıkla **mercan** olduğunu %0.17 olasılıkla **denizşakayığı** ve diğer olasılıkları hücrenin çıktısından takip edebilirsiniz.
"""

data = {"success": False}

pre_image = prepare_image(image, target=(224, 224)) # 224 x 224 boyutlu hale getir

preds = model.predict(pre_image) # Kesirim modeline ön işlemden geçmiş görüntüyü uygula

results = imagenet_utils.decode_predictions(preds) #kestirim
data["predictions"] = []


for (imagenetID, label, prob) in results[0]: # ImageNet veri kümseinden etiket, olasılık ve kestrim sonucunu al
  r = {"label": label, "probability": float(prob)}
  data["predictions"].append(r)
  
data["success"] = True

print(data)

"""### En yüksek olasılıklı sonucun ekrana yazdırılması"""

print("Sınıflandırma tahmini en yüksek olan {0} oranıyla {1}'dır.".format(data["predictions"][0]["probability"],data["predictions"][0]["label"])) 
# En yüksek olasılıklı sonucu ekrana yazdır

"""## ⭐️[TensorFlow Hub Örneğini incelemeniz de çok faydalı olacaktır](https://www.tensorflow.org/tutorials/images/hub_with_keras)⭐️
 ### ⭐️ [Transfer learning from pre-trained models](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)⭐️

# 2. VERSİYON İÇİN ÖRNEK

### Kütüphanelerin kurulması ve gerekli importların yapılması adımı
⏬⏬⏬
"""

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras import backend as K
import pandas as pd
import numpy as np

"""## Görüntülerimizin Boyutlarının Ayarlanması
Ön işlemler
"""

img_width, img_height = 224, 224

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000 #eğitim örnek sayısı
nb_validation_samples = 800 #geçerleme örnek 
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)-

def preprocess_input_vgg(x):
    """
    Paremetreler
    ----------
    x :  numpy 3d dizi (bir tek görüntü ön işlemlendi)
      
    """
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

"""### VGG 16 Derin Öğrenme Modelinin IMAGENET Veri Kümesinde Eğitilmiş Ağırlıklarının Alınması
Keras kütüphanesinden faydalanıyoruz bu aşamada

[VGGNet](https://medium.com/deep-learning-turkiye/deri%CC%87ne-daha-deri%CC%87ne-evri%C5%9Fimli-sinir-a%C4%9Flar%C4%B1-2813a2c8b2a9) Derin Öğrenme Modeli ve [ImageNet](https://medium.com/deep-learning-turkiye/motivasyon-yapay-zeka-ve-derin-%C3%B6%C4%9Frenme-48d09355388d) Veri Kümesi hakkında bilgi için tıklayınız!
"""

vgg16 = VGG16(weights='imagenet')

x  = vgg16.get_layer('fc2').output
prediction = Dense(2, activation='softmax', name='predictions')(x)

model = Model(inputs=vgg16.input, outputs=prediction)


# base_model = VGG16(weights='imagenet',include_top= False, input_shape=input_shape)

# x = base_model.output
# x = Flatten(name='flatten')(x)
# x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
# prediction = Dense(2, activation='linear', name='predictions')(x)
# # prediction = Dense(output_dim=1, activation='sigmoid', name='logit')(x)

# top_model = Sequential()
# top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# predictions = Dense(2, activation='linear', name='predictions')(top_model)
# top_model.load_weights('bootlneck_fc_model.h5')

# model = Model(input= base_model, output=prediction)

# fc2 = vgg16.get_layer('fc2').output
# prediction = Dense(units=2, activation='relu', name='logit')(fc2)
# model = Model(inputs=vgg16.input, outputs=top_model)

"""###  Çıkıştaki tam bağlantı katmanına kadar tamamını Fine-Tuning işlemi için dondur"""

for layer in model.layers:
    if layer.name in ['predictions']:
        continue
    layer.trainable = False


df = pd.DataFrame(([layer.name, layer.trainable] for layer in model.layers), columns=['layer', 'trainable'])


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(directory='data/train',
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
validation_generator = validation_datagen.flow_from_directory(directory='data/validation',
                                                              target_size=[img_width, img_height],
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

"""### Optimizasyon yöntemini Stokastik Gradyan/Bayır İniş ve Küçük bir Öğrenme Oranı ile Çalıştırma"""

sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Optimizasyon Yöntemini ADAM ile de değiştirebilirsiniz
# model.compile(optimizer='nadam',
#                   loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
#                   metrics=['accuracy'])

# ERKEN DURDURMA DA EKLEYEBİLİRSİNİZ
# top_weights_path = 'top_model_weights_fine_tune.h5'
# callbacks_list = [
#         ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
#         EarlyStopping(monitor='val_acc', patience=5, verbose=0)]
    

# FINE-TUNING YAPMAK İÇİN
# model.fit_generator(train_generator,
#                     samples_per_epoch=16,
#                     nb_epoch=10,
#                     validation_data=validation_generator,
#                     nb_val_samples=32);

model.fit_generator(
        train_generator,
        # steps_per_epoch=16,
        steps_per_epoch=2000 // batch_size,
        epochs=4,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
        # validation_steps=32) #,
        # callbacks=callbacks_list)

"""### EĞİTİLMİŞ AĞIRLIKLARIN KAYDEDİLMESİ"""

model.save_weights('vgg16_tf_cat_dog_final_dense2.h5')

model_json_final = model.to_json()
with open("vgg16_tf_cat_dog_final_dense2.json", "w") as json_file:
    json_file.write(model_json_final)

"""### Kestirim Sonucunun Ekrana Gösterilmesi Adımları"""

from IPython.display import display
import matplotlib.pyplot as plt

X_val_sample, _ = next(validation_generator)
y_pred = model.predict(X_val_sample)

nb_sample = 4
for x, y in zip(X_val_sample[:nb_sample], y_pred[:nb_sample]):
    s = pd.Series({'Cat': 1-np.max(y), 'Dog': np.max(y)})
    axes = s.plot(kind='bar')
    axes.set_xlabel('Class')
    axes.set_ylabel('Probability')
    axes.set_ylim([0, 1])
    plt.show()

    img = array_to_img(x)
    display(img)