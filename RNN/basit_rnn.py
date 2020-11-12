import numpy as np

"""* `timesteps` girişteki zaman adımı sayısı
* `input_features` girdi nitelik uzayının boyutu
* `output_features` çıktı nitelik uzayının boyutu
"""

timesteps = 100
input_features = 32
output_features = 64

"""Basit olması için girdi verisini rastgele gürültü olarak seçelim"""

inputs = np.random.random((timesteps, input_features))

"""Başlangıç durumu için tüm elemanları sıfırdan oluşan bir vektör oluşturalım"""

state_t = np.zeros((output_features,))

"""**Rastgele oluşturulan ağırlık matrisleri**"""

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

"""### Girdi ve mevcut duruma göre çıktının oluşturulması"""

successive_outputs = []
for input_t in inputs:
  output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) +b) 
  successive_outputs.append(output_t)
  state_t = output_features

final_output_sequence = np.concatenate(successive_outputs, axis=0)

"""En basit hali ile Yinelemeli Sinir Ağları (RNN) her döngü adımında bir önceki adımın değerini kullanmaktadır. 

![Simple RNN](https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99E7EA485A9E2CBE17)
"""

output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)

output_t

