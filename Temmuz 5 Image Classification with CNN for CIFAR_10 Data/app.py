import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('My_cnn_model.h5')

def process_image(img):
    img=img.resize((32, 32))
    img=img.convert('RGB')
    img=np.array(img)
    img=img / 255.0
    img=np.expand_dims(img, axis=0)
    return img

st.write("Resim seç ve model ne olduğunu tahmin etsin")

file=st.file_uploader('Bir resim seç',type=['jpg','jpeg','png'])# resim yükleme özelliği

if file is not None:  # Eğer resim yüklendiyse
    img = Image.open(file)
    st.image(img, caption='Yüklenen resim')  # yüklenecek resmi gösterir
    image = process_image(img)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]  # En yüksek olasılıklı sınıfı alır
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 sınıf isimleri
    st.write(class_names[predicted_class])  # Tahmin edilen sınıf ismini gösterir