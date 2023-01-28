# create a Streamlit WebApp

import tensorflow as tf
model = tf.keras.models.load_model('C:/Users/Owner/Desktop/streamlit/my_model.hdf5')



import streamlit as st
st.write("""
         # Cat Dog Sign Prediction
         """
         )
            # this statement will be written in Bold (Highlighted)

st.write("Simple image Classification Web-App to predict Cat & Dog Image")
# it will be also written as Comment in Web App
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])



import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75)))/255.
        #,    interpolation=cv2.INTER_CUBIC   ((it should be checked ))

        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a Cat !")
    else:
        st.write("It is a dog!")
    
    st.text("Probability (0: Cat, 1: Dog")
    st.write(prediction)