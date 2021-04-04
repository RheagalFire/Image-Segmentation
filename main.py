import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import PIL
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('Failed')
    # Invalid device or cannot modify virtual devices once initialized.
    pass

model=load_model("Unet")

def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
	return href


if __name__ == '__main__':
    st.title("Cat-Dog-Mask")
    st.markdown(" ### This App returns masked images of your cats and dogs")

    image=st.sidebar.file_uploader("Upload an image",type=['jpg','png','jpeg'])
    if(image and st.sidebar.button("Mask")):
        image_=PIL.Image.open(image).resize((160,160))
        st.image(image_)
        serve=np.array(image_).reshape(1,160,160,3)
        predictions=model.predict(serve)
        predictions=predictions.reshape(160,160,3)
        st.image(predictions)

