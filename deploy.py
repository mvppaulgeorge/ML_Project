import tensorflow as tf
model = tf.keras.models.load_model('cnn_3.h5')
import streamlit as st
import mysql.connector
import cv2
import glob
import os
from gtts import gTTS
from googletrans import Translator

  
# Language in which you want to convert
language = 'en'

import numpy as np
from PIL import Image, ImageOps
st.set_page_config(
     page_title="Breast Cancer Diagnostic Application",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.google.com',
         'Report a bug': "https://mail.google.com/mail",
         'About': "# There is nothing here."
     }
 )


if 'account' not in st.session_state:
     st.session_state.account = dict()
st.write("""
         # Diagnoses of breast cancer(IDC) using medical images
         """
         )
st.write("This is a simple image classification web app to diagnose breast cancer IDC using tissue example images")
email = st.text_input("Enter your email:", "trojan@usc.edu")
name = st.text_input("Enter your name:", "")
if 'email' not in st.session_state:
    if 'name' not in st.session_state:
       st.session_state.email = False


press = st.button('Submit')
if press:
    st.session_state.email = True
    # myObj = {"action":"registration","email":email};
if st.session_state.email:
    st.write('You sign in as', email)
    if email not in st.session_state.account.keys():
          st.session_state.account[email] = 0
    
    file = st.file_uploader("Please upload image files", type=["jpg", "png"], accept_multiple_files=True)

    if file is None:
        st.text("Please upload an image file")
    else:
        # myObj = {"action":"photo","email":email};       
        x = []
        WIDTH = 50
        HEIGHT = 50
        CHANNELS = 3
        for img in file[:30]:
          full_size_image = np.array(Image.open(img))
          x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
    
        X=np.array(x)
        X=X/255.0
        test = X
        l = len(test)
        i_sum = 0
        for i in range(0,l):
            img = test[i]
            img = tf.expand_dims(img,axis=0)
            pred = model.predict(img)
            i_sum = i_sum+pred[0][0]
        if l ==0:
            res_class0=0
            res_class1=0
       
        else:       
            res_class0=(i_sum)/l
            res_class1=1-res_class0
            
            str_0 = "Our application has predicted a "+str(round(res_class1,3))+" possibility of diagnosing IDC breast cancer. "
            str_1 = "Your case is unlikely to be malignant but it is suggested to do further examination"
            str_2 = "Your case is very unlikely to be malignant"
            str_3 = "Your case is likely to be malignant it is suggested to do further examination"
            str_4 = "Your case is likely to be malignant it is suggested to do further examination"
            
            st.write(str_0)
            if res_class0>=0.5 and res_class0<0.75:
                 st.write(str_1)
                 str_cnt=str_0+" "+str_1
            
            elif res_class0>=0.75:
                 st.write(str_2)
                 str_cnt=str_0+" "+str_2

            elif res_class0>=0.25 and res_class0 < 0.5:
                 st.write(str_3)
                 str_cnt=str_0+" "+str_3

            else:
                 st.write(str_4)
                 str_cnt=str_0+" "+str_4
           
          
            myobj = gTTS(text=str_cnt, lang=language, slow=False)

            myobj.save("voice.mp3")

            os.system("voice.mp3")
            audio_file = open('voice.mp3', 'rb')
            audio_bytes = audio_file.read()

            st.audio(audio_bytes, format='audio/mp3')
            
            
            translator = Translator()
            out_lang = st.selectbox(
                "Select your output language",
                ("Chinese", "French", "Spanish", "Korean", "Japanese"),
            )
            if out_lang == "Chinese":
                output_language = "zh-cn"
            elif out_lang == "French":
                output_language = "fr"
            elif out_lang == "Spanish":
                output_language = "es"
            elif out_lang == "Korean":
                output_language = "ko"
            elif out_lang == "Japanese":
                output_language = "ja"
            
            if st.button("Translate"):
                translation = translator.translate(str_cnt, src=language, dest=output_language)
                trans_text = translation.text
                
                st.write(trans_text)
                
                trans = gTTS(trans_text, lang=output_language, slow=False)

                trans.save("tts.mp3")

                os.system("tts.mp3")
                audio_file = open('tts.mp3', 'rb')
                audio_bytes = audio_file.read()

                st.audio(audio_bytes, format='audio/mp3')

def remove_files(n):
    mp3_files = glob.glob("temp/*mp3")
    if len(mp3_files) != 0:
        now = time.time()
        n_days = n * 86400
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n_days:
                os.remove(f)
                print("Deleted ", f)


remove_files(7)


