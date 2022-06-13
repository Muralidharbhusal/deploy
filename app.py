from camera import VideoTransformer
from model import face_detect
from homepage import home
import streamlit as st
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer


def main():

    #Title of the web app
    st.title("Face Emotion detection")
    
    #Available function in the web app
    activities = ["Home", "Detection with webcam", "Detection with picture"]
    choice = st.sidebar.selectbox("Select the required activty to navigate.", activities)

    #Navigating to Homepage
    if choice == "Homepage":
      
      home()

    #Navigating to face emotion detction with pre=trained deep learning model on live web-cam            
    if choice == "Detection with webcam":

      st.text('Click on start to start detecting')
      webrtc_streamer(key="exam", video_transformer_factory=VideoTransformer)

    #Navigating to face emotion detction on pictures uploaded by users            
    if choice == "Detection with picture":

      #Uploading the image
      image_file = st.file_uploader("Upload the image", type=['jpeg', 'png', 'jpg'])

      if image_file is not None:
          image1 = Image.open(image_file)
          image = np.asarray(image1)

          #Displaying the uploaded image
          st.image(image1)
          st.text('Press Process to display the face emotion detected image.')

          #Processing the uploaded image
          if st.button("Process"):
            face_detect(image)


if __name__ == "__main__":
    main()