import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('xception_deepfake_image.h5')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((299, 299))  # Resize the image to match the input size expected by the model
    image = np.array(image)
    image = image.astype('float32') / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    return image

# Function to predict if an image is deep fake or real
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]  # Return the probability

# Function to detect and crop face from image
def detect_and_crop_face(image):
    # Convert PIL image to OpenCV format
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    
    # Assume the first detected face is the target
    x, y, w, h = faces[0]
    cropped_face = img[y:y+h, x:x+w]
    
    # Convert back to PIL image
    return Image.fromarray(cropped_face)

# Function to extract frames from video
def extract_frames(video_path, interval=30):
    frames = []
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        st.error("Failed to open the video file.")
        return frames

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize the progress bar
    progress_bar = st.progress(0)
    
    for i in range(0, frame_count, interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        else:
            st.error(f"Failed to read frame {i}.")
            break
        # Update progress bar
        progress_bar.progress(int((i / frame_count) * 100))
    
    video.release()
    progress_bar.empty()
    return frames

# Function to download video from a URL
def download_video_from_url(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Create a temporary file to save the video
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')  # Ensure .mp4 extension
            with open(temp_video.name, 'wb') as f:
                f.write(response.content)
            return temp_video.name
        else:
            st.error("Failed to download the video. Please check the URL.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Streamlit app layout
st.set_page_config(page_title="Deep Fake Detector Tool", page_icon="🕵️‍♂️")

st.title('Video Deep Fake Detection')

# Upload video
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

# Video URL input
video_url = st.text_input("Or enter video URL:")

if uploaded_video is not None or video_url:
    if uploaded_video is not None:
        # Save uploaded video temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')  # Ensure .mp4 extension
        temp_video.write(uploaded_video.read())
        temp_video.close()
        video_path = temp_video.name
    else:
        # Download video from the URL
        video_path = download_video_from_url(video_url)
    
    if video_path:
        try:
            # Extract frames from the video
            st.write("Extracting frames from the video...")
            frames = extract_frames(video_path)

            if frames:
                fake_count = 0
                real_count = 0

                st.write(f"Total frames extracted: {len(frames)}")
                # Display all extracted frames
                for i, frame in enumerate(frames):
                    st.image(frame, caption=f'Frame {i+1}', use_column_width=True)
                    face_image = detect_and_crop_face(frame)
                    if face_image:
                        prediction = predict(face_image)
                        if prediction > 0.4:
                            fake_count += 1
                            st.write(f'Frame {i+1}: **DEEP FAKE**', color='red')
                        else:
                            real_count += 1
                            st.write(f'Frame {i+1}: **Real Image**', color='green')
                    else:
                        st.write(f'Frame {i+1}: No face detected', color='orange')

                # Summary
                st.write(f"\nSummary:")
                st.write(f"Deep Fake Frames: {fake_count}", color='red')
                st.write(f"Real Frames: {real_count}", color='green')
                if fake_count > real_count:
                    st.write("The video is likely to be a **DEEP FAKE**.", color='red')
                else:
                    st.write("The video is likely to be a **Real Video**.", color='green')
            
        finally:
            # Clean up the temporary file
            if os.path.exists(video_path):
                os.remove(video_path)
    else:
        st.error("Unable to process the video. Please check the video file or URL.")
