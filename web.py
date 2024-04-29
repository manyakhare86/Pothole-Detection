import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("best1.pt")
class_names = model.names

# Function to perform pothole detection on an image
def detect_potholes_image(image):
    img = np.array(image)
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs

    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (img.shape[1], img.shape[0]))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)
                cv2.rectangle(img,(x,y),(x1+x,y1+y),(255,0,0),2)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

# Function to perform pothole detection on a video
def detect_potholes_video(uploaded_video):
    # Save the uploaded video to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Open the temporary video file using cv2.VideoCapture
    cap = cv2.VideoCapture("temp_video.mp4")

    # Video processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        h, w, _ = frame.shape
        results = model.predict(frame)

        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, x1, y1 = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x1+x, y1+y), (255, 0, 0), 2)
                    cv2.putText(frame, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convert the processed frame to RGB (Streamlit expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the processed frame in the Streamlit app
        st.image(frame_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()

# Streamlit app
st.title("Pothole Detection App")

option = st.sidebar.selectbox(
    'Choose an option:',
    ('Image', 'Video', 'Webcam')
)

if option == 'Image':
    st.subheader("Upload an image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.subheader("Result")
        detected_image = detect_potholes_image(image)
        st.image(detected_image, caption='Detected Potholes', use_column_width=True)

elif option == 'Video':
    st.subheader("Upload a video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4"])

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.subheader("Result")
        detect_potholes_video(uploaded_video)

elif option == 'Webcam':
    st.subheader("Webcam Feed")
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Webcam processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        h, w, _ = frame.shape
        results = model.predict(frame)

        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, x1, y1 = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x1+x, y1+y), (255, 0, 0), 2)
                    cv2.putText(frame, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convert the processed frame to RGB (Streamlit expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the processed frame in the Streamlit app
        st.image(frame_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
