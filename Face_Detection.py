from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import cv2
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Uncomment to load the model we saved

#PLEASE DOWNLOAD THE MODELS FROM THE GOOGLE DRIVE LINK !!!!

#model = tf.keras.models.load_model('FM_Simple_CNN.h5')
#model = tf.keras.models.load_model('FM_VGG16.h5')
model = tf.keras.models.load_model('FM_MobileNetV2.h5')

#SET TO TRUE IF USING SIMPLE MODEL!!!
Flip_Classes=False



# Capture from webcam.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    # Read the frame
    _, img = cap.read()
    cv2.imshow("Input", img)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped= np.zeros((244,244,3), np.uint8)
    cropped_res=np.zeros((244,244,3), np.uint8)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=10, minSize=(50, 50),flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        people=len(faces)
        cropped = img[y:y + h, x:x + w]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cv2.imshow("Detected_face", cropped)
        cropped = cv2.resize(cropped, (224, 224))
        cv2.imshow("Resized_Face", cropped)
        cropped = img_to_array(cropped)
        cropped = preprocess_input(cropped)
        cropped = np.expand_dims(cropped, axis=0)
        # pass the face through the model to determine if the face has a mask or not
        if Flip_Classes == True:
            (withoutMask,mask) = model.predict(cropped)[0]
        else:
            (mask,withoutMask) = model.predict(cropped)[0]
        print(model.predict(cropped))
        # determine the class label and color we'll use to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # display the label and bounding box rectangle on the output frame
        cv2.putText(img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print(label)
    cv2.imshow("Output_Feed", img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# Release the VideoCapture object
cap.release()
