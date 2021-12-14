import cv2
import pandas as pd
import numpy as np
from statistics import mode
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
spotify_df = pd.read_csv('datasets/kaggleMusicMoodFinal.csv')
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

def EmotionDetection():
    emotion_labels={0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
    detection_model_path = 'haarcascade_frontalface_default.xml'
    emotion_model_path = 'model.h5'
    frame_window = 10
    emotion_offsets = (20, 40)

    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_classifier = model.load_weights(emotion_model_path)

    emotion_window = []

    video_capture = cv2.VideoCapture(0)
    flag=1
    while True:
        ret, frame = video_capture.read()
        if not ret:
            exit(0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            if flag:
                print("Mood detected is: ",emotion_labels[maxindex])
                print("Playlist for ",emotion_labels[maxindex]," mood:")
                print("\n")
                return(emotion_labels[maxindex])
                flag=0
            cv2.putText(frame, emotion_labels[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Video', cv2.resize(frame,(900,600),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

p1_disgust=spotify_df[spotify_df['Mood'].isin(['Energetic', 'Happy', 'Calm'])]
p2_angry=spotify_df[spotify_df['Mood'].isin(['Energetic', 'Calm'])]
p3_fear=spotify_df[spotify_df['Mood'].isin(['Happy', 'Calm'])]
p4_happy=spotify_df[spotify_df['Mood'].isin(['Sad', 'Happy', 'Calm'])]
p5_sad=spotify_df[spotify_df['Mood'].isin(['Sad', 'Happy', 'Calm'])]
p6_surprise=spotify_df[spotify_df['Mood'].isin(['Energetic', 'Happy', 'Sad'])]
p7_neutral=spotify_df[spotify_df['Mood'].isin(['Energetic', 'Happy', 'Calm'])]

def GeneratePlaylist(list_df):
    s1 = list_df.sort_values(by=['year'], ascending=False)
    s2 = s1.nlargest(30,['popularity'])
    s3 = s2[["name", "artists"]]
    return s3

pdisgust=GetPlaylist(p1_disgust)
pangry=GetPlaylist(p2_angry)
pfear=GetPlaylist(p3_fear)
phappy=GetPlaylist(p4_happy)
psad=GetPlaylist(p5_sad)
psurprise=GetPlaylist(p6_surprise)
pneutral=GetPlaylist(p7_neutral)

def RecommendTop30(x):
    if x == "Disgust":
        playlist=pdisgust
    elif x == "Angry":
        playlist=pangry
    elif x == "Fear":
        playlist=pfear
    elif x == "Happy":
        playlist=phappy
    elif x == "Sad":
        playlist=psad
    elif x == "Surprise":
        playlist=psurprise
    else:
        playlist=pneutral
    print(playlist.to_markdown(index=False))

RecommendTop30(EmotionDetection())
