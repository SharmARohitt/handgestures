import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands and Drawing
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize audio utilities
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the volume range
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# Start processing video frames
with mp_hands.Hands(min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image to detect hands
        results = hands.process(image)

        # Convert the image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the hand detection annotations and control volume
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the thumb tip and wrist
                x1, y1 = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y  # Thumb tip
                x2, y2 = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y  # Wrist

                # Calculate the angle between the thumb and wrist
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

                # Control the volume based on the thumb orientation
                if angle > 160:  # Thumb up
                    vol = minVol + (maxVol - minVol) * 0.5
                elif angle < 20:  # Thumb down
                    vol = minVol + (maxVol - minVol) * 0.1
                else:  # Thumb neutral
                    vol = minVol + (maxVol - minVol) * 0.3

                print("Volume: ", vol)
                volume.SetMasterVolumeLevel(vol, None)

                # Draw the volume bar
                volBar = np.interp(vol, [minVol, maxVol], [400, 150])
                cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

        # Display the image
        cv2.imshow('Hand Volume Control', cv2.flip(image, 1))

        # Exit on 'Esc' key press
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()