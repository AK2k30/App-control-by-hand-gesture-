import cv2
import mediapipe as mp
import webbrowser

mp_hands = mp.solutions.hands # type: ignore
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils # type: ignore

webcam = cv2.VideoCapture(0)

is_youtube_open = False
is_chrome_open = False
is_whatsapp_open = False
is_linkedin_open = False
is_spotify_open = False

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            thumb = landmarks[4].x < landmarks[3].x < landmarks[2].x < landmarks[1].x
            index_finger = landmarks[8].y < landmarks[6].y
            middle_finger = landmarks[12].y < landmarks[10].y
            ring_finger = landmarks[16].y < landmarks[14].y
            pinky_finger = landmarks[20].y < landmarks[18].y
            num_fingers = sum([thumb, index_finger, middle_finger, ring_finger, pinky_finger])

            cv2.putText(image, f"Fingers: {num_fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if num_fingers == 2 and not is_youtube_open:
                webbrowser.open('https://www.youtube.com')
                is_youtube_open = True
            elif num_fingers == 3 and not is_chrome_open:
                webbrowser.open('https://www.google.com')
                is_chrome_open = True
            elif num_fingers == 4 and not is_whatsapp_open:
                webbrowser.open('https://web.whatsapp.com')
                is_whatsapp_open = True
            elif num_fingers == 5 and not is_linkedin_open:
                webbrowser.open('https://www.linkedin.com')
                is_linkedin_open = True
            elif num_fingers == 1 and not is_spotify_open:
                webbrowser.open('https://www.spotify.com')
                is_spotify_open = True

            if num_fingers != 2:
                is_youtube_open = False
            if num_fingers != 3:
                is_chrome_open = False
            if num_fingers != 4:
                is_whatsapp_open = False
            if num_fingers != 5:
                is_linkedin_open = False
            if num_fingers != 1:
                is_spotify_open = False

    cv2.imshow("Hand Gesture Detection", image)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
