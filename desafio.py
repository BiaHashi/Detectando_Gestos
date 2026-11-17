import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

tip_ids = [4,8,12,16,20]

def count_fingers(image, hand_landmarks, hand_no=0, hand_label="Right"):
    if hand_landmarks:
        landmarks = hand_landmarks[hand_no].landmark
        #print(landmarks)

        fingers = []
        
        for lm_index in tip_ids:
            finger_tip_y = landmarks[lm_index].y
            finger_bottom_y = landmarks[lm_index-2].y

            thumb_tip_x = landmarks[lm_index].x
            thumb_bottom_x = landmarks[lm_index-2].x
            
            if lm_index != 4:
                
                if finger_tip_y < finger_bottom_y:
                    fingers.append(1)
                    print("Dedo com ID ",lm_index," está aberto.")
                
                if finger_tip_y > finger_bottom_y:
                    fingers.append(0)
                    print("Dedo com ID ",lm_index," está fechado.")

            else:

                if hand_label == "Right":
                    if thumb_tip_x < thumb_bottom_x:
                        fingers.append(1)
                        print("Polegar direito aberto.")

                    else:
                        fingers.append(0)
                        print("Polegar direito fechado.")

                else:
                    if thumb_tip_x > thumb_bottom_x:
                        fingers.append(1)
                        print("Polegar esquerdo aberto.")
        
                    else:
                        fingers.append(0)
                        print("Polegar esquerdo fechado.")
        
        total_fingers = fingers.count(1)
        
        text = f"{hand_label}: {total_fingers}"

        pos_y = 50 +(hand_no*50)
        
        cv2.putText(
            image,
            text,
            (50,pos_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )

def draw_hand_landmarks(image,hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(203, 192, 255),
                    thickness=2,
                    circle_radius=5
                ),
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=3
                )
            )


while True:
    success,image = cap.read()

    if not success:
        print("Não foi possível capturar a imagem na câmera.")
        break

    image = cv2.flip(image,1)

    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    hand_landmarks = results.multi_hand_landmarks

    handeness = results.multi_handedness

    draw_hand_landmarks(image,hand_landmarks)

    if hand_landmarks and handeness:
        
        for hand_no, hand_info in enumerate(handeness):

            hand_label = hand_info.classification[0].label

            count_fingers(image, hand_landmarks, hand_no, hand_label)


    cv2.imshow("Controlador de Midia",image)

    key = cv2.waitKey(1)

    if key == 32:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()