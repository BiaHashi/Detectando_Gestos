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

def count_fingers (image,hand_landmarks, hand_no=0):
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

                if  thumb_tip_x > thumb_bottom_x:
                    fingers.append(1)
                    print("Polegar está aberto.")
                                
                if  thumb_tip_x < thumb_bottom_x:
                    fingers.append(0)
                    print("Polegar está fechado.")
        
        total_fingers = fingers.count(1)
        
        text = f"Dedos: {total_fingers}"
        
        cv2.putText(image,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

def draw_hand_landmarks(image,hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image,landmarks,mp_hands.HAND_CONNECTIONS)


while True:
    success,image = cap.read()

    if not success:
        print("Não foi possível capturar a imagem na câmera.")
        break

    image = cv2.flip(image,1)

    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    hand_landmarks = results.multi_hand_landmarks

    draw_hand_landmarks(image,hand_landmarks)

    count_fingers(image,hand_landmarks)

    cv2.imshow("Controlador de Midia",image)

    key = cv2.waitKey(1)

    if key == 32:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()