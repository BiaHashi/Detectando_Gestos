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

    cv2.imshow("Controlador de Midia",image)

    key = cv2.waitKey(1)

    if key == 32:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()