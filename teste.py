import mediapipe as mp

print("Versão:", mp.__version__)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

print("MediaPipe inicializado com sucesso!")

hands.close()