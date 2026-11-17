
# Importa o OpenCV, que permite usar a câmera e trabalhar com imagens.
import cv2

# Importa o MediaPipe, que identifica pontos das mãos.
# "mp" é um apelido para escrever menos.
import mediapipe as mp


# Liga a câmera de número 0 (normalmente a webcam principal).
cap = cv2.VideoCapture(0)


# Acessa a ferramenta do MediaPipe que reconhece mãos.
mp_hands = mp.solutions.hands

# Acessa a ferramenta que desenha pontos e linhas nas mãos.
mp_drawing = mp.solutions.drawing_utils


# Configura o detector de mãos.
hands = mp_hands.Hands(

    # False: o programa trabalha com vídeo, não com fotos isoladas.
    static_image_mode=False,

    # Permite identificar até duas mãos.
    max_num_hands=2,

    # Exige confiança mínima de 80% para detectar uma mão.
    min_detection_confidence=0.8,

    # Exige confiança mínima de 50% para acompanhar a mão.
    min_tracking_confidence=0.5
)


# Guarda os números dos pontos das pontas dos cinco dedos.
# 4 = polegar, 8 = indicador, 12 = médio,
# 16 = anelar, 20 = mindinho.
tip_ids = [4, 8, 12, 16, 20]


# Cria uma função para contar os dedos levantados.
# image = imagem da câmera.
# hand_landmarks = pontos das mãos encontradas.
# hand_no = número da mão que queremos analisar.
def count_fingers(image, hand_landmarks, hand_no=0):

    # Só continua se alguma mão foi encontrada.
    if hand_landmarks:

        # Pega os 21 pontos da mão escolhida.
        landmarks = hand_landmarks[hand_no].landmark

        # Mostraria os pontos no terminal (está desativado).
        # print(landmarks)

        # Cria uma lista vazia para guardar os dedos abertos e fechados.
        fingers = []

        # Percorre as pontas dos cinco dedos.
        for lm_index in tip_ids:

            # Pega a posição vertical da ponta do dedo.
            finger_tip_y = landmarks[lm_index].y

            # Pega a posição vertical de uma articulação do dedo.
            finger_bottom_y = landmarks[lm_index-2].y

            # Pega a posição horizontal da ponta do dedo.
            thumb_tip_x = landmarks[lm_index].x

            # Pega a posição horizontal de uma articulação.
            thumb_bottom_x = landmarks[lm_index-2].x

            # Verifica se o dedo NÃO é o polegar.
            if lm_index != 4:

                # Se a ponta está acima da articulação,
                # considera o dedo aberto.
                if finger_tip_y < finger_bottom_y:

                    # Guarda 1 para indicar dedo aberto.
                    fingers.append(1)

                    # Mostra no terminal qual dedo está aberto.
                    print("Dedo com ID ", lm_index, " está aberto.")

                # Se a ponta está abaixo da articulação,
                # considera o dedo fechado.
                if finger_tip_y > finger_bottom_y:

                    # Guarda 0 para indicar dedo fechado.
                    fingers.append(0)

                    # Mostra no terminal qual dedo está fechado.
                    print("Dedo com ID ", lm_index, " está fechado.")

            # Se for o polegar, usa outra regra.
            else:

                # Se a ponta está mais à direita,
                # considera o polegar aberto.
                if thumb_tip_x > thumb_bottom_x:

                    # Guarda 1 para polegar aberto.
                    fingers.append(1)

                    # Mostra a informação no terminal.
                    print("Polegar está aberto.")

                # Se a ponta está mais à esquerda,
                # considera o polegar fechado.
                if thumb_tip_x < thumb_bottom_x:

                    # Guarda 0 para polegar fechado.
                    fingers.append(0)

                    # Mostra a informação no terminal.
                    print("Polegar está fechado.")

        # Conta quantas vezes o número 1 aparece na lista.
        total_fingers = fingers.count(1)

        # Cria o texto com a quantidade de dedos levantados.
        text = f"Dedos: {total_fingers}"

        # Escreve esse texto na imagem da câmera.
        cv2.putText(
            image, text, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2
        )


# Cria uma função para desenhar os pontos das mãos.
def draw_hand_landmarks(image, hand_landmarks):

    # Só continua se encontrou alguma mão.
    if hand_landmarks:

        # Percorre todas as mãos encontradas.
        for landmarks in hand_landmarks:

            # Desenha os pontos e suas conexões na imagem.
            mp_drawing.draw_landmarks(
                image, landmarks, mp_hands.HAND_CONNECTIONS
            )


# Repete o programa continuamente até mandarmos parar.
while True:

    # Captura uma imagem da câmera.
    # success indica se a captura funcionou.
    success, image = cap.read()

    # Se a câmera falhar, mostra um aviso e encerra o laço.
    if not success:
        print("Não foi possível capturar a imagem na câmera.")
        break

    # Espelha a imagem, como se fosse um espelho.
    image = cv2.flip(image, 1)

    # Converte as cores de BGR para RGB.
    # O MediaPipe espera receber a imagem em RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Envia a imagem para o MediaPipe procurar mãos.
    results = hands.process(image_rgb)

    # Guarda os pontos das mãos encontradas.
    hand_landmarks = results.multi_hand_landmarks

    # Desenha os pontos e linhas sobre as mãos.
    draw_hand_landmarks(image, hand_landmarks)

    # Conta os dedos e escreve o resultado na imagem.
    count_fingers(image, hand_landmarks)

    # Mostra a imagem em uma janela chamada Controlador de Midia.
    cv2.imshow("Controlador de Midia", image)

    # Aguarda 1 milissegundo e verifica se alguma tecla foi pressionada.
    key = cv2.waitKey(1)

    # Se a tecla Espaço (código 32) for pressionada, encerra o laço.
    if key == 32:
        break


# Encerra o detector de mãos do MediaPipe.
hands.close()

# Desliga e libera a câmera.
cap.release()

# Fecha todas as janelas abertas pelo OpenCV.
cv2.destroyAllWindows()
