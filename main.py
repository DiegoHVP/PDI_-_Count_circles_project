import cv2
import numpy as np
import time

# fluxo de vídeo RTSP
rtsp_url = "rtsp://192.168.0.4:8080/h264_opus.sdp"

# Abrir o fluxo de vídeo
cap = cv2.VideoCapture(rtsp_url)

# Aguarda 2 segundos para garantir que o stream esteja inicializado
time.sleep(2)

# Verifica se a câmera ou fluxo de vídeo abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir o fluxo de vídeo! Verifique a URL e a conexão.")
    exit()

print("Pressione 'q' para sair.")

# Dicionário para armazenar a contagem de cores detectadas
contagem_cores = {'vermelho': 0, 'azul': 0, 'verde': 0}

# Função para classificar a cor de um objeto com base nos valores médios em BGR
def classificar_cor(media_bgr):
    bgr_color = np.uint8([[media_bgr]])  # Converte a cor média para formato NumPy
    hsv = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]  # Converte para HSV
    hue, sat, val = hsv  # Separa os canais de Matiz (H), Saturação (S) e Valor (V)
    
    # Define intervalos de cores com base no matiz (H)
    if (hue <= 10 or hue >= 170) and sat > 80 and val > 80:
        return 'vermelho'
    elif 90 <= hue <= 130 and sat > 80 and val > 80:
        return 'azul'
    elif 40 <= hue <= 80 and sat > 80 and val > 80:
        return 'verde'
    return None  # Se não corresponder a nenhuma das cores definidas

# Criar uma única janela para exibição
cv2.namedWindow("Detecção de Círculos", cv2.WINDOW_NORMAL)

# Loop principal para capturar e processar os quadros do vídeo
while True:
    ret, frame = cap.read()  # Captura um quadro do vídeo
    
    if not ret:
        print("Erro na captura do quadro. Tentando reconectar...")
        time.sleep(1)
        continue  # Continua tentando caso a conexão falhe momentaneamente
    
    # Redimensiona o quadro para 640x480 para melhor desempenho
    frame_redimensionado = cv2.resize(frame, (640, 480))
    frame_processado = frame_redimensionado.copy()  # Copia para edição
    
    # Converte a imagem para escala de cinza e aplica um desfoque gaussiano
    cinza = cv2.cvtColor(frame_processado, cv2.COLOR_BGR2GRAY)
    desfoque = cv2.GaussianBlur(cinza, (9, 9), 2)
    
    # Detecta círculos na imagem utilizando a Transformada de Hough
    # 
    circulos = cv2.HoughCircles(
        desfoque,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=40,
        param1=100,
        param2=30,
        minRadius=15,
        maxRadius=60
    )
    
    # Reseta a contagem de cores antes de analisar o quadro atual
    contagem_cores = {'vermelho': 0, 'azul': 0, 'verde': 0}
    
    # Se círculos foram detectados
    if circulos is not None:
        # Converte as coordenadas e o raio para inteiros
        circulos = np.uint16(np.around(circulos))
        circulos_filtrados = []

        for circulo in circulos[0, :]:
            x, y, r = circulo
            sobreposto = False

            # Verifica se o círculo atual sobrepõe algum dos círculos já filtrados
            for fc in circulos_filtrados:
                fx, fy, fr = fc
                distancia = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
                if distancia < r + fr:
                    sobreposto = True
                    break

            if not sobreposto:
                circulos_filtrados.append((x, y, r))

        for circulo in circulos_filtrados:
            x, y, r = circulo
            
            # Criar uma máscara para isolar a área do círculo
            mascara = np.zeros_like(cinza)
            cv2.circle(mascara, (x, y), r, 255, -1)
            
            # Aplica a máscara para extrair a região do círculo
            regiao_mascarada = cv2.bitwise_and(frame_processado, frame_processado, mask=mascara)
            
            # Calcula a cor média dentro do círculo
            media_bgr = cv2.mean(regiao_mascarada, mask=mascara)[:3]
            media_bgr = tuple(map(int, media_bgr))
            
            # Determina a cor do círculo
            cor_detectada = classificar_cor(media_bgr)
            
            # Incrementa a contagem da cor detectada
            if cor_detectada in contagem_cores:
                contagem_cores[cor_detectada] += 1
                cor_borda = (0, 0, 255) if cor_detectada == 'vermelho' else (255, 0, 0) if cor_detectada == 'azul' else (0, 255, 0)
                cv2.circle(frame_processado, (x, y), r, cor_borda, 2)  # Desenha o círculo detectado
    
    # Exibir a contagem de cores detectadas na tela
    cv2.putText(frame_processado, f"Vermelho: {contagem_cores['vermelho']}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame_processado, f"Azul: {contagem_cores['azul']}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame_processado, f"Verde: {contagem_cores['verde']}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Atualiza a imagem na janela
    cv2.imshow("Detecção de Círculos", frame_processado)
    
    # Sai do loop se a tecla (q) for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()