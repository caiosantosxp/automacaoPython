import pyautogui
import cv2
import time

# Nome do programa que você deseja monitorar
titulo_programa = "Título da Janela do Programa"

# Nome do arquivo da imagem do objeto que você deseja reconhecer
arquivo_objeto = "objeto.jpg"

# Loop para verificar continuamente se o programa está aberto
while True:
    # Captura a tela do programa com o título específico
    janela = pyautogui.getWindowsWithTitle(titulo_programa)[0]
    left, top, width, height = janela.left, janela.top, janela.width, janela.height

    # Captura a tela da área específica
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot.save('screenshot.png')

    # Carrega a imagem do objeto que você deseja reconhecer
    objeto = cv2.imread(arquivo_objeto, cv2.IMREAD_GRAYSCALE)

    # Carrega a imagem da tela capturada
    tela = cv2.imread('screenshot.png', cv2.IMREAD_GRAYSCALE)

    # Realiza a correspondência entre a imagem do objeto e a imagem da tela
    orb = cv2.ORB_create()
    kp_objeto, des_objeto = orb.detectAndCompute(objeto, None)
    kp_tela, des_tela = orb.detectAndCompute(tela, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_objeto, des_tela)
    matches = sorted(matches, key=lambda x: x.distance)

    # Se houver pelo menos uma correspondência suficiente
    if len(matches) > 10:
        print("Objeto encontrado!")

        # Obter coordenadas do objeto correspondido
        objeto_pts = np.float32([kp_objeto[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        tela_pts = np.float32([kp_tela[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Desenha um retângulo vermelho ao redor do objeto encontrado
        H, _ = cv2.findHomography(objeto_pts, tela_pts, cv2.RANSAC, 5.0)
        objeto_corners = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
        tela_corners = cv2.perspectiveTransform(objeto_corners, H)
        cv2.polylines(tela, [np.int32(tela_corners)], True, (0, 0, 255), 2)

        # Adicione seu código para notificar o usuário ou realizar outras ações necessárias aqui

    # Exibe a imagem com o retângulo desenhado
    cv2.imshow("Objeto Encontrado", tela)

    # Aguarde por alguns segundos antes de verificar novamente (por exemplo, 5 segundos)
    time.sleep(5)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha todas as janelas
cv2.destroyAllWindows()
