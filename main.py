import cv2
import pytesseract

# Configuração do caminho do executável do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Nome do arquivo da imagem que você deseja reconhecer o texto
arquivo_imagem = "sua_imagem.png"

# Fator de escala para aumentar a imagem
fator_escala = 1.2 # Ajuste conforme necessário

# Carrega a imagem
imagem = cv2.imread(arquivo_imagem)

# Redimensiona a imagem aumentando o tamanho
imagem_aumentada = cv2.resize(imagem, None, fx=1.0, fy=1.5, interpolation=cv2.INTER_CUBIC)

# Converte a imagem para escala de cinza
imagem_gray = cv2.cvtColor(imagem_aumentada, cv2.COLOR_BGR2GRAY)

# Usa o Tesseract para reconhecer o texto na imagem
texto = pytesseract.image_to_string(imagem_gray, lang='eng')  # Use 'eng' para inglês ou 'por' para português

# Exibe o texto reconhecido
print("Texto encontrado:")
print(texto)
