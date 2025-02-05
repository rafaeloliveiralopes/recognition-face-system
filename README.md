# Projeto de Detecção Facial em Python com OpenCV e Google Colab

O objetivo principal deste projeto é trabalhar com as bibliotecas e frameworks estudados e analisados nas aulas do Bootcamp Machine Learning Pratctioner. Neste sentido, a proposta padrão envolve um sistema de detecção e reconhecimento de faces, utilizando o framework TensorFlow em conjuntos com as bibliotecas que o projetista julgue necessárias, de forma ilimitada.

Este repositório contém um exemplo completo de como capturar imagens da webcam em uma sessão do Google Colab, utilizar o modelo pré-treinado de detecção de rostos baseado em **Deep Learning** (rede Caffe) e exibir as detecções resultantes em uma imagem.

## Entregáveis
- [x] Utilizar uma rede de detecção treinada para detectar faces.
- [ ] Utilizar uma rede de classificação para classificar a face detectada.


## Sumário
1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Tecnologias Utilizadas](#tecnologias-utilizadas)
3. [Como Executar](#como-executar)
4. [Explicação do Código](#explicação-do-código)
   - [Importações](#importações)
   - [Função de Captura de Imagem com JavaScript](#função-de-captura-de-imagem-com-javascript)
   - [Captura da Imagem e Exibição](#captura-da-imagem-e-exibição)
   - [Download do Modelo Pré-Treinado](#download-do-modelo-pré-treinado)
   - [Carregamento do Modelo e Preparação da Imagem](#carregamento-do-modelo-e-preparação-da-imagem)
   - [Detecção e Marcação de Rostos](#detecção-e-marcação-de-rostos)
5. [Resultado](#resultado)
6. [Contatos](#contatos)


---

## Visão Geral do Projeto
Este projeto ilustra como usar a webcam de um computador ou notebook em uma sessão do Google Colab, capturar uma imagem, redimensioná-la e aplicar um modelo de **detecção facial**. O modelo utilizado é baseado em um **SSD (Single Shot Multibox Detector)** pré-treinado em Caffe, fornecido pela biblioteca OpenCV.

## Tecnologias Utilizadas
- [Python 3](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/) (biblioteca principal para tratamento de imagens)
- [imutils](https://pypi.org/project/imutils/) (auxiliares para manipulação de imagens e vídeos)
- [Google Colab](https://colab.research.google.com/) (execução de notebooks na nuvem)
- [Javascript](https://developer.mozilla.org/pt-BR/docs/Web/JavaScript) (para capturar a imagem da webcam no Colab)

## Como Executar
1. **Clone este repositório** ou copie o conteúdo do notebook para o seu ambiente de desenvolvimento.  
2. **Abra o notebook no Google Colab** (ou em um ambiente que suporte Python e tenha acesso à webcam).  
3. **Instale as dependências** listadas em [Tecnologias Utilizadas](#tecnologias-utilizadas) (se necessário).  
4. **Execute as células** do notebook em sequência, seguindo as instruções comentadas em cada etapa do código.

## Explicação do Código

### Importações
No topo do arquivo, são importadas as bibliotecas necessárias:

```python
import imutils
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
```

### Função de Captura de Imagem com JavaScript
Cria-se uma função Python que injeta código **JavaScript** na página, permitindo acessar a webcam, exibir o vídeo ao usuário e capturar um frame quando o usuário clicar no botão **Capture**:

```python
def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
  ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename
```

### Captura da Imagem e Exibição
Chamamos a função `take_photo`, que solicita acesso à webcam, e armazenamos o arquivo de imagem na variável `image_file`. Em seguida, a imagem é carregada pelo OpenCV, redimensionada e exibida.

```python
image_file = take_photo()
image = cv2.imread(image_file)
image = imutils.resize(image, width=400)
(h, w) = image.shape[:2]
print(w, h)
cv2_imshow(image)
```

### Download do Modelo Pré-Treinado
Baixamos dois arquivos necessários para detecção de rostos, contendo a definição de rede (`deploy.prototxt`) e os pesos aprendidos (`res10_300x300_ssd_iter_140000.caffemodel`):

```bash
!wget -N https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
!wget -N https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

### Carregamento do Modelo e Preparação da Imagem
Após o download, carregamos a rede e criamos um *blob* para ser fornecido à rede neural. O *blob* é normalizado e redimensionado para `300x300`:

```python
print("[INFO] loading model...")
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

image = imutils.resize(image, width=400)
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0)
)
```

### Detecção e Marcação de Rostos
Enviamos o *blob* à rede (`net.setInput(blob)`) e executamos a inferência para detectar rostos. Em seguida, desenhamos retângulos e porcentagens de confiança em cada rosto identificado.

```python
print("[INFO] computando detecção de objetos...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Consideramos apenas detecções com confiança maior que 50%
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2_imshow(image)
```

## Resultado
Ao executar este notebook, o modelo detectará rostos na imagem capturada e desenhará um retângulo ao redor do rosto, juntamente com a porcentagem de confiança. Por exemplo, um rosto pode ser identificado com **99% de certeza**, o que reflete a boa acurácia do modelo para detecção facial em 2D.

## Considerações
Embora o modelo detecte rostos com alta confiança, ele **não consegue diferenciar um rosto real de uma fotografia ou imagem**. Para aumentar a segurança e garantir que o rosto seja de uma pessoa real (evitando ataques de apresentação com fotos ou vídeos), recomenda-se a implementação de algoritmos de **liveness detection**, tais como:
- Análise de profundidade ou **3D**;
- Verificações de piscadas, movimentos naturais ou microexpressões;
- Detecção de textura de pele, uso de infravermelho, entre outras técnicas avançadas.

## Contato
Dúvidas ou sugestões? Entre em contato:

- **Email:** rafaellopes.dev@gmail.com
- **LinkedIn:** [Rafael Lopes](https://www.linkedin.com/in/rafael-lopes-desenvolvedor-fullstack/)
