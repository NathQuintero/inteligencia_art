# streamlit_audio_recorder y whisper by Alfredo Diaz - version Mayo 2024

# En VsC seleccione la version de Python (recomiendo 3.9) 
#CTRL SHIFT P  para crear el enviroment (Escriba Python Create Enviroment) y luego venv 

#o puede usar el siguiente comando en el shell
#Vaya a "view" en el menú y luego a terminal y lance un terminal.
#python -m venv env

#Verifique que el terminal inicio con el enviroment o en la carpeta del proyecto active el env.
#cd D:\flores\env\Scripts\
#.\activate 

#Debe quedar asi: (.venv) D:\proyectos_ia\Flores>

#Puedes verificar que no tenga ningun libreria preinstalada con
#pip freeze
#Actualicie pip con pip install --upgrade pip

#pip install tensorflow==2.15 La que tiene instalada Google Colab o con la versión qu fué entrenado el modelo
#Verifique se se instaló numpy, no trate de instalar numpy con pip install numpy, que puede instalar una version diferente
#pip install streamlit
#Verifique se se instaló no trante de instalar con pip install pillow
#Esta instalacion se hace si la requiere pip install opencv-python

#Descargue una foto de una flor que le sirva de ícono 

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import warnings
from gtts import gTTS
import base64

warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(
    page_title="¿Qué producto es?",
    page_icon="icono.ico",
    initial_sidebar_state='auto',
    menu_items={
        'Report a bug': 'http://www.unab.edu.co',
        'Get Help': "https://docs.streamlit.io/get-started/fundamentals/main-concepts",
        'About': "Nathalia Quintero & Angelly Cristancho. Inteligencia Artificial *Ejemplo de clase* Ingeniería de sistemas!"
    }
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'productosa.h5')
    model = tf.keras.models.load_model(model_path)
    return model

with st.spinner('Modelo está cargando..'):
    model = load_model()
    
# Generar saludo
def generar_saludo():
    texto = "¡Hola! soy Beimax, tu asistente neuronal personal, ¿cómo te sientes hoy?"
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def reproducir_audio(mp3_fp):
    try:
        audio_bytes = mp3_fp.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al reproducir el audio: {e}")

# Reproducir el saludo al inicio
mp3_fp = generar_saludo()
reproducir_audio(mp3_fp)

with st.sidebar:

    confianza = st.slider("Seleccione el nivel de Confianza", 0, 100, 50) / 100
    st.markdown("¿Cómo poner el producto correctamente en la cámara?") 

    # Ruta del archivo de video
    video_file_path = './videos/SI.mp4'
    try:
        with open(video_file_path, 'rb') as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except FileNotFoundError:
        st.error(f"El archivo de video no se encontró en la ruta: {video_file_path}")

    # Ruta del archivo de video
    video_file_path = './videos/NO.mp4'
    try:
        with open(video_file_path, 'rb') as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except FileNotFoundError:
        st.error(f"El archivo de video no se encontró en la ruta: {video_file_path}")
        
tab1, tab2 = st.tabs(["Prediccion de productos", "Introducción"])

with tab1:

    # Título de la página
    st.image("./videos/banner.png", use_column_width=True)
    st.write("# Detección de Productos")

    def import_and_predict(image_data, model, class_names):
        if image_data.mode != 'RGB':
            image_data = image_data.convert('RGB')
            
        image_data = image_data.resize((180, 180))
        image = tf.keras.utils.img_to_array(image_data)
        image = tf.expand_dims(image, 0)  # Create a batch
        prediction = model.predict(image)
        index = np.argmax(prediction)
        score = tf.nn.softmax(prediction[0])
        class_name = class_names[index].strip()
        return class_name, score

    def generar_audio(texto):
        tts = gTTS(text=texto, lang='es')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp

    def reproducir_audio(mp3_fp):
        audio_bytes = mp3_fp.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)

    class_names = open("./clases (1).txt", "r").readlines()

    option = st.selectbox(
        "¿Qué te gustaría usar para subir la foto?",
        ("Tomar foto", "Subir archivo", "URL"),
        index=None,
        placeholder="Selecciona cómo subir la foto"
    )


    img_file_buffer = None

    if option == "Tomar foto":
        # Reproducir el audio según la opción seleccionada
        # Opción para capturar una imagen desde la cámara
        img_file_buffer = st.camera_input("Capture una foto para identificar el producto")

    elif option == "Subir archivo":
        # Opción para cargar una imagen desde un archivo local
        if img_file_buffer is None:
            img_file_buffer = st.file_uploader("Cargar imagen desde archivo", type=["jpg", "jpeg", "png"])

    elif option == "URL":
        # Opción para cargar una imagen desde una URL
        if img_file_buffer is None:
            image_url = st.text_input("O ingrese la URL de la imagen")
            if image_url:
                try:
                    response = requests.get(image_url)
                    img_file_buffer = BytesIO(response.content)
                except Exception as e:
                    st.error(f"Error al cargar la imagen desde la URL: {e}")


    # Procesar la imagen y realizar la predicción
    if img_file_buffer:
        try:
            image = Image.open(img_file_buffer)
            st.image(image, use_column_width=True)

            # Realizar la predicción
            class_name, score = import_and_predict(image, model, class_names)
            max_score = np.max(score)

            # Mostrar el resultado y generar audio
            if max_score > confianza:
                resultado = f"Tipo de Producto: {class_name}\nPuntuación de confianza: {100 * max_score:.2f}%"
                st.subheader(f"Tipo de Producto: {class_name}")
                st.text(f"Puntuación de confianza: {100 * max_score:.2f}%")
            else:
                resultado = "No se pudo determinar el tipo de producto"
                st.text(resultado)

            # Generar y reproducir el audio
            mp3_fp = generar_audio(resultado)
            reproducir_audio(mp3_fp)
            
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")
    else:
        st.text("Por favor, cargue una imagen usando una de las opciones anteriores.")
        
with tab2:
    with st.container( border=True):
        st.subheader("Asistente de Identificación de Productos para Personas con Baja Visión mediante Redes Neuronales")
        #Se desea usar emoji lo puedes buscar aqui.
        st.write("""Realizado por Keren Nathalia Quintero &
                    Angely Gabriela Cristancho:\U0001F33B\U0001F42C:""")
        st.write("""

Propósito del Proyecto
El objetivo principal de este proyecto es asistir a personas con baja visión en la identificación de productos de manera autónoma y eficiente. Muchas veces, las personas con discapacidad visual se enfrentan a desafíos significativos al intentar reconocer productos en su entorno diario. Este proyecto utiliza tecnología avanzada de reconocimiento de imágenes para proporcionar una solución práctica y accesible a este problema.

Descripción del Proyecto
Este sistema está diseñado para funcionar como un asistente personal que puede identificar productos a través de una cámara. La aplicación permite a los usuarios tomar una foto del producto, subir una imagen desde su dispositivo o ingresar una URL de la imagen del producto. Una vez que la imagen es proporcionada, el sistema utiliza un modelo de predicción basado en redes neuronales para identificar el producto y proporcionar una descripción de este.

Cómo Funciona
Interfaz de Usuario:

La aplicación ofrece una interfaz sencilla e intuitiva donde los usuarios pueden seleccionar cómo desean subir la imagen del producto.
Se proporcionan opciones para tomar una foto con la cámara, cargar una imagen desde un archivo local o ingresar la URL de una imagen.
Procesamiento de la Imagen:

Una vez que la imagen es subida, el sistema la procesa y ajusta para que sea compatible con el modelo de predicción.
La imagen es convertida a formato RGB y redimensionada a las dimensiones requeridas por el modelo.

Modelo de Predicción:

Utilizamos un modelo de predicción de productos basado en redes neuronales profundas. Este modelo ha sido entrenado para reconocer una variedad de productos con alta precisión.
El modelo analiza la imagen y proporciona una predicción del tipo de producto, junto con una puntuación de confianza.

Retroalimentación Auditiva:

Además de mostrar los resultados en pantalla, el sistema genera un mensaje de audio que describe el producto identificado. Esta característica es crucial para usuarios con discapacidad visual, ya que les proporciona información auditiva sobre el producto.
El sistema también proporciona retroalimentación auditiva cuando se selecciona una opción de entrada (como tomar una foto o subir un archivo), asegurando una experiencia de usuario accesible y fluida.

Beneficios
Accesibilidad: Este sistema mejora significativamente la accesibilidad para personas con baja visión, permitiéndoles identificar productos de manera independiente.

Conveniencia: La posibilidad de utilizar diferentes métodos para subir imágenes ofrece flexibilidad y comodidad a los usuarios.

Tecnología Avanzada: La utilización de redes neuronales profundas para el reconocimiento de imágenes garantiza un alto nivel de precisión en la identificación de productos.

Interfaz Amigable: La interfaz simple y las instrucciones claras hacen que la aplicación sea fácil de usar para personas de todas las edades y habilidades tecnológicas.

""")
    with st.container(border=True,height=250):
        st.subheader("Detalles")
        st.write(""" Este es un ejemplo de despliegue de los modelos de Machine Learning entrenados en
                Google Colab con las librerias de scikit-learn par Naive Bayes, Arbles de Decisión y Bosques Aleatorios.
                En este notebook podrás verificar el preprocesamiento del dataset y el entrenamiento y las pruebas
                y scores obtenidos.
                https://colab.research.google.com/drive/1Lth_RqbnAnBVAMjSWinpXoTitI9OPaIv""")

