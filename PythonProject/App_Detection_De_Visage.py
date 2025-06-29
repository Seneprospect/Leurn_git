# Importation des bibliothèques nécessaires
import cv2  # OpenCV pour la vision par ordinateur
import streamlit as st  # Streamlit pour l'interface web
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase # Pour la gestion de la webcam web
import numpy as np # Pour la manipulation des images

# Chemin RELATIF vers le classificateur en cascade pré-entraîné

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Chargement du classificateur en cascade pré-entraîné pour la détection de visages
try:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        st.error(f"Erreur: Impossible de charger le fichier XML du classificateur de visages à l'emplacement: {FACE_CASCADE_PATH}. Vérifiez si le fichier est bien présent et non corrompu.")
        st.stop() # Arrêter l'exécution si le fichier n'est pas trouvé
except Exception as e:
    st.error(f"Erreur inattendue lors du chargement du classificateur de visages: {e}")
    st.stop()


# Classe de traitement vidéo pour streamlit-webrtc
# Cette classe gérera la logique de détection de visages sur chaque frame de la webcam
class FaceDetector(VideoProcessorBase):
    def __init__(self):
        # Le classificateur est chargé une seule fois lors de l'initialisation de l'application
        self.face_cascade = face_cascade

    def recv(self, frame: VideoFrame) -> VideoFrame:
        # Convertir l'image de VideoFrame en un tableau numpy OpenCV (format BGR)
        img = frame.to_ndarray(format="bgr24")

        # Convertir l'image en niveaux de gris (meilleure performance pour la détection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détection des visages dans l'image
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1, # Réduction progressive de la taille de l’image à chaque échelle
            minNeighbors=5,  # Nombre de voisins pour valider une détection de visage
            minSize=(30, 30) # Taille minimale du visage à détecter (largeur, hauteur)
        )

        # Dessiner un rectangle vert autour de chaque visage détecté
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # (0, 255, 0) pour le vert, 2 pour l'épaisseur du trait

        # Retourner la frame traitée. streamlit-webrtc se chargera de l'afficher dans le navigateur.
        return VideoFrame.from_ndarray(img, format="bgr24")


# Interface web avec Streamlit
def app():
    st.title("Application de Détection de Visages avec Viola-Jones")
    st.write("Cette application Streamlit utilise l'algorithme Viola-Jones pour détecter les visages en temps réel via votre webcam.")
    st.warning(" Veuillez autoriser l'accès à votre webcam dans votre navigateur lorsque vous y serez invité.")

    # Utilisation de webrtc_streamer pour obtenir le flux vidéo de la webcam.
    # Le 'key' doit être unique pour chaque composant webrtc_streamer dans votre application.
    # 'video_processor_factory' pointe vers la classe qui traite les frames vidéo.
    webrtc_streamer(key="face-detection-app", video_processor_factory=FaceDetector)

    st.info("La détection démarre automatiquement une fois que vous autorisez l'accès à la webcam.")
    st.write("Les visages détectés seront encadrés par des rectangles verts.")


# Point d'entrée du programme
# S'assure que la fonction 'app()' est appelée lorsque le script est exécuté directement.
if __name__ == "__main__":
    app()

