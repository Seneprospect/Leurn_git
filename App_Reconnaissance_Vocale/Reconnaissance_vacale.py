# Importation de la bibliothèque Streamlit (pour créer l'application web interactive)
import streamlit as st

# Importation de la bibliothèque SpeechRecognition (pour faire la reconnaissance vocale)
import speech_recognition as sr

# Définition de la fonction pour effectuer la transcription vocale
def transcribe_speech():
    # Initialisation de l'objet de reconnaissance vocale
    r = sr.Recognizer()

    # Utilisation du microphone comme source audio
    with sr.Microphone() as source:
        # Affichage d'un message d'information dans l'application Streamlit
        st.info("Parlez maintenant...")  # Indique à l'utilisateur qu'il peut commencer à parler

        # Écoute et enregistrement de l'audio du microphone
        audio_text = r.listen(source)

        # Message pour indiquer que la transcription est en cours
        st.info("Transcription en cours...")

        try:
            # Utilisation de l'API de Google pour convertir l'audio en texte
            # La langue par défaut est l'anglais. On peut spécifier `language='fr-FR'` pour le français si nécessaire
            text = r.recognize_google(audio_text, language='fr-FR')
            return text

        # En cas d'erreur (ex. : aucun son détecté ou problème de connexion)
        except:
            return "Désolé, je n'ai pas compris."

# Fonction principale qui définit l’interface Streamlit
def main():
    # Titre de l’application affiché sur la page
    st.title("Application de Reconnaissance Vocale")

    # Sous-titre ou explication pour l'utilisateur
    st.write("Cliquez sur le micro pour commencer à parler :")

    # Bouton pour démarrer la reconnaissance vocale
    if st.button("Démarrer l'enregistrement"):
        text = transcribe_speech()  # Appel de la fonction de transcription
        st.write("Transcription :", text)  # Affichage du texte reconnu

# Point d’entrée du programme : exécute la fonction main si ce fichier est le principal
if __name__ == "__main__":
    main()
