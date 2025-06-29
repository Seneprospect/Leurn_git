import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
#=================================

#Chargement du fichier texte et prétraitement des données
with open('MonFichierText.txt', 'r', encoding='utf-8') as f :
    data = f.read().replace('\n', ' ')
# Tokeniser le texte en phrases
phrase = sent_tokenize(data)
# Définir une fonction pour prétraiter chaque phrase
def preprocess(phrase) :
    # Tokenize the sentence into words (Tokenisation de la phrase en mots)
    words = word_tokenize(phrase)
    # Suppression des mots vides et de la ponctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatisation des mots
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words
#====================

# Prétraitement de chaque phrase du texte
corpus = [preprocess(phrase) for sentence in phrase]

# Définir une fonction pour trouver la phrase la plus pertinente en fonction d'une requête
def get_most_relevant_phrase(query) :
    # Prétraitement de la requête
    query = preprocess(query)
    # Calcule la similarité entre la requête et chaque phrase du texte
    max_similarity = 0
    phrase_la_plus_pertinente = ""
    for phrase in corpus :
        similarité = len(set(query).intersection(phrase)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity :
            max_similarité = similarité
            most_relevant_phrase = " ; " .join(phrase)
    return most_relevant_phrase
    #========================================

def chatbot(question) :
    # Trouver la phrase la plus pertinente
    phrase_la_plus_pertinente = get_most_relevant_sentence(question)
    # Retourne la réponse
    return most_relevant_sentence
    #==================================

# Créer une application Streamlit
def main() :
    st.title("LE TRAITEMENT DES DONNÉES" )
    st.write("Bonjour ! Je suis un chatbot. Pour le traistement des données,demandez-moi n'importe quoi sur le sujet dans le fichier texte." )
    # Obtenir la question de l'utilisateur
    question = st.text_input("You:")
    # Créer un bouton pour soumettre la question
    if st.button("Submit" ):
        # Appeler la fonction chatbot avec la question et afficher la réponse
        response = chatbot(question)
        st.write("Chatbot : " + response)
if __name__ == "__main__" :
    main()