import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import pickle

nltk.download("wordnet")

custom_patterns = [r"a+w+" , r"w+p+" , r"u+h+" , r"w+" , r"im"]

custom_words = {
    "fuck": ["fucking" , "fuckin" , "f*$%-ing"] ,
    "dick" : ["dihck"],
    "nigga" : ["nihgaa", "nigger"],
    "ass" : ["arse"] #6170 obscene
}


wnl = WordNetLemmatizer()

stop_words  = set(stopwords.words("english"))



def clean_sentence(sentence):
    sentence = sentence.lower()

    sentence = re.sub(r"\n"," ",sentence)
    sentence = re.sub(r"[( )]"," ",sentence)
    
    words = [word for word in sentence.split() if len(word)> 1]
    
    for i in range(len(words)):
        
        words[i] = wnl.lemmatize(words[i])
    sentence = " ".join(words)
    
    sentence = decontracted(sentence)

    for pattern in custom_patterns:
        sentence = re.sub(pattern, '', sentence)

    for word,variant in custom_words.items():
        for var in variant:
            sentence = str(sentence).replace(var,word)
            

    sentence = re.sub(r"[^a-zA-Z]"," ",sentence)
    
    sentence = re.sub("([^\x00-\x7F])+"," ",sentence)

    sentence = re.sub(r"\s+"," ",sentence)
    
    sentence = sentence.strip()

    # words = sentence.split()
    

    return sentence


def decontracted(phrase):
    # specific
    phrase = [word for word in phrase.split() if word not in stop_words]
    
    phrase = " ".join(phrase)
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"weren\'t" , "were not",phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    return phrase

def read_pickle(pickle_file):
    with open(pickle_file,"rb") as file:
        instance = pickle.load(file)
    return instance

def read_model(model_path):
    model = load_model(model_path)
    return model

