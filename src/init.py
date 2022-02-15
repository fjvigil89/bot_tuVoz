from turtle import shape
from unittest import result
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn 
import tensorflow as tf
import json 
import random 
import pickle

#nltk.download('punkt')

with open("/home/frank/sites/botIA/src/content.json", encoding="utf-8") as f:
    data = json.load(f)
try:
    with open("/home/frank/sites/botIA/data/variables.packle", "rb") as archivoPickle:    
        palabras, tags, entrenamiento, salida = pickle.load(archivoPickle)
except:
    palabras =[]
    tags = []
    auxX = []
    auxY = []

    ## Lectura de los datos
    for contenido in data["contenido"]:
        for patrones in contenido["patrones"]:
            auxPalabra = nltk.word_tokenize(patrones)
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido["tag"])

            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])



    palabras = [stemmer.stem(w.lower()) for w in palabras if w != "?" ]
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    ## Creando las etiquetas del entrenamiento
    entrenamiento = []
    salida = []
    filaVacia = [0 for _ in range(len(tags))]
    for i, documnto in enumerate(auxX):
        cubeta=[]
        auxPalabra = [stemmer.stem(w.lower()) for w in documnto]
        for item in palabras:
            if item  in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        
        filaSalida = filaVacia[:]
        filaSalida[tags.index(auxY[i])]=1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

    entrenamiento = np.array(entrenamiento)
    salida = np.array(salida)
    with open("/home/frank/sites/botIA/data/variables.packle", "wb") as archivoPickle:    
        pickle.dump((palabras, tags, entrenamiento, salida), archivoPickle)

## Red Neuronal
red = tflearn.input_data(shape=[None, len(entrenamiento[0])])
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, len(salida[0]), activation="softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red)
try:
    modelo.load("/home/frank/sites/botIA/modelo/modleo.tflearn")
except:
    modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=10, show_metric=True)
    modelo.save("/home/frank/sites/botIA/modelo/modleo.tflearn")

def mainBot():
    while True:
        entrada = input("Tu: ")
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(entrada)
        entradaProcesada = [stemmer.stem(w.lower()) for w in entradaProcesada] 

        for palabraIndividual in entradaProcesada:
            for i, palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i]=1
        
        result= modelo.predict([np.array(cubeta)])
        resultIndex = np.argmax(result)

        tag = tags[resultIndex]

        for item in data["contenido"]:
            if item["tag"] == tag:
                respuesta=item["respuestas"]
        print("Bot: ",random.choice(respuesta))

mainBot()