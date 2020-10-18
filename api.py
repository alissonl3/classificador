# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:37:53 2020

@author: ALISSON
"""

import nltk

# Importanto bibliotecas da API
from flask import request, json, Response
from flask_api import FlaskAPI
from dotenv import load_dotenv
from os.path import dirname

from nltk.stem import RSLPStemmer

import ast

# Importanto o salvamento do classificador
import pickle

# stop words
# Realiza o download da lista de stopwords do nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('portuguese')

# Carrega o modelo treinado
#filename = 'best_svm_treinado.sav'
#filenameTfIdf = 'best_tfidf_treinado.sav'
filename = 'melhor_classificador.sav'
filenameTfIdf = 'melhor_tfidf.sav'
classificador_load = pickle.load(open(filename, 'rb'))
tfidf_load = pickle.load(open(filenameTfIdf, 'rb'))


dotenv_path = str.join(dirname(__file__), '.env')  # Path to .env file
load_dotenv(dotenv_path)

app = FlaskAPI(__name__);

# stemizacao
# Processo no qual relativiza todas as palavras para um único radical cortando seu sufxo.
# Por exemplo: correr, corrida -> corr.
def stemizaComentario(comentario):
    stemmer = RSLPStemmer()
    result = []
    comentario = str(comentario).split(' ')
    for word in comentario:
        if word :
            result.append(stemmer.stem(word))       
    return " ".join(result)


def stemizaComentarioTeste(comentario):
    stemmer = RSLPStemmer()
    result = []
    for word in comentario:
        result.append(stemmer.stem(word.lower()))
    return " ".join(result)


def trataRetorno(comentarios, result):
    retorno = []
    for index in range(len(comentarios)):
        dado = { "comentario": comentarios[index], "classificacao": result[index] }
        retorno.append(dado)
    return retorno;     



@app.route('/classificar', methods=['POST'])
def classificaComentario():
    data = request.get_data(as_text=True)
    
    comentarios = ast.literal_eval(data)
    
    #print('antes:\n')
    #print(stemizaComentario('tem ótimos professores correndo correr'))
    
    comentariosTratados = list(map(stemizaComentario,comentarios))
    
    print('novos:\n')
    print(comentariosTratados)
    
    text_vect_uni_idf = tfidf_load.transform(comentariosTratados)    
    result = classificador_load.predict(text_vect_uni_idf); 
    
    print('Retorno')
    print(trataRetorno(comentarios, result))

    
    #content = {'result': str(result)}
    content = {'result': trataRetorno(comentarios, result)}
    print(content)
    data = json.dumps(content, ensure_ascii=False);
    return Response(data, status=200, mimetype='application/json');


app.run(debug=False)