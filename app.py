import numpy as np
import pickle
import regex as re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)
print('Model loaded. Check http://127.0.0.1:5000/home')

def model_predict(q):
    regex = "[^A-Za-z0-9\s]" 
    stemmer = PorterStemmer()
    a=re.sub(regex, "",q)
    a=a.lower()
    l2=a.split(" ")
    l3=[]
    l=[]
    for i in l2:
        l3.append(stemmer.stem(i))
    l.append(' '.join(l3))  
    vectorizer=pickle.load(open('vectorizer.pkl','rb'))
    X = vectorizer.transform(l)
    X=X.toarray()
    model=pickle.load(open('naive.pkl','rb'))
    prediction=model.predict(X)
    return prediction[0]
    
@app.route('/home', methods=['GET'])
def index():
    return render_template('main.html')
@app.route("/prediction.html")
def know():
    return render_template('prediction.html')
@app.route("/Aboutus.html")
def know2():
    return render_template('Aboutus.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("hi")
    if request.method == 'POST':
        q=request.form.get("mail")
        preds = model_predict(q)
        if(preds==0):
            return render_template('result.html',data='ham')
        else:
            return render_template('result.html',data='spam')
        
    return None


if __name__ == '__main__':
    app.run(debug=True)
    