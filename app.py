import flask
from flask import Flask, render_template, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField
import scipy
import pickle as pkl
from sentence_transformers import SentenceTransformer

def return_prediction(model,corpus,corpus_embeddings,sample_json):
    e = []
    query_json = sample_json['form_query']
    query = [query_json]
    query_embeddings = model.encode(query)
    number_of_matches = 5
    for query, query_embedding in zip(query, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
    
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda X: X[1])
    
        #print("You are looking for: ",query)
        #print("Top Search matches:")
        
        for id, distance in results[0:number_of_matches]:
             e.append(corpus[id].strip()+ "(Cosine function: %.4f)" % (1-distance))
    print ('\n\n'.join(e))
    return e

app = Flask(__name__)

app.config['SECRET_KEY'] = 'someRandomKey'



sentence_model = SentenceTransformer('search')

with open("embeddings.pkl", "rb") as in_file:
    corpus_embeddings1 = pkl.load(in_file)

with open("corpus.pkl", "rb") as input_file:
    corpus1 = pkl.load(input_file)

class SearchForm(FlaskForm):
    
    form_query = TextField('Query')
    
    submit = SubmitField('Search')
    

@app.route('/', methods=['GET', 'POST'])

def index():
    
    form = SearchForm()
    
    
    if form.validate_on_submit():
        
        session['formquery'] = form.form_query.data
        
        return redirect(url_for("prediction"))
    
    return render_template('home.html', form=form)

@app.route('/prediction')

def prediction():
    
    content = {}
    
    content['form_query'] = str(session['formquery'])
    
    results = return_prediction(model=sentence_model, corpus=corpus1, corpus_embeddings=corpus_embeddings1, sample_json=content)
    
    results1 = '<br><br>'.join(results)
    
    return render_template('prediction.html', results=results1)

if __name__ == '__main__':
    app.run(debug=True)