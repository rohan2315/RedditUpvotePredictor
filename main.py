from flask import Flask, render_template, url_for, request, jsonify      
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import numpy as np

app = Flask(__name__)

# Load the TF-IDF vocabulary specific to the category
with open(r"score_vect.pkl", "rb") as f:
    tox = pickle.load(f)

# Load the pickled RDF models
with open(r"score_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

# Render the HTML file for the home page
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    # Take a string input from user
    user_input = request.form['text']
    data = [user_input]

    vect = tox.transform(data)
    pred_tox = score_model.predict_proba(vect)[:,1]


    out_tox = round(pred_tox[0], 2)


    print(out_tox)

    return render_template('index.html', 
                            pred_ide = 'Upvote numbers: {}'.format(out_tox)                        
                            )
     
# Server reloads itself if code changes so no need to keep restarting:
if __name__ == '__main__':
	app.run(host="127.0.0.1",port=8080,debug=True)