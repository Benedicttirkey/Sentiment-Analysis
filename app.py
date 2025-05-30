
from flask import Flask, request, jsonify, render_template
import pickle

# Load the model and vectorizer
with open(r'LR_Tained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open(r'vector.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)




app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Get JSON data from the request
    #data = request.get_json()

    user_input = request.form['text']  # Expecting 'text' as the key in the JSON body

    # Preprocess and vectorize the input
    input_vector = vectorizer.transform([user_input])

    # Make prediction
    prediction = model.predict(input_vector)

    # Return the result as JSON
    #return jsonify({'prediction': prediction[0]})

    
    if prediction[0]==1:
        sentiment="Positive"
    else:
        sentiment="Negative"


    # Return the result back to the HTML template
    return render_template('index.html',word=user_input , prediction=sentiment)



if __name__ == '__main__':
    app.run( debug=True)



