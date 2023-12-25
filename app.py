from flask import Flask, request, render_template
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stopwords = set(stopwords.words('english'))

app = Flask(__name__)

# Load the pre-trained machine learning model and tfidf

try:
    with open('bert.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf.pkl', 'rb') as model_file:
        tfidf_vectorizer = pickle.load(model_file)
    print('model and tfidf loading complete ............')  
except FileNotFoundError:
    print("Error: Model files not found.")
except Exception as e:
    print("Error: An unexpected error occurred -", str(e))

# Function to tokenize text
def tokenize_text(text):
    tokens = word_tokenize(text)  # Tokenize the text into words
    return tokens

# function to remove punctuations
import string

def remove_punctuations(text):
    # Define a translation table to remove punctuations
    translator = str.maketrans('', '', string.punctuation)
    # Use translate() method to remove punctuations from the text
    text_without_punctuations = text.translate(translator)
    return text_without_punctuations

# Function to remove stopwords
def remove_stopwords(text):
  ''' function to remove stopwords'''
  ## make text smallcase and then remove stopwords
  text = [word.lower() for word in text if word.lower() not in stopwords]

  return text

# Function to perform lemmatization on a text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize each word
    return " ".join(lemmatized_text)

def preprocess_text(text):
    # remove punctuations
    words = remove_punctuations(text)
    # tokenize
    words = word_tokenize(words)
    # remove stopwords
    words = remove_stopwords(words)
    # Apply Lemmatization
    words = lemmatize_text(words)

    return words

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        test_me = tfidf_vectorizer.transform([preprocessed_text])

        prediction = model.predict(test_me)
        
        print('-------PREDICTION----:',prediction)
        
        if prediction == 1:
            result = 'TRUE'
        else:
            result = 'FALSE'

        return render_template('index.html', text=text, result=result)

if __name__ == '__main__':
    app.run(debug=True)
