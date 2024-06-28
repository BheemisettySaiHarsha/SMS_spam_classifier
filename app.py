import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn  # Ensure scikit-learn is installed

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("SMS Spam Classifier")

# Sample messages
sample_messages = {
    "Sample 1": "congratulations you won 1000 call on this number to get your prize",
    "Sample 2": "You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out.",
    "Sample 3": "I am free today, lets go out for a movie. What do you say?",
    "Sample 4": "A loan for £950 is approved for you if you receive this SMS. 1 min verification & cash in 1 hr at www.[redacted].co.uk to opt out reply stop",
    "Sample 5": "Did you see the match? It was insane"
}

option = st.selectbox("Choose a sample message or type your own:", [""] + list(sample_messages.keys()))
st.markdown('or')
if option:
    input_sms = sample_messages[option]
    st.write(f"**Selected Message ({option}):** {input_sms}")
else:
    input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
