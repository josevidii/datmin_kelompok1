from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
import streamlit as st

# NLTK Downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# WordCloud
from wordcloud import WordCloud

# Removing common words
from nltk.corpus import stopwords

# For vectorization
from sklearn.feature_extraction.text import CountVectorizer

# For lemmatization
from nltk.stem import WordNetLemmatizer

# For text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# For splitting the dataset
from sklearn.model_selection import train_test_split

# For model building
from sklearn.linear_model import LogisticRegression

# For evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# For undersampling
from imblearn.under_sampling import RandomUnderSampler

import gensim.corpora as corpora
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import io

import pickle

# Load pre-trained models using pickle
pickle_in1 = open('logreg_model.pkl', 'rb')
logreg = pickle.load(pickle_in1)

pickle_in2 = open('lda_model.pkl', 'rb')
lda = pickle.load(pickle_in2)

pickle_in3 = open('vectorizer.pkl', 'rb')
vectorizer = pickle.load(pickle_in3)

# Sidebar for navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Prediksi Lowongan Pekerjaan", "Topik-topik di Lowongan Pekerjaan Palsu"])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r'[\^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def prediction(vec_text):
    pred = logreg.predict(vec_text)
    probabilities = logreg.predict_proba(vec_text)
    return pred, probabilities

if menu == "Prediksi Lowongan Pekerjaan":
    
    # Input fields
    st.title("Job Posting Classification")
    title = st.text_input("Job Title")
    company_profile = st.text_area("Company Profile")
    description = st.text_area("Job Description")
    requirements = st.text_area("Job Requirements")
    benefits = st.text_area("Job Benefits")
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Temporary", "Contract", "Other"])
    required_experience = st.text_input("Required Experience")
    required_education = st.text_input("Required Education")
    industry = st.text_input("Industry")
    function = st.text_input("Job Function")

    # Submit button logic
    if st.button("Submit"):      
        # Process input text
        data_list = [title, company_profile, description, requirements, benefits, employment_type, 
                    required_experience, required_education, industry, function]
        
        combined_string = " ".join(data_list)
        
        input_df = pd.DataFrame([combined_string], columns=["job_posting"])
    
        input_df['job_posting'] = input_df['job_posting'].apply(preprocess_text)
        input_df['job_posting_tokens'] = input_df['job_posting'].apply(word_tokenize)
        input_df["lemmatized_job_posting"] = input_df['job_posting_tokens'].apply(lemmatize_tokens)
        
        # Flatten the list of lists into a single list of strings
        text_flattened = [' '.join(sublist) for sublist in input_df["lemmatized_job_posting"]]

        # Vectorize the text data
        text_vec = vectorizer.transform(text_flattened)

        pred, proba = prediction(text_vec)

        # Display the results
        if (pred[0] == 0):
            result = "NOT FRAUD"
            color = "green"
        elif (pred[0] == 1):
            result = "FRAUD"
            color = "red"

        # Header untuk prediksi
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 15px; font-weight: bold; color: {color};">
                PREDICTION
            </div>
            """,
            unsafe_allow_html=True
        )

        # Hasil prediksi dengan border
        st.markdown(
            f"""
            <div style="
                text-align: center; 
                font-size: 40px; 
                font-weight: bold; 
                color: white; 
                border: 3px solid {color}; 
                border-radius: 10px; 
                padding: 20px; 
                background: {color}
                ">
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display confidence
        confidence = round(max(proba[0]) * 100, 2)

        st.markdown(
            f"""
            <div style="
                text-align: center; 
                font-size: 20px; 
                font-weight: bold; 
                color: black;">
                Confidence: <span style="color: {color};">{confidence}%</span>
            </div>
            """,
            unsafe_allow_html=True
            )

elif menu == "Topik-topik di Lowongan Pekerjaan Palsu":

    lda_df = pd.read_csv("fake_job_postings.csv")
    lda_df.fillna("", inplace=True)

    # List of columns to concatenate
    columns_to_concat = ['title', 'company_profile', 'description', 'requirements', 'benefits', 'employment_type',
                        'required_experience', 'required_education', 'industry', 'function']

    # Concatenate the values of specified columns into a new column 'job_posting'
    lda_df['job_posting'] = lda_df[columns_to_concat].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    # Create a new DataFrame with columns 'job_posting' and 'fraudulent'
    new_lda_df = lda_df[['job_posting', 'fraudulent']].copy()

    new_lda_df = new_lda_df[new_lda_df["fraudulent"] == 1]

    new_lda_df['job_posting'] = new_lda_df['job_posting'].apply(preprocess_text)

    new_lda_df['job_posting_tokens'] = new_lda_df['job_posting'].apply(word_tokenize)

    new_lda_df['lemmatized_job_posting'] = new_lda_df['job_posting_tokens'].apply(lemmatize_tokens)

    word_counts = Counter(word for words in new_lda_df["lemmatized_job_posting"] for word in words)

    # Hapus kata dengan frekuensi rendah (<= 1)
    new_lda_df["lemmatized_job_posting"] = new_lda_df["lemmatized_job_posting"].apply(
        lambda words: [word for word in words if word_counts[word] > 10]
    )

    new_lda_df["lemmatized_job_posting"] = new_lda_df["lemmatized_job_posting"].apply(
        lambda words: [word for word in words if len(word) >= 4]
    )

    # Create Corpus
    texts = new_lda_df["lemmatized_job_posting"]
    id2word = corpora.Dictionary(new_lda_df["lemmatized_job_posting"])

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    st.title("Topics in Fake Job Postings")

    # Prepare the visualization using pyLDAvis
    vis = gensimvis.prepare(lda, corpus, id2word, sort_topics=True)

    # Convert the visualization to HTML format
    html = pyLDAvis.prepared_data_to_html(vis)

    # Render the LDA visualization in Streamlit using HTML component
    st.write("### Topic Modeling Visualization:")

    html_content = f"""
        <div style="width: 100%; height: 800px; overflow: auto;">
            {html}
        </div>
        """
    
    st.components.v1.html(html_content, height=800)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda, texts=new_lda_df["lemmatized_job_posting"], dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    st.text('\n\n')   
    st.markdown(f"**Coherence Score : {coherence_lda}**") 

    text_input = st.text_area("Enter the text for topic modeling")

    if st.button("Analyze Topics"):
        if text_input.strip() == "":
            st.error("Please enter text for analysis.")
        else:
            try:
                # Assuming lda model requires tokenized text as input
                tokenized_input = word_tokenize(text_input)
                topics = lda.get_document_topics(corpora.Dictionary([tokenized_input]).doc2bow(tokenized_input))

                # Sort topics by the probability value (second item in the tuple), in descending order
                sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)

                # Get the topic with the highest probability
                highest_topic = sorted_topics[0]

                topic_id = highest_topic[0]
                topic_probability = highest_topic[1] * 100
                
                st.write("### Topics Identified:")
                st.markdown(f"**The highest probability topic is Topic {topic_id+1} with {round(topic_probability, 2)}% relevance.**")
                for i, topic_prob in enumerate(topics):
                    st.write(f"Topic {i + 1}: {round(topic_prob[1] * 100, 2)}% relevance")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
