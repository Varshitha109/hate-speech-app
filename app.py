import streamlit as st
import pickle
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textstat.textstat import textstat
from better_profanity import profanity
import spacy


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

import nltk
import os

# Download the required NLTK data
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Define stopwords and stemmer
stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()

# Define preprocess and tokenize functions
def preprocess(text_string):
    spaces = '\s+'
    url = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
           '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention = '@[\w\-]+'
    text = re.sub(spaces, ' ', text_string)
    text = re.sub(url, '', text)
    text = re.sub(mention, '', text)
    return text

def tokenize(tweet):
    tweet = " ".join(re.split("[^a-zA-Z]+", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split() if t]
    return tokens

def basic_tokenize(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    return tweet.split()

# Load model and vectorizers
pipeline = pickle.load(open('pipeline.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
pos_vectorizer = pickle.load(open('pos_vectorizer.pkl', 'rb'))

# Rest of your functions
def count_twitter_objs(text_string):
    spaces = '\s+'
    giant_url = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                 '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention = '@[\w\-]+'
    hashtag = '#[\w\-]+'
    text = re.sub(spaces, ' ', text_string)
    text = re.sub(giant_url, 'URLHERE', text)
    text = re.sub(mention, 'MENTIONHERE', text)
    text = re.sub(hashtag, 'HASHTAGHERE', text)
    return (text.count('URLHERE'), text.count('MENTIONHERE'), text.count('HASHTAGHERE'))

def ner_features(tweet):
    doc = nlp(tweet)
    ner_counts = {}
    for ent in doc.ents:
        if ent.label_ in ner_counts:
            ner_counts[ent.label_] += 1
        else:
            ner_counts[ent.label_] = 1
    common_entity_types = ['PERSON', 'NORP', 'FAC', 'ORG', 'LOC', 'DATE']
    features = [ner_counts.get(entity, 0) for entity in common_entity_types]
    return features

def extract_dependency_tuples(tweet):
    doc = nlp(tweet)
    dep_tuples = [(token.head.text, token.dep_, token.text) for token in doc]
    return dep_tuples

sentiment_analyzer = SentimentIntensityAnalyzer()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[\W_]+', '', text)
    text = re.sub(r'\s+', '', text)
    return text

custom_profanity_list = [
    "nigga", "nigger", "fuck", "shit", "bitch", "cunt", "asshole",
    "fucking", "motherfucker", "dumbass", "slut", "whore", "cock",
    "dick", "pussy", "fag", "faggot", "retard", "bastard", "twat"
]
profanity.load_censor_words(custom_profanity_list)

obfuscation_patterns = [
    r'f[\W_]*u[\W_]*c[\W_]*k',
    r's[\W_]*h[\W_]*i[\W_]*t',
    r'b[\W_]*i[\W_]*t[\W_]*c[\W_]*h',
    r'n[\W_]*i[\W_]*g[\W_]*g[\W_]*a',
    r'p[\W_]*u[\W_]*s[\W_]*s[\W_]*y',
    r'n[\W_]*i[\W_]*g[\W_]*g[\W_]*e[\W_]*r',
    r'f[\W_]*a[\W_]*g',
    r'f[\W_]*a[\W_]*g[\W_]*g[\W_]*o[\W_]*t'
]
compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in obfuscation_patterns]

def contains_profanity(text):
    normalized_text = normalize_text(text)
    return int(any(word in normalized_text for word in custom_profanity_list))

def detect_obfuscation(text):
    return int(any(pattern.search(text) for pattern in compiled_patterns))

def get_profanity_scores(text):
    return [contains_profanity(text), detect_obfuscation(text)]

def other_features(tweet):
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    words = preprocess(tweet)
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))
    dep_tuples = extract_dependency_tuples(tweet)
    dep_types = [dep[1] for dep in dep_tuples]
    dep_count = {dep: dep_types.count(dep) for dep in set(dep_types)}
    nsubj_count = dep_count.get('nsubj', 0)
    dobj_count = dep_count.get('dobj', 0)
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)
    twitter_objs = count_twitter_objs(tweet)
    ner_feats = ner_features(tweet)
    prof_scores = get_profanity_scores(tweet)
    features = [
        FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
        num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
        twitter_objs[2], twitter_objs[1], twitter_objs[0],
        nsubj_count, dobj_count
    ] + ner_feats + prof_scores
    return features

def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

def class_to_name(class_number):
    classes = {0: "Hate", 1: "Offensive", 2: "Neither"}
    return classes.get(class_number, "Unknown")

def get_tweets_predictions(tweets):
    tfidf_tweets = vectorizer.transform(tweets).toarray()
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    pos_tags = pos_vectorizer.transform(pd.Series(tweet_tags)).toarray()
    other_feats = get_feature_array(tweets)
    tweet_features = np.concatenate([tfidf_tweets, pos_tags, other_feats], axis=1)
    prediction = pipeline.predict(tweet_features)
    probabilities = pipeline.predict_proba(tweet_features)  # Get probabilities
    return prediction, probabilities

# Streamlit interface
st.title("Hate Speech Detection App")
st.write("Enter a tweet below to classify it as Hate, Offensive, or Neither.")

# Text input box
user_input = st.text_area("Enter Tweet", height=150)

# Predict button
if st.button("Classify"):
    if user_input:
        tweet_to_test = [user_input]
        prediction, probs = get_tweets_predictions(tweet_to_test)
        result = class_to_name(prediction[0])
        st.success(f"Predicted Class: {result}")
        st.write(f"Confidence Scores:")
        st.write(f"Hate: {probs[0][0]:.2f}")
        st.write(f"Offensive: {probs[0][1]:.2f}")
        st.write(f"Neither: {probs[0][2]:.2f}")
    else:
        st.warning("Please enter a tweet!")
