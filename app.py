import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from better_profanity import profanity
import spacy

# NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# spaCy model
nlp = spacy.load('en_core_web_sm')

# VADER sentiment
sentiment_analyzer = VS()

# Stopwords and stemmer
stopwords = stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()

# Profanity list
custom_profanity_list = [
    "nigga", "nigger", "fuck", "shit", "bitch", "cunt", "asshole",
    "fucking", "motherfucker", "dumbass", "slut", "whore", "cock",
    "dick", "pussy", "fag", "faggot", "retard", "bastard", "twat"
]
profanity.load_censor_words(custom_profanity_list)

# Obfuscation patterns
obfuscation_patterns = [
    r'f[\W_]*u[\W_]*c[\W_]*k', r's[\W_]*h[\W_]*i[\W_]*t', r'b[\W_]*i[\W_]*t[\W_]*c[\W_]*h',
    r'n[\W_]*i[\W_]*g[\W_]*g[\W_]*a', r'p[\W_]*u[\W_]*s[\W_]*s[\W_]*y',
    r'n[\W_]*i[\W_]*g[\W_]*g[\W_]*e[\W_]*r', r'f[\W_]*a[\W_]*g', 
    r'f[\W_]*a[\W_]*g[\W_]*g[\W_]*o[\W_]*t'
]
compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in obfuscation_patterns]

# Load .pkl files
try:
    pipeline = joblib.load('pipeline_compressed.pkl')
    vectorizer = joblib.load('vectorizer_compressed.pkl')
    pos_vectorizer = joblib.load('pos_vectorizer_compressed.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer files not found! Ensure .pkl files are uploaded.")
    st.stop()

# Preprocessing functions
def preprocess(text_string):
    spaces = '\s+'
    url = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
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

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[\W_]+', '', text)
    text = re.sub(r'\s+', '', text)
    return text

def contains_profanity(text):
    normalized_text = normalize_text(text)
    return int(any(word in normalized_text for word in custom_profanity_list))

def detect_obfuscation(text):
    return int(any(pattern.search(text) for pattern in compiled_patterns))

def get_profanity_scores(text):
    return [contains_profanity(text), detect_obfuscation(text)]

def count_twitter_objs(text_string):
    spaces = '\s+'
    giant_url = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
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
    return [ner_counts.get(entity, 0) for entity in common_entity_types]

def extract_dependency_tuples(tweet):
    doc = nlp(tweet)
    dep_tuples = [(token.head.text, token.dep_, token.text) for token in doc]
    return dep_tuples

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
        twitter_objs[2], twitter_objs[1], twitter_objs[0], nsubj_count, dobj_count
    ] + ner_feats + prof_scores
    return features

def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

# Prediction function
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
    return prediction

# Class mapping
def class_to_name(class_number):
    classes = {0: "Hate", 1: "Offensive", 2: "Neither"}
    return classes.get(class_number, "Unknown")

# Streamlit UI
st.title("Hate Speech Detector with CatBoost 😊")
st.write("Tweet type chey, adi **Hate**, **Offensive**, or **Neither** ani cheptha!")

# User input
tweet_input = st.text_area("Tweet type chey:", height=100)

if st.button("Classify Cheyyi"):
    if not tweet_input.strip():
        st.error("Tweet type cheyyali, empty ga vaddu!")
    else:
        tweet_to_test = [tweet_input]
        try:
            prediction = get_tweets_predictions(tweet_to_test)
            predicted_class = class_to_name(prediction[0])
            st.success(f"**Prediction**: {predicted_class}")
            st.write(f"**Tweet**: {tweet_input}")
        except Exception as e:
            st.error(f"Error vachindi: {str(e)}")
