from transformers import pipeline
import spacy

sentiment_analyzer = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")

def analyze_sentiment(user_input):
    result = sentiment_analyzer(user_input)
    return result[0]['label'], result[0]['score']

def extract_keywords(user_input):
    doc = nlp(user_input)
    return [ent.text for ent in doc.ents if ent.label_ in ['EVENT', 'EMOTION', 'ORG', 'PERSON']]

def map_sentiment_to_mood(sentiment):
    mood_mapping = {
        'POSITIVE': 'happy',
        'NEGATIVE': 'sad',
        'NEUTRAL': 'calm',
        'MIXED': 'conflicted',
        'EXTREMELY POSITIVE': 'excited',
        'EXTREMELY NEGATIVE': 'angry',
        'VERY POSITIVE': 'joyful',
        'VERY NEGATIVE': 'depressed',
    }
    return mood_mapping.get(sentiment.upper(), 'calm')
