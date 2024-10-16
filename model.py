import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import spacy
import os
dataset_path = os.getenv('DATA_PATH', 'cleaned_music_dataset.csv')
df = pd.read_csv(dataset_path)
audio_features = ['danceability', 'energy', 'valence', 'tempo', 'liveness', 'loudness', 'speechiness', 'acousticness', 'instrumentalness']

sentiment_analyzer = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[audio_features]), columns=audio_features)

df_scaled['track_id'] = df['track_id']
df_scaled['track_name'] = df['track_name']
df_scaled['artist'] = df['artist']

def perform_clustering(n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_scaled['cluster'] = kmeans.fit_predict(df_scaled[audio_features])
    return df_scaled

df_scaled = perform_clustering()

def content_based_recommendation(track_id, top_n=5):
    track_data = df_scaled[df_scaled['track_id'] == track_id]
    
    if track_data.empty:
        return None
    
    track_cluster = track_data['cluster'].values[0]
    cluster_songs = df_scaled[df_scaled['cluster'] == track_cluster]
    
    similarity = cosine_similarity([track_data[audio_features].values[0]], cluster_songs[audio_features])[0]
    
    similar_songs_idx = np.argsort(similarity)[::-1][1:top_n+1]
    similar_songs = cluster_songs.iloc[similar_songs_idx]
    
    return similar_songs[['track_name', 'artist', 'track_id']]

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
    }
    return mood_mapping.get(sentiment.upper(), 'calm')

def mood_based_recommendation(mood, top_n=5):
    mood_mapping = {
        'happy': {'valence': 0.7, 'energy': 0.6},
        'sad': {'valence': 0.2, 'energy': 0.3},
        'energetic': {'energy': 0.8, 'danceability': 0.7},
        'calm': {'acousticness': 0.8, 'valence': 0.5},
        'relaxed': {'acousticness': 0.7, 'energy': 0.4},
        'excited': {'energy': 0.9, 'danceability': 0.8},
        'nostalgic': {'valence': 0.6, 'loudness': 0.5},
        'motivated': {'energy': 0.8, 'tempo': 0.6},
        'romantic': {'valence': 0.7, 'loudness': 0.4},
        'reflective': {'acousticness': 0.6, 'valence': 0.5}
    }
    
    if mood not in mood_mapping:
        return None
    
    mood_criteria = mood_mapping[mood]
    filtered_songs = df_scaled.copy()
    
    for feature, threshold in mood_criteria.items():
        filtered_songs = filtered_songs[filtered_songs[feature] >= threshold]
    
    if filtered_songs.empty:
        return None
    
    return filtered_songs[['track_name', 'artist', 'track_id']].sample(n=top_n)

def hybrid_recommendation(user_id=None, track_id=None, mood=None, user_input=None, top_n=5):
    recommendations = pd.DataFrame()

    if user_input:
        sentiment, _ = analyze_sentiment(user_input)
        mood = map_sentiment_to_mood(sentiment) if mood is None else mood
        keywords = extract_keywords(user_input)
        print(f"Detected Mood: {mood}, Keywords: {keywords}")

    if track_id:
        cb_recommendations = content_based_recommendation(track_id, top_n)
        if cb_recommendations is not None:
            recommendations = pd.concat([recommendations, cb_recommendations], ignore_index=True)

    if mood:
        mood_recommendations = mood_based_recommendation(mood, top_n)
        if mood_recommendations is not None:
            recommendations = pd.concat([recommendations, mood_recommendations], ignore_index=True)

    recommendations = recommendations.drop_duplicates()

    if recommendations.empty:
        return None

    return recommendations
