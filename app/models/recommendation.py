import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendation(df_scaled, track_id, top_n=5):
    track_data = df_scaled[df_scaled['track_id'] == track_id]
    
    if track_data.empty:
        return None
    
    track_cluster = track_data['cluster'].values[0]
    cluster_songs = df_scaled[df_scaled['cluster'] == track_cluster]
    
    similarity = cosine_similarity([track_data.iloc[0].values], cluster_songs.iloc[:, :-3].values)[0]
    
    similar_songs_idx = np.argsort(similarity)[::-1][1:top_n+1]
    similar_songs = cluster_songs.iloc[similar_songs_idx]
    
    return similar_songs[['track_name', 'artist', 'track_id']]

def mood_based_recommendation(df_scaled, mood, mood_mapping, top_n=5):
    mood_criteria = mood_mapping.get(mood, {})
    filtered_songs = df_scaled.copy()
    
    for feature, threshold in mood_criteria.items():
        filtered_songs = filtered_songs[filtered_songs[feature] >= threshold]
    
    if filtered_songs.empty:
        return None
    
    return filtered_songs[['track_name', 'artist', 'track_id']].sample(n=top_n)
