import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

audio_features = ['danceability', 'energy', 'valence', 'tempo', 'liveness', 'loudness', 'speechiness', 'acousticness', 'instrumentalness']

def scale_features(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[audio_features]), columns=audio_features)
    df_scaled['track_id'] = df['track_id']
    df_scaled['track_name'] = df['track_name']
    df_scaled['artist'] = df['artist']
    return df_scaled

def perform_clustering(df_scaled, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_scaled['cluster'] = kmeans.fit_predict(df_scaled[audio_features])
    return df_scaled
