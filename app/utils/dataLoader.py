import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from dotenv import load_dotenv  

load_dotenv()

def load_dataset():
    dataset_path = os.getenv('DATA_PATH')
    if not dataset_path:
        raise ValueError("DATA_PATH environment variable not set.")
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at the specified path: {dataset_path}")
    
    audio_features = ['danceability', 'energy', 'valence', 'tempo', 'liveness', 'loudness', 'speechiness', 'acousticness', 'instrumentalness']
    scaler = MinMaxScaler()
    df[audio_features] = scaler.fit_transform(df[audio_features])
    
    return df
