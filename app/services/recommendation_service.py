import pandas as pd
from models.clustering import scale_features, perform_clustering
from models.recommendation import content_based_recommendation, mood_based_recommendation
from models.sentiment import analyze_sentiment, map_sentiment_to_mood, extract_keywords

def hybrid_recommendation(df, user_id=None, track_id=None, mood=None, user_input=None, top_n=5):
    df_scaled = perform_clustering(scale_features(df))
    recommendations = pd.DataFrame()

    if user_input:
        sentiment, _ = analyze_sentiment(user_input)
        mood = map_sentiment_to_mood(sentiment) if mood is None else mood
        keywords = extract_keywords(user_input)
        print(f"Detected Mood: {mood}, Keywords: {keywords}")

    if track_id:
        cb_recommendations = content_based_recommendation(df_scaled, track_id, top_n)
        if cb_recommendations is not None:
            recommendations = pd.concat([recommendations, cb_recommendations], ignore_index=True)

    if mood:
        mood_recommendations = mood_based_recommendation(df_scaled, mood, top_n)
        if mood_recommendations is not None:
            recommendations = pd.concat([recommendations, mood_recommendations], ignore_index=True)

    recommendations = recommendations.drop_duplicates()

    if recommendations.empty:
        return None

    return recommendations
