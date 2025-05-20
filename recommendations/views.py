from django.shortcuts import render
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.sentiment import SentimentIntensityAnalyzer

movies = pd.read_csv('data/movies.csv')
critics = pd.read_csv('data/critic_reviews.csv')

sia = SentimentIntensityAnalyzer()

def get_sentiment_score(review):
    score = sia.polarity_scores(review)
    return score['compound']  

critics['sentiment_score'] = critics['review_content'].apply(get_sentiment_score)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_movies(title, cosine_sim=cosine_sim):
    idx = movies.index[movies['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  
    movie_indices = [i[0] for i in sim_scores]
    
    recommended_movies = movies['title'].iloc[movie_indices]
    recommended_sentiment_scores = []
    for movie in recommended_movies:
        avg_sentiment = critics[critics['movie_id'] == movies[movies['title'] == movie].iloc[0]['movie_id']]['sentiment_score'].mean()
        recommended_sentiment_scores.append(avg_sentiment)
    
    return list(zip(recommended_movies, recommended_sentiment_scores))

def index(request):
    return render(request, 'index.html')

def recommend(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        recommendations = recommend_movies(title)
        return render(request, 'index.html', {'title': title, 'recommendations': recommendations})
    return render(request, 'index.html')
