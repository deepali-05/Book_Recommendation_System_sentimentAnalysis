#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
from transformers import pipeline




# In[13]:


books_df = pd.read_csv('Documents/Project 7/Data/Book.csv')


# In[15]:


reviews_df = pd.read_csv('Documents/Project 7/Data/Customers_reviews.csv')


# In[17]:


print(books_df.head())
print(reviews_df.head())


# In[19]:


books_df.rename(columns={'book title': 'book name'}, inplace=True)

# Merge the datasets on 'book name'
merged_df = pd.merge(books_df, reviews_df, on='book name', how='left')

# Display the merged dataset
print(merged_df.head())


# In[21]:


merged_df['rating'] = merged_df['rating'].astype(float)
merged_df['book price'] = merged_df['book price'].astype(float)
merged_df['reviewer rating'] = merged_df['reviewer rating'].astype(float)


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[25]:


books_df['combined_features'] = books_df.apply(lambda x: f"{x['book name']} {x['author']} {x['genre']}", axis=1)


# In[27]:


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['combined_features'])


# In[29]:


# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[31]:


# Function to get book recommendations based on the cosine similarity score
def get_content_based_recommendations(title, cosine_sim=cosine_sim):
    idx = books_df.index[books_df['book name'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    return books_df.iloc[book_indices]


# In[33]:


print(get_content_based_recommendations('Iron Flame (The Empyrean, 2)'))


# In[35]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


# In[37]:


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(reviews_df[['reviewer', 'book name', 'reviewer rating']], reader)


# In[39]:


# Use Singular Value Decomposition (SVD) for collaborative filtering
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[41]:


# Train the algorithm on the entire dataset
trainset = data.build_full_trainset()
svd.fit(trainset)


# In[43]:


def get_collaborative_recommendations(user_id, num_recommendations=10):
    unique_books = reviews_df['book name'].unique()
    user_ratings = reviews_df[reviews_df['reviewer'] == user_id]
    user_unrated_books = set(unique_books) - set(user_ratings['book name'])
    
    recommendations = []
    for book in user_unrated_books:
        pred = svd.predict(user_id, book)
        recommendations.append((book, pred.est))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    recommended_books = [book for book, _ in recommendations[:num_recommendations]]
    return books_df[books_df['book name'].isin(recommended_books)]

    


# In[45]:


print(get_collaborative_recommendations('Murderess Marbie'))


# In[47]:


from transformers import pipeline


# In[49]:


sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')


# In[50]:


MAX_SEQ_LENGTH = 512


# In[53]:


def analyze_sentiment(text):
    if len(text) > MAX_SEQ_LENGTH:
        text = text[:MAX_SEQ_LENGTH]
    result = sentiment_pipeline(text)
    return result[0]['label'], result[0]['score']


# In[55]:


merged_df['review_title_sentiment'], merged_df['review_title_confidence'] = zip(*merged_df['review title'].apply(lambda x: analyze_sentiment(str(x))))
merged_df['reviewer_description_sentiment'], merged_df['reviewer_description_confidence'] = zip(*merged_df['review description'].apply(lambda x: analyze_sentiment(str(x))))


# In[56]:


print(merged_df.head())


# In[57]:


def get_hybrid_recommendations(user_id, book_title, num_recommendations=10):
    content_recs = get_content_based_recommendations(book_title)
    collab_recs = get_collaborative_recommendations(user_id, num_recommendations)
    hybrid_recs = pd.concat([content_recs, collab_recs]).drop_duplicates().head(num_recommendations)
    return hybrid_recs
    


# In[58]:


print(get_hybrid_recommendations('Murderess Marbie', 'Iron Flame (The Empyrean, 2)'))


# In[63]:


import streamlit as st


# In[65]:


st.title('Hybrid Book Recommendation System with Sentiment Analysis')


# In[67]:


st.sidebar.header('User Input')
user_id = st.sidebar.text_input('Enter User ID')
book_title = st.sidebar.text_input('Enter Book Title')


# In[69]:


if st.sidebar.button('Get Recommendations'):
    if user_id and book_title:
        st.subheader('Recommendations for User:')
        hybrid_recs = get_hybrid_recommendations(user_id, book_title)
        st.write(hybrid_recs)
    else:
        st.warning('Please enter both User ID and Book Title')


# In[79]:


st.subheader('Sentiment Analysis on Book Reviews')
st.write(merged_df[['book name', 'review title', 'review_title_sentiment', 'review_title_confidence',
                    'review description', ]])


# In[ ]:




