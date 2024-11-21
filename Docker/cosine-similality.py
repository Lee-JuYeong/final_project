from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(data, text_column):
    count_vectorizer = CountVectorizer()
    vectors = count_vectorizer.fit_transform(data[text_column])
    return cosine_similarity(vectors, vectors)
