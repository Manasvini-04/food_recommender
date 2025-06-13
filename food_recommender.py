# AI-Based Food Recommendation System (Keyword-Based Search)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("DATASETFR.csv")

# Fill missing values if any
df['Describe'] = df['Describe'].fillna('')

# Combine category and description for richer context
df['Features'] = df['C_Type'] + ' ' + df['Veg_Non'] + ' ' + df['Describe']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Features'])

# Helper: Recommend based on user keywords
def recommend_by_keywords(keywords, top_n=5):
    # Transform user input using the same vectorizer
    query_vec = tfidf.transform([keywords])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top-N indices
    top_indices = sim_scores.argsort()[-top_n:][::-1]

    recommendations = []
    for i in top_indices:
        recommendations.append({
            'Name': df.iloc[i]['Name'],
            'Type': df.iloc[i]['C_Type'],
            'Veg/Non-Veg': df.iloc[i]['Veg_Non'],
            'Description': df.iloc[i]['Describe']
        })
    return recommendations

# Example usage
if __name__ == "__main__":
    print("Enter keywords related to ingredients, type, or veg/non-veg (e.g., 'veg spicy almonds', 'non-veg salad', 'dessert chocolate'):")
    user_input = input("Your keywords: ")
    results = recommend_by_keywords(user_input)
    print(f"\nTop recommended dishes for: '{user_input}'\n")
    for i, rec in enumerate(results, 1):
        print(f"#{i}: {rec['Name']} | {rec['Type']} | {rec['Veg/Non-Veg']}\nDescription: {rec['Description']}\n")
