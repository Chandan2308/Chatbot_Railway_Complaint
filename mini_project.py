# -*- coding: utf-8 -*-
"""Mini Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1T0x7MtO7th3Jvd_aSyAwjyTqTDvoR0co

Libraries Installation
"""

# !pip install pandas scikit-learn sqlite3

"""Import Code"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

"""Read The CSV file"""

file_path = "/model/train.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

"""converting the complaints from the csv file to a python list for better results"""

complaints = df["SentimentText"].astype(str).tolist()

"""converting the complaints to tf-idf vectors"""

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(complaints)

"""SQL Commands for creating database file and a table inside it"""

conn = sqlite3.connect("complaints.db")
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_complaint TEXT,
        category TEXT
    )
''')
conn.commit()

"""Complaint Managing Chatbot"""

def categorize_complaint(user_query):
    query_vector = vectorizer.transform([user_query])  # Convert query to vector
    similarities = cosine_similarity(query_vector, X)  # Compute similarity
    best_match_index = similarities.argmax()  # Find the best match
    similarity_score = similarities[0, best_match_index]  # Get similarity score

    # Categorizes the complaints according to the similarity score
    if similarity_score > 0.5:
        category = "Priority Complaint"
    else:
        category = "Normal Complaint"

    # Store in database
    cursor.execute("INSERT INTO complaints (user_complaint, category) VALUES (?, ?)",
                   (user_query, category))
    conn.commit()

    return category

"""Executing The Chatbot"""

print("🚆 Railway Complaint Chatbot (Type 'exit' to stop) 🚆")

while True:
    user_query = input("\nEnter your railway complaint: ")

    if user_query.lower() == "exit":
        print("🔴 Thank you for your patience! All complaints are stored in the database.")
        break

    category = categorize_complaint(user_query)
    print(f"✅ Your complaint is categorized as: {category}")

"""Display Complaints according to the category"""

cursor.execute("SELECT user_complaint, category FROM complaints")
records = cursor.fetchall()

priority_complaints = [row[0] for row in records if row[1] == "Priority Complaint"]
normal_complaints = [row[0] for row in records if row[1] == "Normal Complaint"]

print("\n🔴 Priority Complaints")
print("-" * 30)
for complaint in priority_complaints:
    print(complaint)

print("\n🟢 Normal Complaints")
print("-" * 30)
for complaint in normal_complaints:
    print(complaint)

"""Delete all the complaints"""

conn = sqlite3.connect("complaints.db")
cursor = conn.cursor()


cursor.execute("DELETE FROM complaints")
conn.commit()

print("✅ All complaints have been deleted from the database.")