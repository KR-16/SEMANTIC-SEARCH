"""
    Create an Streamlit app that does the following:

    - Reads an input from the user
    - Embeds the input
    - Search the vector DB for the entries closest to the user input
    - Outputs/displays the closest entries found
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from embed_and_store_data import embedding_generator
from clean_data import DataPreprocessor
import numpy as np
st.title("SEMANTIC SEARCH ON IMDB MOVIES DATASET TOP 1000 ONLY")

input = st.text_input("Enter the query to search")
k = st.number_input("Enter the top nearest values to be displayed", min_value=1, max_value=1000, step=1, format="%d")
model = SentenceTransformer("paraphrase-distilroberta-base-v1")

start = st.button("Start")

if start:
    if not input or not k:
        st.error("Enter the query and the k value")
    else:
        user_embedding = model.encode([input])[0]
        data, index = embedding_generator()
        D, I = index.search(np.array([user_embedding.astype("float32")]), k)
        # data = DataPreprocessor("imdb_top_1000.csv")
        st.write(f"Top {k} closest values from the search")
        for i in I[0]:
            st.write(data[["Series_Title", "Runtime", "Genre", "IMDB_Rating",	"Overview"]].iloc[i])
