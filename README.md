# Semantic Search Using Streamlit

This application was designed to perform semantic search on an IMDB dataset. 
Streamlit is used to represent user inputs and outputs.The Sentence Transformers library embeds text data. 
The search functionality allows users to enter a query, which is then embedded and applied to search for entries in the dataset that are very similar to the query.

- Reads an input from the user.
- Embeds the input using a pre-trained transformer model - Sentence Transformer.
- Searches the vector database for entries closest to the user input.
- Displays the closest entries found.

## Prerequisites

Before running the application, ensure that you have the following dependencies installed:

- Python 3.x
- Streamlit
- Sentence Transformers
- Pandas
- Numpy
- faiss
- Kaggle API (optional, only required if you want to use the provided dataset)

## Installation

You can install the required Python packages using pip:

pip install -r requirements.txt

## Usage

Run the Streamlit application:

streamlit run app.py

## Additional Scripts

- **clean_data.py**: Contains a class for data preprocessing tasks such as cleaning columns, handling missing values, and creating text columns for embedding.
- **create_dataset.py**: Downloads and preprocesses the IMDb dataset of top 1000 movies and TV shows.
- **embed_and_store.py**: Prepares text data for embedding, chooses a sentence embedding model, embeds the text data, and stores the embeddings in a vector database.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Kaggle API](https://github.com/Kaggle/kaggle-api) (for dataset access)
- [Semantic Search](https://medium.com/@pankaj_pandey/exploring-semantic-search-using-embeddings-and-vector-databases-with-some-popular-use-cases-2543a79d3ba6)
- [FAISS Database](https://iamajithkumar.medium.com/working-with-faiss-for-similarity-search-59b197690f6c)
