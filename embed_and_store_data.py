"""
- Prepare the text to embed for each reccord of your dataset.
    - Create the reccord.
        - Clean the text.
        - Concatenate fields.
- Choose a Sentence Embedding Model.
- Embed the text generated in the previous step for each reccord.
- Store the embeddings in a vector database (i.e. elasticsearch).
"""
from sentence_transformers import SentenceTransformer
from clean_data import DataPreprocessor
import faiss

def embedding_generator():
    model = SentenceTransformer("paraphrase-distilroberta-base-v1")
    processing = DataPreprocessor("imdb_top_1000.csv")
    text = processing.concatenate_text_columns()
    embeddings = model.encode(text.tolist())
    # print(embeddings.shape) - 1000, 768 (1000 rows and 768 dimensions)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index