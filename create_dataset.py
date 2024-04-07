import kaggle
import pandas as pd
import zipfile

kaggle.api.authenticate()
dataset = "harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows"
kaggle.api.dataset_download_files(dataset)

with zipfile.ZipFile(
    "imdb-dataset-of-top-1000-movies-and-tv-shows.zip", "r"
) as zip_ref:
    zip_ref.extractall(".")

movies = pd.read_csv("imdb_top_1000.csv")
print(movies.columns)
print(movies[["Series_Title", "Overview"]].head(10))
