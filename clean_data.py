# Which columns will you use?
# Clean your columns
# Concatenate the columns needed for your embedding
# Create new column with concatenated and clean text

import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, data):
        self.data = pd.read_csv(data)
    
    def display_data(self):
        return self.head()
    
    def display_shape_data(self):
        return self.data.shape
    
    def display_columns(self):
        return self.data.columns
    
    def missing_values(self):
        return self.data.isnull().sum()
    
    def preprocess_missing_values(self):
        self.data["Certificate"] = self.data["Certificate"].fillna(self.data["Certificate"].mode()[0])
        self.data["Meta_Score"] = self.data["Meta_Score"].fillna(self.data["Meta_Score"].mean())
        self.data["Gross"] = self.data["Gross"].str.replace(",", "")
        self.data["Gross"] = self.data["Gross"].astype("float32").replace(np.nan, 0).astype("int32")

    def data_correlation(self):
        numeric_columns = self.data.select_dtypes(["float32", "int32"])
        return numeric_columns
    
    def concatenate_text_columns(self):
        self.data['text'] = self.data[["Overview", "Director", "Star1", "Star2", "Star3", "Star4", "Series_Title"]].apply(lambda x: ' '.join(map(str, x)), axis=1)
        return self.data["text"]

    def drop_columns(self):
        self.data = self.data.drop(columns=["Poster_Link", "Series_Title", "Overview", "Director", "Star1", "Star2", "Star3", "Star4"], axis=1).head()

    def output(self):
        return self.data[["Series_Title", "Overview", "IMDB_Rating"]]
