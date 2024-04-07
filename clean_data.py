# Which columns will you use?
# Clean your columns
# Concatenate the columns needed for your embedding
# Create new column with concatenated and clean text

import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.preprocessedData()
    
    def display_data(self):
        return self.head()
    
    def display_shape_data(self):
        return self.data.shape
    
    def display_columns(self):
        return self.data.columns
    
    def missing_values(self):
        return self.data.isnull().sum()
    
    def preprocessedData(self):
        self.data["Certificate"] = self.data["Certificate"].fillna(self.data["Certificate"].mode()[0])
        self.data["Meta_score"] = self.data["Meta_score"].fillna(self.data["Meta_score"].mean())
        self.data["Gross"] = self.data["Gross"].astype(str).str.replace(",", "").astype("float32").replace(np.nan, 0).astype("int32")
        self.data['text'] = self.data[["Overview", "Director", "Star1", "Star2", "Star3", "Star4", "Series_Title"]].apply(lambda x: ' '.join(map(str, x)), axis=1)
        self.data = self.data.drop(columns=["Poster_Link", "Released_Year", "Certificate", "Director", "Star1", "Star2", "Star3", "Star4", "Meta_score","No_of_Votes","Gross"], axis=1)

    def text_column(self):
        return self.data['text']

    def data_correlation(self):
        numeric_columns = self.data.select_dtypes(["float32", "int32"])
        return numeric_columns

    def output(self):
        return self.data
