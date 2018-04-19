"""

Author : Badr YOUBI IDRISSI

This file will contain the Database object which loads the different dataframes needed and prepares them

Different functionalities include : 
    - load from csv
    - sliceToChunks(column) : slices the dataframe to chunks with the same value for "column"
"""

import pandas as pd


class Database(pd.DataFrame):
    def __init__(self, filepath):
        super().__init__(pd.read_csv(filepath))

    def sliceToChunks(self, column, dropCol = True):
        chunks = {}
        values = self[column].unique()

        for v in values:
            if dropCol:
                chunk = self[self[column] == v].drop(column, axis=1)
            else:
                chunk = self[self[column] == v]
            chunks[v] = chunk
        return chunks
