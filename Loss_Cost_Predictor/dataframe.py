import os
import kagglehub
import pandas as pd
import numpy as np

class DataFrame:

    def __init__(self, df):
        self.df = df

    def get_headers(self):
        return self.df.columns.tolist()

    def is_categorical(self, column_name):
        try:
            self.df[column_name]/1
            return False
        except TypeError:
            try:
                self.df[column_name].astype(int)
                return False
            except (ValueError, TypeError):
                return True

    def get_2D_array(self, extraneous_columns:list):
        self.df.drop(columns=extraneous_columns)
        return self.df

    def get_observed_claim_count(self, column):
        return self.df[column].to_numpy()

    def get_observed_claim_amount(self, column):
        return self.df[column].to_numpy()

    def claim_amount_by_ID(self, ID_column, severity):
        return self.df.groupby(ID_column)[severity].sum().to_numpy()

    def exposure_by_ID(self, other):
        if isinstance(other, DataFrame):
            merged_df = self.df.merge(
                other.df[['IDpol', 'Exposure']],
                on='IDpol',
                how='left'
            )

        return merged_df.groupby('IDpol')['Exposure'].first().to_numpy()