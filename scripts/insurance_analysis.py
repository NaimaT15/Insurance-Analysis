import pandas as pd

def handle_missing_values(df):
    # Fill missing values for categorical columns with 'Unknown' or mode
    df['Bank'] = df['Bank'].fillna('Unknown')
    df['AccountType'] = df['AccountType'].fillna('Unknown')
    df['MaritalStatus'] = df['MaritalStatus'].fillna('Unknown')
    df['Gender'] = df['Gender'].fillna('Unknown')
    
    # For categorical columns like 'VehicleType', 'make', 'Model', fill missing with the mode
    categorical_columns = ['VehicleType', 'make', 'Model']
    for col in categorical_columns:
        mode_value = df[col].mode()[0]  # Get the most frequent value
        df[col] = df[col].fillna(mode_value)

    # Fill missing values for numerical columns with the median
    numerical_columns = ['Cylinders', 'cubiccapacity', 'mmcode']
    for col in numerical_columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    return df
import matplotlib.pyplot as plt

def plot_numerical_histograms(df, numerical_columns):
    for col in numerical_columns:
        plt.figure(figsize=(8, 6))
        plt.hist(df[col].dropna(), bins=20, color='blue', alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
def plot_categorical_bars(df, categorical_columns):
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar', color='green', alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()
