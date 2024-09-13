import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

       

def plot_scatter_plots(df, x_col, y_col, category_col):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], c=df[category_col], cmap='viridis', alpha=0.5)
    plt.title(f'Scatter plot of {y_col} vs {x_col} (colored by {category_col})')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.colorbar(label=category_col)
    plt.grid(True)
    plt.show()


def plot_correlation_matrix(df, numerical_columns):
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numerical_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Correlation Matrix')
    plt.show()
