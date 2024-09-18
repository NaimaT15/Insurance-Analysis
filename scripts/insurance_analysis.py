import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import scipy.stats as stats


def handle_missing_values(df):
    # Fill missing values for categorical columns with 'Unknown' or mode
    categorical_columns = ['Bank', 'AccountType', 'MaritalStatus', 'Gender', 'VehicleType', 'make', 'Model',
                           'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'AlarmImmobiliser',
                           'TrackingDevice'] 
    for col in categorical_columns:
        df[col] = df[col].fillna('Unknown')

    # Fill missing values for columns where the mode is appropriate
    mode_columns = ['mmcode', 'bodytype', 'CapitalOutstanding']
    for col in mode_columns:
        mode_value = df[col].mode()[0]  # Get the most frequent value
        df[col] = df[col].fillna(mode_value)

    # Handle date columns separately - Convert to datetime format first
    if 'VehicleIntroDate' in df.columns:
        df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')  # Convert to datetime
        df['VehicleIntroDate'] = df['VehicleIntroDate'].fillna(df['VehicleIntroDate'].median())  # Fill missing dates

    # Fill missing values for numerical columns with the median
    numerical_columns = ['Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors', 'CustomValueEstimate']
    
    for col in numerical_columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    
    # Special handling for 'NumberOfVehiclesInFleet'
    if 'NumberOfVehiclesInFleet' in df.columns:
        # Fill with a default value, e.g., '1' or '0' if missing
        df['NumberOfVehiclesInFleet'] = df['NumberOfVehiclesInFleet'].fillna(1)  # Assuming most are individual vehicles
    
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

def plot_cover_type_trends(df, geo_col, cover_col):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=geo_col, hue=cover_col, data=df, palette='Set2')
    plt.title(f'Comparison of {cover_col} across {geo_col}')
    plt.xticks(rotation=45)
    plt.xlabel(geo_col)
    plt.ylabel('Count')
    plt.legend(title=cover_col)
    plt.grid(True)
    plt.show()

def plot_premium_trends(df, geo_col, premium_col):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=geo_col, y=premium_col, data=df, palette='Set1')
    plt.title(f'Distribution of {premium_col} across {geo_col}')
    plt.xticks(rotation=45)
    plt.xlabel(geo_col)
    plt.ylabel(premium_col)
    plt.grid(True)
    plt.show()

def plot_auto_make_trends(df, geo_col, make_col):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=geo_col, hue=make_col, data=df, palette='Set3')
    plt.title(f'Comparison of {make_col} across {geo_col}')
    plt.xticks(rotation=45)
    plt.xlabel(geo_col)
    plt.ylabel('Count')
    plt.legend(title=make_col)
    plt.grid(True)
    plt.show()


def plot_outliers_boxplot(df, numerical_columns):
    for col in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col], palette="Set3")
        plt.title(f'Box Plot for Outlier Detection in {col}')
        plt.xlabel(col)
        plt.grid(True)
        plt.show()

def plot_correlation_heatmap(df):
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['number'])
    
    # Calculate the correlation matrix for numerical columns only
    plt.figure(figsize=(10, 8))
    corr_matrix = numerical_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()

def plot_stacked_bar(df, x_col, hue_col):
    plt.figure(figsize=(12, 6))
    df_grouped = df.groupby([x_col, hue_col]).size().unstack().fillna(0)
    df_grouped.plot(kind='bar', stacked=True, colormap='Set3', figsize=(12, 8))
    plt.title(f'Distribution of {hue_col} Across {x_col}')
    plt.xlabel(x_col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title=hue_col)
    plt.grid(True)
    plt.show()

def plot_box_premium_by_vehicle(df, vehicle_col, premium_col):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=vehicle_col, y=premium_col, data=df, palette='Set2')
    plt.title(f'Box Plot of {premium_col} by {vehicle_col}')
    plt.xticks(rotation=45)
    plt.xlabel(vehicle_col)
    plt.ylabel(premium_col)
    plt.grid(True)
    plt.show()



def select_kpi(df, kpi_column):
    """
    Select the KPI (key performance indicator) column for analysis.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the insurance data.
    kpi_column (str): The name of the column representing the KPI.
    
    Returns:
    pd.Series: The KPI values.
    """
    if kpi_column in df.columns:
        return df[kpi_column]
    else:
        raise ValueError(f"KPI column {kpi_column} not found in the dataset.")
def create_ab_groups(df, feature_column, value_a, value_b):
    """
    Split the dataset into two groups (control and test) based on the feature.
    
    """
    group_a = df[df[feature_column] == value_a]
    group_b = df[df[feature_column] == value_b]
    
    return group_a, group_b
def chi_squared_test(group_a, group_b, feature_column):
    """
    Perform a chi-squared test to check if there is a significant difference
    between two groups for a categorical feature.
    """
    combined = pd.concat([group_a, group_b])

    # Create the contingency table
    contingency_table = pd.crosstab(combined[feature_column], combined.index)

    # Perform the chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return p

def t_test(group_a, group_b, kpi_column):
    """
    Perform a t-test to check if there is a significant difference between two groups for a numerical KPI.
    """
    stat, p = ttest_ind(group_a[kpi_column], group_b[kpi_column], equal_var=False)
    return p

def analyze_p_value(p_value, alpha=0.05):
    """
    Analyze the p-value and determine whether to reject the null hypothesis.
    """
    if p_value < alpha:
        return "Reject the null hypothesis. There is a statistically significant difference."
    else:
        return "Fail to reject the null hypothesis. There is no statistically significant difference."

# Step 2: Group the data by PostalCode and check for differences in 'TotalClaims'
def anova_test_for_postalcode_risk(df, kpi_column, group_column):
    # Get unique postal codes
    postal_codes = df[group_column].unique()

    # Prepare data for ANOVA: collect 'TotalClaims' for each postal code
    postal_groups = [df[df[group_column] == postal_code][kpi_column].dropna() for postal_code in postal_codes]

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*postal_groups)

    return f_stat, p_value
def anova_test_for_postalcode_margin(df, kpi_column, group_column):
    # Get unique postal codes
    postal_codes = df[group_column].unique()
    print(postal_codes)
    # Prepare data for ANOVA: collect 'Margin' for each postal code
    postal_groups = [df[df[group_column] == postal_code][kpi_column].dropna() for postal_code in postal_codes]

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*postal_groups)

    return f_stat, p_value
def t_test_men_women(df_male, df_female, kpi_column):
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(df_male[kpi_column].dropna(), df_female[kpi_column].dropna(), equal_var=False)
    return t_stat, p_value