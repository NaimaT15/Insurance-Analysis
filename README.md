## Welcome to AlphaCare Insurance Solutions (ACIS) Analysis 👋

## Description

This repository, titled "AlphaCare Insurance Solutions (ACIS) Analysis," provides a comprehensive toolkit for analyzing historical car insurance claim data and implementing Data Version Control (DVC). The project aims to identify low-risk clients for potential premium reductions and optimize marketing strategies. Additionally, DVC is used to manage, track, and version datasets for collaboration and reproducibility. Key features include:

- **Premium and Claims Analysis:** Uncovers patterns in premium and claim data to highlight high-risk and low-risk clients.

- **Geographic Insights:** Analyzes trends in premiums and claims across different regions, providing marketing insights for targeted campaigns.

- **Feature Importance Analysis:** Identifies which client and vehicle attributes have the greatest impact on insurance premiums and claims using regression models.

- **Data Version Control (DVC):** Implements DVC for efficient dataset versioning and management, ensuring smooth collaboration across different stages of analysis.
  

This project is modular, with separate directories for scripts, data, and notebooks, making it highly extendable for insurance analysts and data scientists.

## Getting Started
Follow the instructions below to set up and get started with the project.

## Prerequisites
Before you begin, ensure you have the following installed:

- Python
- VS Code or any other IDE
- Jupyter Notebook Extension on VS Code
- DVC (Data Version Control)
## Installation
Clone the repository:

``` bash
git clone https://github.com/NaimaT15/Insurance-Analysis.git

```

Navigate to the project root:

``` bash
cd Insurance-Analysis
```
Set up a Virtual Environment (Optional but recommended):

``` bash

python -m venv venv

```
Activate the virtual environment:
``` bash

source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the project dependencies:
``` bash

pip install -r requirements.txt
```
Install DVC (if not already installed):

``` bash

pip install dvc
```
## Usage
**Data Analysis Notebooks:** Open the analysis notebook to explore and analyze the insurance dataset:
``` bash
jupyter notebook notebooks/insurance_analysis.ipynb
```
**Run Python Scripts:** Execute the analysis scripts from the command line for data cleaning, feature engineering, and modeling:
``` bash
python scripts/insurance_analysis.py
```
**DVC Usage:** Track changes to datasets using DVC to ensure version control:
``` bash

dvc add data/MachineLearningRating_v3.csv
dvc push
```
## Features
- **Premium and Claims Data Analysis:** Investigates premium distributions, outliers, and risk segmentation among clients.

- **Geographic Trends:** Analyzes claims and premiums by regions to guide marketing strategies.

- **Regression Modeling:** Uses linear regression, decision trees, random forests, and XGBoost to predict premiums and claims.

- **DVC for Data Version Control:** Ensures efficient version tracking and collaboration using DVC for data management.

## Contribution
Contributions are welcome! Please create a pull request or issue to suggest improvements or add new features.

## Author
👤 **Naima Tilahun**

* Github: [@NaimaT15](https://github.com/NaimaT15)
## Show your support
Give a ⭐️ if this project helped you!
