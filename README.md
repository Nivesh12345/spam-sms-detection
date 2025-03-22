# Spam SMS Detection

Welcome to my Spam SMS Detection project! In this project, I build a machine learning model to classify SMS messages as either **spam** or **ham** (not spam). This project demonstrates my ability to work through an end-to-end machine learning pipeline, including data cleaning, exploratory analysis, feature extraction, modeling, hyperparameter tuning, and interpretation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

## Project Overview

In this project, I work with the SMS Spam Collection dataset to create a robust spam detection system. The model I developed uses TF-IDF for feature extraction and Multinomial Naïve Bayes for classification. Additionally, I perform hyperparameter tuning using GridSearchCV and conduct an error analysis to understand misclassifications. I also provide insights into the model's decision-making by extracting the top influential features for both spam and ham messages.

## Dataset

The dataset used in this project is the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) available on Kaggle. It consists of over 5,500 SMS messages labeled as either "ham" or "spam". Some key points about the dataset:
- **Total Messages:** 5,572
- **Labels:** "ham" (non-spam) and "spam" (spam)
- **Format:** CSV file with columns `v1` (label) and `v2` (message)

## Methodology

1. **Data Loading & Cleaning:**  
   - I loaded the dataset using Pandas and removed unnecessary columns.
   - Performed basic text cleaning (lowercasing, punctuation removal, digit removal, and whitespace stripping).

2. **Exploratory Data Analysis (EDA):**  
   - Visualized the distribution of spam vs. ham messages.
   - Gained insights into the class imbalance inherent in the data.

3. **Text Preprocessing:**  
   - Applied TF-IDF vectorization to convert text into numerical features.
   - Cleaned text further to improve model performance.

4. **Modeling:**  
   - Built a pipeline that combines TF-IDF vectorizer with a Multinomial Naïve Bayes classifier.
   - Used GridSearchCV for hyperparameter tuning (adjusting `max_df`, `ngram_range`, and `alpha`).

5. **Evaluation & Error Analysis:**  
   - Evaluated the model using accuracy, a detailed classification report, and a confusion matrix.
   - Performed error analysis by reviewing misclassified examples.

6. **Model Interpretation:**  
   - Manually extracted and displayed the top features (words) that drive the model's decisions for both spam and ham messages.

## Results

- **Accuracy:** Approximately 97% on the test set.
- **Error Analysis:** Detailed review of misclassified messages was conducted to gain insights for further improvements.
- **Interpretability:** Top features for spam and ham were extracted using the model’s log probabilities to understand which words are most influential.

## Installation & Requirements

To run this project locally or on a platform like Kaggle, you'll need the following packages:

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- eli5

You can install all required packages using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib eli5
