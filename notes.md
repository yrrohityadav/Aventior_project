# Data exploration notes
# EDA complete - target is imbalanced (4.8% fraud)
# Added text length and categorical feature analysis
# Binary features (logo, questions) show correlation with fraud
# Implemented text preprocessing pipeline
# Steps: lowercase, HTML removal, URL removal, punctuation removal
# Added lemmatization using WordNetLemmatizer
# Combined 5 text columns into single feature for better classification
# TF-IDF vectorization with 10000 features and bigrams
# Implemented train-test split with stratification
# Trained Logistic Regression with balanced class weights
# Trained Random Forest with 200 estimators
