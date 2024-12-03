# 🔍 Fake Job Detection using NLP

A machine learning project to detect fraudulent job postings using Natural Language Processing (NLP) techniques.

## 📋 Project Overview

This project builds a text classification model to identify fake/fraudulent job postings from the **EMSCAD (Employment Scam Aegean Dataset)**. The pipeline includes text preprocessing, TF-IDF vectorization, and classification using Logistic Regression and Random Forest models.

## 📊 Dataset

- **Source**: [Kaggle - Real or Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Records**: 17,880 job postings
- **Features**: 18 columns (title, location, description, requirements, etc.)
- **Target**: `fraudulent` (0 = Legitimate, 1 = Fraudulent)
- **Class Distribution**: ~95.2% Legitimate, ~4.8% Fraudulent

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.x | Programming Language |
| Pandas | Data Manipulation |
| NLTK | NLP Preprocessing |
| Scikit-learn | Machine Learning |
| Matplotlib & Seaborn | Data Visualization |
| WordCloud | Text Visualization |

## 🔧 NLP Pipeline

```
Raw Text → Lowercase → Remove HTML/URLs → Remove Punctuation
→ Tokenization → Stop-word Removal → Lemmatization → TF-IDF Vectorization
```

### Preprocessing Steps:
1. **Text Combination**: Merged 5 text columns (title, company_profile, description, requirements, benefits)
2. **Cleaning**: Removed HTML tags, URLs, emails, numbers, and special characters
3. **Tokenization**: Split text into individual words using NLTK
4. **Stop-word Removal**: Removed common English stop words
5. **Lemmatization**: Reduced words to their base/root form
6. **TF-IDF Vectorization**: Converted text to numerical features (10,000 features, unigrams + bigrams)

## 🤖 Models

### 1. Logistic Regression
- Solver: L-BFGS
- Class weight: Balanced (to handle imbalanced data)
- Regularization: C=1.0

### 2. Random Forest Classifier
- Number of trees: 200
- Class weight: Balanced
- Max depth: None (fully grown trees)

## 📈 Results

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | ~96% | ~97% |
| F1-Score | ~70%+ | ~75%+ |
| ROC-AUC | ~0.97 | ~0.98 |

> Both models achieve **92%+ accuracy** with ROC-AUC scores > 0.95

## 📊 Visualizations

The project generates 10 comprehensive visualizations:

1. **Target Distribution** - Bar chart + Pie chart of fraudulent vs legitimate
2. **Missing Values** - Analysis of null values across features
3. **Categorical Analysis** - Employment type & experience vs fraud
4. **Text Length Analysis** - Distribution of text lengths by class
5. **Binary Features** - Company logo & questions vs fraud correlation
6. **Word Clouds** - Most frequent words in legitimate vs fraudulent postings
7. **Confusion Matrices** - Model prediction accuracy breakdown
8. **ROC-AUC Curves** - Model comparison with AUC scores
9. **Model Comparison** - Side-by-side metric comparison
10. **Feature Importances** - Top predictive words for each model

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupytext
```

### Run as Jupyter Notebook
```bash
jupyter notebook Fake_Job_Detection_NLP.ipynb
```

### Run as Python Script
```bash
python fake_job_detection.py
```

> **Note**: The notebook auto-installs required packages on first run.

## 📁 Project Structure

```
fake_job/
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── fake_job_postings.csv            # Dataset (EMSCAD)
├── fake_job_detection.py            # Python source script
├── Fake_Job_Detection_NLP.ipynb     # Jupyter notebook (with outputs)
├── 01_target_distribution.png       # Visualization
├── 02_missing_values.png            # Visualization
├── 03_categorical_analysis.png      # Visualization
├── 04_text_length_analysis.png      # Visualization
├── 05_binary_features.png           # Visualization
├── 06_wordclouds.png                # Visualization
├── 07_confusion_matrices.png        # Visualization
├── 08_roc_auc_curves.png            # Visualization
├── 09_model_comparison.png          # Visualization
└── 10_feature_importances.png       # Visualization
```

## 🔮 Future Improvements

- Deep learning approaches (BERT, LSTM)
- Additional meta-feature engineering
- Ensemble methods (XGBoost, Stacking)
- Web application deployment with Flask/Streamlit
- Real-time prediction API

## 👤 Author

**Rohit Yadav**
- Project Duration: Nov 2024 – Jan 2025

## 📄 License

This project is for educational purposes.
