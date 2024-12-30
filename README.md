# 📰 Fake News Prediction   

A simple and efficient machine learning project for detecting fake news articles using Logistic Regression. 

## 🚀 Features  

- **Text Preprocessing**: Removes noise like stopwords, punctuation, and performs tokenization.  
- **TF-IDF Vectorization**: Converts textual data into meaningful numerical features.  
- **Logistic Regression**: A robust linear model for binary classification.  
- **Model Evaluation**: Provides accuracy, precision, recall, F1-score, and confusion matrix for performance analysis.  

## 📂 Project Structure  

```
Fake-News-Prediction/
├── data/
│   ├── train.csv             # Training dataset
│   ├── test.csv              # Test dataset
├── notebooks/
│   ├── fake_news_pipeline.ipynb  # End-to-end notebook with preprocessing, training, and evaluation
├── src/
│   ├── preprocessing.py      # Text preprocessing and cleaning functions
│   ├── logistic_model.py     # Logistic Regression model implementation
├── requirements.txt           # Python dependencies
├── README.md                  # Project README
└── app/
    ├── app.py                 # Streamlit app for live predictions
```  

## 📊 Dataset  

We used the [Fake News Dataset](https://www.kaggle.com/c/fake-news) from Kaggle, which contains labeled news articles with `title`, `text`, and `label` (`1` for fake news, `0` for real news). The dataset is split into training and test sets.  

## 🧠 How It Works  

1. **Text Preprocessing**:  
   - Text is cleaned to remove special characters, numbers, and stopwords.  
   - TF-IDF Vectorizer transforms text into numerical features.  

2. **Model Training**:  
   - Logistic Regression is trained on the TF-IDF features using the `train.csv` dataset.  
   - Hyperparameter tuning with GridSearchCV to optimize performance.  

3. **Model Evaluation**:  
   - The model is evaluated on the `test.csv` dataset.  
   - Performance metrics include accuracy, precision, recall, F1-score, and ROC-AUC.  

## 📈 Results  

| Metric               | Value   |  
|----------------------|---------|  
| Accuracy(Train data) | 98.63%  |  
| Accuracy(Test  data) | 97.90%  |

## 🔧 Future Improvements  


- Implement additional feature engineering techniques.  
- Deploy the app using Docker for scalability.  

## 🤝 Contributing  

Contributions are welcome! Please submit a pull request or create an issue for suggestions.  

## 🙌 Acknowledgments  

- [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news)  
- Libraries: Scikit-learn, Pandas, NumPy, Streamlit  


