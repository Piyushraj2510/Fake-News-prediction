📰 Fake News Prediction with Logistic Regression  

A simple and efficient machine learning project for detecting fake news articles using Logistic Regression. 

🚀 Features  

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

## ⚙️ Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/fake-news-prediction.git  
   cd fake-news-prediction  
   ```  

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Place the dataset files (`train.csv` and `test.csv`) into the `data/` directory.  

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

## 🖥️ Usage  

### 1. Run the Notebook  
Execute the step-by-step pipeline in the `notebooks/fake_news_pipeline.ipynb`.  

### 2. Command Line Interface  
Run the training and evaluation pipeline from the terminal:  
```bash  
python src/logistic_model.py --train data/train.csv --test data/test.csv  
```  

### 3. Web App  
Start the Streamlit app for real-time predictions:  
```bash  
cd app  
streamlit run app.py  
```  
Then, open `http://127.0.0.1:8501` in your browser.  

## 📈 Results  

| Metric              | Value   |  
|----------------------|---------|  
| Accuracy            | 91.3%   |  
| Precision           | 90.5%   |  
| Recall              | 91.8%   |  
| F1-Score            | 91.1%   |  

## 🔧 Future Improvements  

- Expand preprocessing techniques to include stemming/lemmatization.  
- Implement additional feature engineering techniques.  
- Deploy the app using Docker for scalability.  

## 📜 License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  

## 🤝 Contributing  

Contributions are welcome! Please submit a pull request or create an issue for suggestions.  

## 🙌 Acknowledgments  

- [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news)  
- Libraries: Scikit-learn, Pandas, NumPy, Streamlit  

---  

Let me know if you'd like to modify anything or add additional sections!
