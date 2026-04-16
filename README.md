# spam-email-classifier-ml
A Machine Learning-based Email Spam Classification System that detects whether an email is spam or not using Natural Language Processing (TF-IDF) and Naive Bayes algorithm. The project includes a trained ML model and an interactive web application built with Streamlit for real-time predictions.
# Email Spam Classifier (Computer Science + Machine Learning)

## 🔥 Introduction
The current project is a Spam Email Classification system created using Machine Learning and Natural Language Processing methods. The system determines whether an email is spam or not using text processing algorithms and probability calculations.

---

## 👩‍💻 Goal
- To learn about text classification in computer science
- To implement probabilistic ML algorithms 
- To create an actual spam filter
- To deploy it as a web application

---

## 🤔 ML Algoirthms Implemented
- TF-IDF Vectorization (feature extraction method)
- Naive Bayes Classifier (main CS algorithm for text classification)

---

## ⚡ Pipeline
1. Get input from user in form of email text 
2. Convert input to numeric features 
3. Feed features into ML model
4. Get predicted value: spam or no spam
5. Show predicted output on UI

---

## 🛠️ Application Features
- Real-time spam prediction
- Simple UI
- Lightweight ML algorithm 
- Prediction speed

---

## 💻 Run Application
```bash
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py
