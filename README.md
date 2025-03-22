# Spam SMS Detection

Welcome to my **Spam SMS Detection** project! In this project, I build a machine learning model to classify SMS messages as either **spam** or **ham** (not spam). This project includes a fully functional **web app** built with Streamlit, allowing users to enter a message and get instant predictions.

🔗 **Live Web App:** [Spam SMS Detection Web App](https://spam-sms-detection-jlunztznqpypc4jb4gbnar.streamlit.app/)  
📘 **Kaggle Notebook:** [Spam SMS Detection on Kaggle](https://www.kaggle.com/code/bustergone/spam-sms-detection)  

---

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation & Running Locally](#installation--running-locally)
- [🌐 Streamlit Web App](#streamlit-web-app)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

---

## 🔍 Project Overview

This project uses the **SMS Spam Collection dataset** to develop a spam detection system. The **Multinomial Naïve Bayes** model, combined with **TF-IDF vectorization**, provides a strong baseline for classifying spam messages. 

The final model is deployed via **Streamlit**, allowing users to interactively test messages and see predictions in real-time.

🔗 **Try it here:** [Spam SMS Detection Web App](https://spam-sms-detection-jlunztznqpypc4jb4gbnar.streamlit.app/)
📘 **Kaggle Notebook:** [Spam SMS Detection on Kaggle](https://www.kaggle.com/code/bustergone/spam-sms-detection)

---

## 📊 Dataset

The dataset used in this project is the **[SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)** available on Kaggle. It consists of:

- **Total Messages:** 5,572 SMS messages  
- **Labels:** "ham" (not spam) and "spam" (spam)  

---

## ⚙️ Methodology

1. **Data Loading & Cleaning:**  
   - Removed unnecessary columns, renamed headers, cleaned text.

2. **Feature Extraction:**  
   - Used **TF-IDF (Term Frequency-Inverse Document Frequency)**.

3. **Modeling & Hyperparameter Tuning:**  
   - Trained a **Multinomial Naïve Bayes** classifier.
   - Used **GridSearchCV** to optimize model performance.

4. **Evaluation & Deployment:**  
   - Evaluated model with accuracy, precision, recall, and confusion matrix.
   - Deployed a **Streamlit web app** for live testing.

---

## ✅ Results

- **Accuracy:** ~97% on the test set.
- **Interpretability:** Extracted **top features** (words) for spam and ham.
- **Error Analysis:** Misclassified messages were reviewed for insights.

---

## 🚀 Installation & Running Locally

To run this project on your local machine, follow these steps:

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your_username/spam-sms-detection.git
cd spam-sms-detection
pip install -r requirements.txt
streamlit run app.py
