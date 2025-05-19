📊 AI-Powered Data Cleaning & Visualization Tool

An AI-enhanced Streamlit web app for:
- Data cleaning (missing value handling, column dropping)
- Data visualization (histograms, boxplots, heatmaps)
- AI sentiment analysis on text input
- PDF report generation
- Dataset upload history tracking

🚀 Live Demo
((https://ai-data-cleaning-app-jrw2m5uchjtudjb9pzi2kb.streamlit.app/))

 📦 How to Run Locally

pip install -r requirements.txt
streamlit run app.py

📝 Features

- 📥 **Upload and clean CSV datasets**
- 🗑️ **Auto-suggest columns to drop** based on uniqueness or null percentage
- 🔍 **Handle missing values automatically**
  - Numeric columns filled with **median**
  - Categorical columns filled with **most frequent (mode)**
- 📊 **Visualize data with various charts**
  - Histograms
  - Boxplots
  - Correlation Heatmaps
- 🧠 **AI Sentiment Analysis** on user-entered text using Hugging Face models
- 📄 **Export cleaned data as CSV & PDF report**
- 📜 **Track upload history** of cleaned datasets and reports

---

🔗 Tech Stack

- 🐍 **Python**
- 🚀 **Streamlit**
- 📊 **Pandas**, **NumPy**, **Seaborn**, **Matplotlib**
- 🤗 **Transformers (Hugging Face)**
- 📄 **ReportLab**
