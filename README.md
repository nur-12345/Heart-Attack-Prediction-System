# Heart-Attack-Prediction-System
A Machine Learning-based system that predicts the risk of heart attack using clinical parameters. 
Built with:

- 🧠 **XGBoost** Model (with feature engineering and hyperparameter tuning)
- ⚡ **FastAPI** for backend API serving
- 🌐 **Streamlit** for interactive frontend UI
- ✅ **Model Explainability Ready** (SHAP & ROC supported optionally)

---

## 📌 Features

- Risk classification: **Positive (High Risk)** or **Negative (Low Risk)**
- Feature-engineered inputs like:
  - Pulse pressure, BP ratio
  - Log-transformed vitals
  - Flags for high glucose, troponin, KCM
- Cross-platform: Test via **browser, Swagger UI, or `curl`**
- Easy to deploy, run locally, or host on cloud

---

## 📂 Project Structure

.
├── app.py # ML model training & saving (XGBoost)
├── main.py # FastAPI backend (API for prediction)
├── frontend.py # Streamlit UI for user input & results
├── test_inputs.py # Sample test requests (High & Low risk)
├── xgboost_heart_attack_model.pkl # Trained model
├── Heart Attack.csv # Dataset (if public)
├── requirements.txt # Dependencies
└── README.md # You're here!


---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/your-username/heart-disease-prediction-ml-app.git
cd heart-disease-prediction-ml-app
pip install -r requirements.txt
🔁 Train the Model (Optional)
If you want to retrain:

python app.py
🖥️ Run the App

▶️ Start FastAPI Backend
uvicorn main:app --reload
Access:

FastAPI API: http://127.0.0.1:8000
Swagger UI: http://127.0.0.1:8000/docs
🖼️ Start Streamlit Frontend
In a new terminal:

streamlit run frontend.py
App opens at: http://localhost:8501

🧪 Test the API (optional)

Use Swagger UI or run:

📦 Requirements

fastapi
uvicorn
streamlit
scikit-learn
xgboost
matplotlib
seaborn
joblib
numpy
pandas
requests

Made with ❤️ by Nupur Shivani

⚠️ Disclaimer

This project is for educational and demonstration purposes only. Not to be used as a medical tool without validation from certified medical professionals.


