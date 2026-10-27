# NetGuard ML 🛡️
### Machine Learning-Based Network Intrusion Detection System

NetGuard ML is a machine-learning-based network intrusion detection prototype developed to analyze network traffic and identify potentially suspicious activity using the CIC-IDS2017 dataset.

## 🚀 Features

- **Multiclass Intrusion Detection:** Classifies network traffic into supported categories, including benign traffic and several attack types.
- **Anomaly Detection:** Uses an experimental Isolation Forest model to identify traffic that differs from learned patterns.
- **Interactive Dashboard:** Built with Streamlit to make analysis easier.
- **CSV Upload:** Allows users to analyze network traffic datasets.
- **Results Summary:** Displays prediction counts and traffic classifications.
- **Downloadable Results:** Allows users to export analysis results.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib
- JSON
- Jupyter Notebook

## 📂 Project Structure

```text
NetGuard-ML/
├── app.py
├── NetGuard_ML.ipynb
├── models/
├── data/
├── outputs/
├── requirements.txt
└── README.md
```

*The actual folder contents may differ depending on which files are included in the repository.*

## 📊 Dataset

This project uses the CIC-IDS2017 network intrusion detection dataset.

The dataset contains benign network traffic and multiple attack categories. It is used to train and evaluate machine learning models for network traffic classification.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/NetGuard-ML.git
cd NetGuard-ML
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## ⚠️ Current Limitations

- The application analyzes uploaded CSV files rather than capturing live network traffic.
- An anomaly prediction is not proof of a malicious attack.
- Model performance may vary across datasets and traffic patterns.
- The project is a research and educational prototype, not a production-grade security system.

## 🔮 Future Improvements

- Real-time network traffic monitoring.
- Evaluation on independent network captures.
- Improved detection of previously unseen attacks.
- Enhanced dashboards and alerting.
- Integration with network monitoring tools.

## 👨‍💻 Author

**Aditya Singh**

B.Tech — Computer Science and Engineering (AI/ML)

## 📄 Disclaimer

This project is intended for educational and research purposes. Predictions should be validated before being used for security decisions.
