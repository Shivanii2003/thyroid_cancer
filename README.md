
# 🧠 Thyroid Cancer Recurrence Prediction

This project predicts the recurrence of thyroid cancer using machine learning techniques. It was developed during my internship with **Unified Mentor**, and is focused on building an accurate model using patient data for early risk assessment.

---

## 📌 Objective

To develop a predictive model that classifies whether a thyroid cancer patient is likely to experience a recurrence after treatment, helping in clinical decision-making.

---

## 🗃️ Dataset

The dataset includes anonymized medical records of thyroid cancer patients with features such as:

- Age
- Gender
- Tumor size
- Number of lymph nodes involved
- Surgery type
- Radiation therapy
- Histological subtype
- Cancer stage
- Time since initial treatment

The target variable is:
- `Recurred` – indicates if the patient experienced cancer recurrence.

---

## 🛠️ Technologies Used

- Python
- NumPy, Pandas – data manipulation
- Matplotlib, Seaborn – data visualization
- Scikit-learn – ML modeling
- Google Colab – development environment

---

## 🧪 ML Workflow

1. **Data Preprocessing**
   - Dropped missing values
   - Encoded categorical variables using `LabelEncoder`
   - Applied `StandardScaler` for feature scaling

2. **Model Building**
   - Used `RandomForestClassifier` from scikit-learn
   - Trained on 80% of the dataset using `train_test_split`

3. **Model Evaluation**
   - Accuracy score
   - Classification report (precision, recall, F1-score)
   - Confusion matrix
   - Feature importance visualization

---

## 📈 Results

- **Model Used:** Random Forest Classifier
- **Accuracy:** Achieved high classification accuracy
- **Top Features:** Age, cancer stage, lymph node involvement, tumor size

---

## 🚀 Deployment (Suggested)

To deploy this model:
- Save the trained model using `pickle`
- Build a UI with Streamlit or Flask
- Accept patient input and return recurrence prediction

---

## 📂 Folder Structure

```
thyroid-cancer-recurrence/
│
├── UM_thyroid_cancer.ipynb   # Jupyter Notebook
├── thyroid_model.pkl         # Trained model (to be saved)
├── scaler.pkl                # Fitted StandardScaler (to be saved)
├── README.md                 # Project documentation
└── app.py                    # (Optional) Streamlit app for prediction


