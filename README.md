# **Data Preprocessing and Augmentation Project**  

## **Overview**  
This project focuses on **preparing customer transaction data** for machine learning. Our key tasks include:  
✅ **Cleaning data** (handling missing values, ensuring quality)  
✅ **Augmenting data** (generating synthetic data using SMOTE/interpolation)  
✅ **Merging datasets** (combining related but indirect data sources)  
✅ **Feature engineering** (extracting insights like moving averages & engagement scores)  
✅ **Bonus: Training ML models** (predicting customer spending behavior)  

This project was developed by **Group 17**:  
- **John Akech**  
- **Geu Aguto Garang**  
- **Kuir Juach Kuir Thuch**  

---

## 📌 **Table of Contents**  
1. [Prerequisites](#prerequisites)  
2. [Dataset Description](#dataset-description)  
3. [How to Run the Code](#how-to-run-the-code)  
4. [Project Structure](#project-structure)  
5. [Workflow Breakdown](#workflow-breakdown)  
6. [Outputs](#outputs)  
7. [Challenges & Solutions](#challenges-&-solutions)  

---

## 🛠 **Prerequisites**  
Before running the code, make sure you have:  
- **Python 3.8+**  
- **Jupyter Notebook** (if using `.ipynb`)  
- The following Python libraries:  

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
```  

---

## 📊 **Dataset Description**  
We’re using three datasets:  
📌 **`customer_transactions.csv`** → Transaction details (amounts, categories, ratings)  
📌 **`customer_social_profiles.csv`** → Customer social data  
📌 **`id_mapping.csv`** → Mapping file to link datasets  

Make sure these files are in the same directory as our code!  

---

## 🚀 **How to Run the Code**  

### **Step 1: Clone the Repository**  
If the project is on GitHub, use:  

```bash
git clone git@github.com:John-Akech/Formative-2---Data-Preprocessing_Final-Notebook.git
cd Formative-2---Data-Preprocessing_Final-Notebook
```  

Or download and place the files in your working directory.  

### **Step 2: Open the Notebook**  
For the Jupyter Notebook version:  

```bash
jupyter notebook Formative_2_Data_Preprocessing.ipynb
```  

For the Python script version:  

```bash
python main.py
```  

### **Step 3: Run the Code**  
Go step by step:  
1️⃣ Load `customer_transactions.csv`  
2️⃣ Clean and handle missing values  
3️⃣ Generate synthetic data  
4️⃣ Merge datasets with `id_mapping.csv`  
5️⃣ Perform quality checks  
6️⃣ (Bonus) Train ML models to predict spending  

---

## 📂 **Project Structure**  

```
├── README.md                     # This file  
├── Formative 2 - Data Preprocessing.ipynb  # Notebook version  
├── data_files/                    # Input datasets  
│   ├── customer_transactions.csv  
│   ├── customer_social_profiles.csv  
│   ├── id_mapping.csv  
├── model_files/                   # Trained models  
│   ├── linear_regression_model.pkl  
│   ├── random_forest_model.pkl  
│   ├── xgboost_model.pkl  
├── outputs/                       # Processed data  
│   ├── customer_transactions_augmented.csv  
│   ├── final_customer_data_group17.csv  
│   ├── final_dataset_ready_group17.csv  
```  

---

## 🔬 **Workflow Breakdown**  

### **1️⃣ Data Cleaning**  
✔ Handle missing values (median/mode imputation)  
✔ Encode categorical variables  

### **2️⃣ Data Augmentation**  
✔ Apply random noise to numerical data  
✔ Use **SMOTE** to balance data  

### **3️⃣ Merging Datasets**  
✔ Link datasets using `id_mapping.csv`  
✔ Aggregate duplicate data  

### **4️⃣ Feature Engineering**  
✔ Calculate moving averages  
✔ Generate engagement scores  
✔ Convert text data using **TF-IDF**  

### **5️⃣ ML Model Training (Bonus)**  
✔ Train **Linear Regression, Random Forest, XGBoost**  
✔ Evaluate models (MAE, MSE, R² score)  

---

## 🎯 **Outputs**  

✅ **Augmented Dataset:** `customer_transactions_augmented_<timestamp>.csv`  
✅ **Final Processed Dataset:** `final_customer_data_group17.csv`  
✅ **Trained Models:** Saved in `model_files/`  
✅ **Model Evaluation:** Printed in the notebook/script  

---

## ⚠️ **Challenges & Solutions**  

### 🔹 **Challenge 1: Missing Values in Target Variable**  
**Problem**: Some target values were missing.  
**Fix**: Dropped those rows and handled missing data in features.  

### 🔹 **Challenge 2: Inconsistent Column Names**  
**Problem**: Merging was tricky due to different column names.  
**Fix**: Used `id_mapping.csv` to standardize IDs.  

### 🔹 **Challenge 3: Imbalanced Customer Ratings**  
**Problem**: Biased predictions due to class imbalance.  
**Fix**: Used **SMOTE** to generate synthetic samples.  

### 🔹 **Challenge 4: Handling Text Data**  
**Problem**: The `review_sentiment` column was unstructured.  
**Fix**: Used **TF-IDF** to convert text into numerical features.  


---

### 🎉 That’s it! Happy coding! 🚀  
