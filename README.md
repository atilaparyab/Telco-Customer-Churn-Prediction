# 📊 Telco Customer Churn Prediction & AI Dashboard
This project is an end-to-end Machine Learning solution developed to predict customer churn (churn) for a telecommunications company. The project encompasses all stages, from processing raw data and training the model to transforming it into a web interface (Dashboard) that allows end-users to make predictions.

# 🚀 Project Specifications
* Data Analysis & Cleaning: Eliminating missing data and optimizing numerical data such as TotalCharges.

* Feature Engineering: Deriving new variables such as Monthly_to_Total_Ratio to measure customer loyalty.

* Algorithmic Comparison: Performance analysis of Decision Trees, Support Vector Machines (SVM), and Random Forest models.

* Unbalanced Data Management: Increasing the recall rate of the minority "Abandoned Customer" class using Class Weight techniques.

* Live Application: Converting the trained model into an interactive Web Panel using the Streamlit library.

# 🛠️ Technologies
* Language: Python 3.x

* Data Processing: Pandas, NumPy

* Visualization: Matplotlib, Seaborn

* Machine Learning: Scikit-Learn (Random Forest, SVM, Decision Tree)

* Model Registration: Joblib

* Web Interface: Streamlit

# 💻 Running the Application
To run the application on your local computer:

```

pip install streamlit joblib pandas numpy scikit-learn

```
Start the application:
```

streamlit run app.py

```
