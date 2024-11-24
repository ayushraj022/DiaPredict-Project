
# 🩺 Diabetes Prediction Project  

Welcome to my diabetes prediction project! This is a machine learning project that predicts whether a person is diabetic based on some health metrics like 
glucose level, BMI, and more. It's simple, fun, and a great way to dive into some basic data science and machine learning concepts.

## 📂 What’s Inside?
- **Data Collection and Analysis**  
   The project uses the **PIMA Diabetes Dataset** to analyze and predict diabetes outcomes. It has details like glucose levels, insulin levels, and more.  
- **Model Training**  
   I used a Support Vector Machine (SVM) classifier to train the model because SVMs are awesome at handling binary classification tasks.  
- **Accuracy Scores**  
   Checked how well the model works using accuracy scores for both training and testing data. Spoiler: It performs pretty decently!  
- **Prediction System**  
   Built a tiny predictive system that takes input data (like health stats) and tells you if someone might be diabetic.

## 🚀 How It Works
1. Load the dataset.  
2. Standardize the data (make it nice and uniform).  
3. Split the data into training and testing sets.  
4. Train the SVM model using the training set.  
5. Test the model's accuracy on unseen test data.  
6. Finally, use it to predict whether a new input suggests diabetes or not.  

## 🛠️ Tools Used
- **Python** (The superstar language)  
- **NumPy** and **Pandas** for data manipulation.  
- **scikit-learn** for the machine learning part.  
- A sprinkle of love for coding ❤️

## 📝 How to Run It?
1. Clone this repo.  
2. Make sure you have Python and the required libraries installed.  
   - Run `pip install -r requirements.txt` if a `requirements.txt` file is included.  
3. Place the `diabetes.csv` or your csv dataset in the right directory.  
4. Run the script (`python diabetes_pred.py`) or your csv file name, and follow the prompts.  
