# IPL 1st Innings Score Prediction using Machine Learning

This project predicts the **final score of the first innings of an IPL match** based on historical ball-by-ball data from IPL seasons (2008–2017).  
The model uses machine learning regression algorithms (Random Forest, Decision Tree, Linear Regression, Neural Networks, etc.) to achieve high accuracy.

---

## **Project Overview**
- **Goal:** Predict the total score of a team batting first in an IPL match.
- **Approach:** 
  - Data cleaning and feature engineering (removing irrelevant columns, filtering teams, encoding).
  - Model training using multiple ML regressors.
  - Selecting the best-performing model based on test set accuracy.
- **Dataset:** [IPL Ball-by-Ball Data (2008–2017)](https://www.kaggle.com/yuvrajdagur/ipl-dataset-season-2008-to-2017).

---

## **Features Used for Prediction**
- Batting Team
- Bowling Team
- Current Runs
- Wickets Fallen
- Overs Completed
- Runs Scored in Last 5 Overs
- Wickets Fallen in Last 5 Overs

---

## **Tech Stack**
- **Programming Language:** Python 3.x
- **Libraries:**
  - `pandas`, `numpy` – Data handling
  - `matplotlib`, `seaborn` – Visualizations
  - `scikit-learn` – ML model training & evaluation
  - `joblib` – Model saving/loading

---

## **Model Pipeline**
1. **Data Preprocessing:**
   - Dropped irrelevant columns (`mid`, `date`, `venue`, `batsman`, `bowler`, `striker`, `non-striker`).
   - Filtered consistent teams.
   - Applied **Label Encoding** and **One-Hot Encoding** for categorical variables.
2. **Model Training:**
   - Trained multiple regressors:
     - Decision Tree
     - Linear Regression
     - Random Forest (Best Performer)
     - Lasso Regression
     - Support Vector Regression (SVR)
     - Neural Network (MLPRegressor)
3. **Evaluation:**
   - Metrics: **R² score**, **MAE**, **RMSE**
   - Random Forest showed the best performance.
4. **Prediction Function:**
   - `predict_score()` – Takes match parameters and outputs predicted final score.
5. **Model Export:**
   - Saved as `best_model.pkl` for easy deployment.

---

## **Visualizations**
The notebook includes:
- Distribution of runs
- Correlation heatmaps
- Runs vs overs trends
- Model performance comparison (bar charts)

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/ipl-score-prediction.git
cd ipl-score-prediction
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Code
Open the Jupyter Notebook or Python script:

bash
Copy
Edit
jupyter notebook IPL_Prediction_Model.ipynb
Or:

bash
Copy
Edit
python ipl_score_predictor.py
How to Use the Prediction Function
python
Copy
Edit
from joblib import load
import numpy as np

model = load("best_model.pkl")

def predict_score(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5):
    teams = ['Chennai Super Kings','Delhi Daredevils','Kings XI Punjab',
             'Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals',
             'Royal Challengers Bangalore','Sunrisers Hyderabad']
    arr = [1 if batting_team == t else 0 for t in teams]
    arr += [1 if bowling_team == t else 0 for t in teams]
    arr += [runs, wickets, overs, runs_last_5, wickets_last_5]
    arr = np.array([arr])
    return int(round(model.predict(arr)[0]))

# Example
print(predict_score("Mumbai Indians", "Chennai Super Kings", 85, 2, 11.3, 40, 1))
Future Improvements
Include venue and player performance stats for better accuracy.

Deploy the model as a web app using Streamlit or Flask.

Add real-time data fetching for live IPL matches.

**Amaan Suheab**  
GitHub: [https://github.com/amaansuheab](https://github.com/amaansuheab)

---

## **License**
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.


Contributions and suggestions are welcome!
