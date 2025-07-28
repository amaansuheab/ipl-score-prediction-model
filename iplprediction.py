import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('ipl_colab.csv')
data = data.drop(['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker'], axis=1)
teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']
data = data[(data['batting_team'].isin(teams)) & (data['bowling_team'].isin(teams))]
data = data[data['overs'] >= 5.0]

le = LabelEncoder()
for col in ['batting_team', 'bowling_team']:
    data[col] = le.fit_transform(data[col])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
data = np.array(ct.fit_transform(data))

cols = ['batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
        'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
        'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
        'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab',
        'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
        'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad',
        'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']

df = pd.DataFrame(data, columns=cols)
X = df.drop(['total'], axis=1)
y = df['total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "DecisionTree": DecisionTreeRegressor(),
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "Lasso": LassoCV(),
    "SVR": SVR(),
    "NeuralNet": MLPRegressor(activation='logistic', max_iter=500)
}

best_model = None
best_score = -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_model = model

print("Best Model:", best_model.__class__.__name__)
print("R2 Score:", best_score)
print("MAE:", mean_absolute_error(y_test, best_model.predict(X_test)))
print("RMSE:", np.sqrt(mean_squared_error(y_test, best_model.predict(X_test))))

dump(best_model, "best_model.pkl")
