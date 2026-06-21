import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data.csv")


df.drop(['mid', 'date'], axis=1, inplace=True)

df.drop(['batsman', 'bowler'], axis=1, inplace=True)

X = df.drop('total', axis=1)
y = df['total']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

categorical_cols = [
    'venue',
    'batting_team',
    'bowling_team'
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_cols
        )
    ],
    remainder='passthrough'
)

models = {

    "Linear Regression":
        LinearRegression(),

    "Lasso":
        Lasso(alpha=0.1),

    "Decision Tree":
        DecisionTreeRegressor(max_depth=10),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        ),

    "SVR":
        SVR(kernel='rbf'),

    "MLP":
        MLPRegressor(
            hidden_layer_sizes=(128,64),
            max_iter=1000,
            random_state=42
        )
}

results = []

best_model = None
best_r2 = -999

for name, model in models.items():

    if name in ["SVR", "MLP"]:

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler(with_mean=False)),
            ('model', model)
        ])

    else:

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)
    print(f"{name} completed!")

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        [name, mae, rmse, r2]
    )

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

print("\nModel Comparison:\n")
print(results_df.sort_values(
    by='R2',
    ascending=False
))

results_df.to_csv(
    "model_results.csv",
    index=False
)

pickle.dump(
    best_model,
    open("best_model.pkl", "wb")
)

print("\nBest Model Saved Successfully")