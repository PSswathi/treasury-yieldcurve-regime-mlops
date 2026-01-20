from xgboost import XGBRegressor

def train(df):
    X=df.drop(columns=["ds","y"])
    y=df["y"]

    model=XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X,y)
    return model
