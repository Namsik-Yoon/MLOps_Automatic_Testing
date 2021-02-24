import pickle
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error as mse 
import json

model = pickle.load(open("models/model.pkl", "rb"))
model_name = open("model.txt", 'r').read()

# Generate some data for validation
X_test, y = make_regression(1000,n_features = 10)

# Test on the model
y_hat = model.predict(X_test)

# squared=Fase -> False returns RMSE value
metric = mse(y, y_hat, squared=False)

with open("metrics.json", 'w') as outfile:
    json.dump({ "model_name": model_name, "RMSE": rmse}, outfile)


