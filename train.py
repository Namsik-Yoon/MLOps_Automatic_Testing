from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import os
import pickle

X, y = make_regression(10000,n_features = 10)

# Train a model
reg = ElasticNet().fit(X, y.ravel())
# Print out training r2
print(reg.score(X,y.ravel() ))

# Write the model to a file
if not os.path.isdir("models/"):
    os.mkdir("models")

filename = 'models/model.pkl'
pickle.dump(reg, open(filename, 'wb'))

with open("model.txt", 'w') as outfile:
    outfile.write(str(reg))