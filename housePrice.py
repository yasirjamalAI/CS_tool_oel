mport pdb

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the California housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['Target'] = data.target
#print(df)

# Split the data into features (X) and target variable (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

# Take input from the user to predict a new instance

user_input = []
for feature in X.columns:
    value = float(input(f"Enter value for {feature}: "))
    user_input.append(value)

# Convert the user input to a NumPy array
user_input = np.array(user_input).reshape(1, -1)

# Make a prediction using the trained model

prediction = model.predict(user_input)
#pdb.set_track()
if(prediction<0):
   print("incorrect inputs.Out Of range")
else:
   print(f'Predicted House Price: ${prediction[0]:,.2f}')