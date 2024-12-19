import pandas as pd
import statsmodels.api as sm

# Load your SPX and VIX data
data = pd.read_csv("spx_vix.csv")

# Add a constant term for the intercept
X = sm.add_constant(data["VIX"])
y = data["SPX"]

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

import matplotlib.pyplot as plt

# Plot the scatter plot
plt.scatter(data["VIX"], data["SPX"])

# Plot the regression line
plt.plot(X, model.fittedvalues, color='red')

plt.xlabel("VIX")
plt.ylabel("SPX")
plt.title("SPX vs. VIX Regression")
plt.show()
