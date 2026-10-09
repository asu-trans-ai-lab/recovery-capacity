import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Define the regression model function
def regression_model(x, a, b):
    return a * x + b

# Read the CSV file and specify the columns to use
df = pd.read_csv('./data/states_fitting.csv', usecols=["delta", "Qmax"])

# Compute the Z-scores for the 'delta' values to filter outliers
df['z_score'] = np.abs(stats.zscore(df['delta']))

# Set a threshold for the Z-score, typically 3.0
z_threshold = 3.0

# Filter the data to exclude rows with a Z-score above the threshold
df_filtered = df[df['z_score'] < z_threshold]

# Drop the 'z_score' column, as it's no longer needed
df_filtered = df_filtered.drop(columns=['z_score'])

# Separate the data into negative and positive delta values
negative_delta_df = df_filtered[df_filtered['delta'] < 0]
positive_delta_df = df_filtered[df_filtered['delta'] >= 0]

# Create a new figure for the regression analysis
fig, ax = plt.subplots(figsize=(16, 12))
plt.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.2)

# Perform regression for negative delta values
delta_values_neg = negative_delta_df['delta'].values
Qmax_values_neg = negative_delta_df['Qmax'].values

params_neg, covariance_neg = curve_fit(regression_model, delta_values_neg, Qmax_values_neg, p0=[1, 1])
a_fit_neg, b_fit_neg = params_neg

# Generate fitted values for negative delta
delta_fit_neg = np.linspace(delta_values_neg.min(), delta_values_neg.max(), 100)
Qmax_fit_neg = regression_model(delta_fit_neg, a_fit_neg, b_fit_neg)

# Perform regression for positive delta values
delta_values_pos = positive_delta_df['delta'].values
Qmax_values_pos = positive_delta_df['Qmax'].values

params_pos, covariance_pos = curve_fit(regression_model, delta_values_pos, Qmax_values_pos, p0=[1, 1])
a_fit_pos, b_fit_pos = params_pos

# Generate fitted values for positive delta
delta_fit_pos = np.linspace(delta_values_pos.min(), delta_values_pos.max(), 100)
Qmax_fit_pos = regression_model(delta_fit_pos, a_fit_pos, b_fit_pos)

# Scatter plot for negative delta values
ax.scatter(negative_delta_df['delta'], negative_delta_df['Qmax'],
           color='red', label='Negative $\Delta$', linewidth=6)

# Scatter plot for positive delta values
ax.scatter(positive_delta_df['delta'], positive_delta_df['Qmax'],
           color='green', label='Positive $\Delta$', linewidth=6)

# Plot the regression line for negative delta
ax.plot(delta_fit_neg, Qmax_fit_neg, color='blue', label='Negative $\Delta$ fit', linewidth=3)

# Plot the regression line for positive delta
ax.plot(delta_fit_pos, Qmax_fit_pos, color='orange', label='Positive $\Delta$ fit', linewidth=3)

# Labels and formatting
ax.set_xlabel('\n\n\n($\Delta$)', fontsize=30, fontname='Arial')
ax.set_ylabel('Disruption magnitude ($Q_{max}$)', fontsize=30, fontname='Arial')

# Set the limits for x and y axes based on the overall data
ax.set_xlim(df_filtered['delta'].min() - 5, df_filtered['delta'].max() + 5)
ax.set_ylim(df_filtered['Qmax'].min() - 5, df_filtered['Qmax'].max() + 5)

# Add legend
ax.legend(loc='upper right', fontsize=20)

# Adjust tick label size and font
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(20)
    label.set_fontname('Arial')

# Save the plot
filename = "./fig/Qmax_delta_CI_neg_pos.png"
plt.savefig(filename, format='png')

# Show the plot
plt.show()