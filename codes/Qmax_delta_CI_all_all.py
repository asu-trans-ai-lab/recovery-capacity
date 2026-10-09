import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.cm import get_cmap


def regression_model(x, a, b):
    return a * x + b


# Read the CSV file and specify the columns to use
df = pd.read_csv('./data/states_fitting.csv', usecols=["delta", "Qmax", 'mobility_type'])

# Compute the Z-scores for the 'Qmax' values
df['z_score'] = np.abs(stats.zscore(df['delta']))

# Set a threshold for the Z-score, typically 3.0
z_threshold = 3.0

# Filter the data to exclude rows with a Z-score above the threshold
df_filtered = df[df['z_score'] < z_threshold]

# Drop the 'z_score' column, as it's no longer needed
df_filtered = df_filtered.drop(columns=['z_score'])

# Create a new figure for all mobility types
fig, ax = plt.subplots(figsize=(16, 12))
plt.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.2)

# Get the list of unique mobility types
mobility_types = df_filtered['mobility_type'].unique()

# Set up a colormap to assign different colors to each mobility type
cmap = get_cmap('tab10')  # This colormap provides distinct colors for up to 10 categories
# List of different markers for different mobility types
markers = ['o', '^', 's', 'd', 'P', 'X', '*', 'v', 'h', 'D']  # Add as many as needed

# Proceed with the rest of your regression analysis using the filtered data
for i, mobility_type in enumerate(mobility_types):
    # Filter the data for the current mobility type
    mobility_df = df_filtered[df_filtered['mobility_type'] == mobility_type]

    # Prepare data for fitting
    delta_values = mobility_df['delta'].values
    Qmax_values = mobility_df['Qmax'].values

    # Perform the curve fitting (regressing Qmax on delta)
    params, covariance = curve_fit(regression_model, delta_values, Qmax_values, p0=[1, 1])

    # Extracting the parameters a, b
    a_fit, b_fit = params

    # Generate fitted values using the model function and the optimized parameters
    delta_fit = np.linspace(delta_values.min(), delta_values.max(), 100)
    Qmax_fit = regression_model(delta_fit, a_fit, b_fit)

    # Calculate the fitted Qmax_values at the exact data points for residual calculation
    Qmax_fit_exact = regression_model(delta_values, a_fit, b_fit)

    # Calculate R-squared for the fitted model using the exact data points
    residuals = Qmax_values - Qmax_fit_exact
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Qmax_values - np.mean(Qmax_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate the standard error of the parameters and the predictions
    perr = np.sqrt(np.diag(covariance))
    mean_x = np.mean(delta_values)
    n = len(delta_values)

    # 95% confidence interval (two-tailed)
    t = stats.t.ppf(0.975, n - 2)
    se_fit = np.sqrt(np.sum((Qmax_values - regression_model(delta_values, a_fit, b_fit)) ** 2) / (n - 2))
    ci = t * se_fit * np.sqrt(1 / n + (delta_fit - mean_x) ** 2 / np.sum((delta_values - mean_x) ** 2))

    # Calculate confidence interval bounds
    ci_upper = Qmax_fit + ci
    ci_lower = Qmax_fit - ci

    # Assign a color and marker from the colormap and marker list
    color = cmap(i % 10)  # Use mod to wrap around the colormap if there are more than 10 mobility types
    marker = markers[i % len(markers)]  # Use mod to wrap around the marker list

    # Scatter plot for the current mobility type with different markers
    ax.scatter(mobility_df['delta'], mobility_df['Qmax'], label=mobility_type,
               color=color, marker=marker, s=100, linewidth=3)

    # Plot the regression line for the current mobility type
    ax.plot(delta_fit, Qmax_fit, color=color, linewidth=5)

    # Plot the confidence interval for the current mobility type
    ax.fill_between(delta_fit, ci_lower, ci_upper, color=color, alpha=0.2)

# Labels and formatting
ax.set_xlabel('\n\n\n($- \Delta$)', fontsize=30, fontname='Arial')
ax.set_ylabel('Disruption magnitude ($|Q_{max}|$)', fontsize=30, fontname='Arial')

# Set the limits for x and y axes based on the overall data
ax.set_xlim(df_filtered['delta'].min() - 5, df_filtered['delta'].max() + 5)
ax.set_ylim(df_filtered['Qmax'].min() - 5, df_filtered['Qmax'].max() + 5)

# Add legend
ax.legend(loc='lower right', fontsize=20)

# Adjust tick label size and font
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(20)
    label.set_fontname('Arial')

# Save the plot for all mobility types
filename = "./fig/Qmax_delta_CI_all_mobility_types_all_regression.png"
plt.savefig(filename, format='png')

# Show the plot
plt.show()