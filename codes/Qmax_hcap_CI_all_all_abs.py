import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.cm import get_cmap


def regression_model(x, a, b):
    return a * x + b


# Read the CSV file and specify the columns to use
df = pd.read_csv('./data/states_fitting.csv', usecols=["hcap", "reverse_Qmax", 'mobility_type'])

# Compute the Z-scores for the 'reverse_Qmax' values
df['z_score'] = np.abs(stats.zscore(df['hcap']))

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

# Initialize an empty list to store the regression results
regression_results = []

# Proceed with the rest of your regression analysis using the filtered data
for i, mobility_type in enumerate(mobility_types):
    # Filter the data for the current mobility type
    mobility_df = df_filtered[df_filtered['mobility_type'] == mobility_type]

    # Prepare data for fitting
    hcap_values = mobility_df['hcap'].values
    reverse_Qmax_values = mobility_df['reverse_Qmax'].values

    # Perform the curve fitting (regressing reverse_Qmax on hcap)
    params, covariance = curve_fit(regression_model, hcap_values, reverse_Qmax_values, p0=[1, 1])

    # Extracting the parameters a, b
    a_fit, b_fit = params

    # Generate fitted values using the model function and the optimized parameters
    hcap_fit = np.linspace(hcap_values.min(), hcap_values.max(), 100)
    reverse_Qmax_fit = regression_model(hcap_fit, a_fit, b_fit)

    # Calculate the fitted reverse_Qmax_values at the exact data points for residual calculation
    reverse_Qmax_fit_exact = regression_model(hcap_values, a_fit, b_fit)

    # Calculate R-squared for the fitted model using the exact data points
    residuals = reverse_Qmax_values - reverse_Qmax_fit_exact
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((reverse_Qmax_values - np.mean(reverse_Qmax_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Store the mobility type, slope (a_fit), intercept (b_fit), and R-squared (r_squared)
    regression_results.append([mobility_type, a_fit, b_fit, r_squared])

    # Calculate the standard error of the parameters and the predictions
    perr = np.sqrt(np.diag(covariance))
    mean_x = np.mean(hcap_values)
    n = len(hcap_values)

    # 95% confidence interval (two-tailed)
    t = stats.t.ppf(0.975, n - 2)
    se_fit = np.sqrt(np.sum((reverse_Qmax_values - regression_model(hcap_values, a_fit, b_fit)) ** 2) / (n - 2))
    ci = t * se_fit * np.sqrt(1 / n + (hcap_fit - mean_x) ** 2 / np.sum((hcap_values - mean_x) ** 2))

    # Calculate confidence interval bounds
    ci_upper = reverse_Qmax_fit + ci
    ci_lower = reverse_Qmax_fit - ci

    # Assign a color and marker from the colormap and marker list
    color = cmap(i % 10)  # Use mod to wrap around the colormap if there are more than 10 mobility types
    marker = markers[i % len(markers)]  # Use mod to wrap around the marker list

    # Scatter plot for the current mobility type with different markers
    ax.scatter(mobility_df['hcap'], mobility_df['reverse_Qmax'], label=mobility_type,
               color=color, marker=marker, s=100, linewidth=3)

    # Plot the regression line for the current mobility type
    ax.plot(hcap_fit, reverse_Qmax_fit, color=color, linewidth=5)

    # Plot the confidence interval for the current mobility type
    ax.fill_between(hcap_fit, ci_lower, ci_upper, color=color, alpha=0.2)

# Perform overall regression using all data points
all_hcap_values = df_filtered['hcap'].values
all_reverse_Qmax_values = df_filtered['reverse_Qmax'].values

# Perform the overall curve fitting (regressing reverse_Qmax on hcap for all data)
overall_params, overall_covariance = curve_fit(regression_model, all_hcap_values, all_reverse_Qmax_values, p0=[1, 1])

# Extract the parameters for the overall regression
overall_a_fit, overall_b_fit = overall_params

# Generate fitted values for the overall regression
overall_hcap_fit = np.linspace(all_hcap_values.min(), all_hcap_values.max(), 100)
overall_reverse_Qmax_fit = regression_model(overall_hcap_fit, overall_a_fit, overall_b_fit)

# Plot the overall regression line in grey
ax.plot(overall_hcap_fit, overall_reverse_Qmax_fit, color='grey', linewidth=8, alpha=0.8, linestyle='--', label='Overall')

# Calculate R-squared for the overall regression
overall_reverse_Qmax_fit_exact = regression_model(all_hcap_values, overall_a_fit, overall_b_fit)
overall_residuals = all_reverse_Qmax_values - overall_reverse_Qmax_fit_exact
overall_ss_res = np.sum(overall_residuals ** 2)
overall_ss_tot = np.sum((all_reverse_Qmax_values - np.mean(all_reverse_Qmax_values)) ** 2)
overall_r_squared = 1 - (overall_ss_res / overall_ss_tot)

# Labels and formatting
ax.set_xlabel('Healing capacity ($h_{cap}$)', fontsize=30, fontname='Arial')
ax.set_ylabel('Disruption magnitude ($|Q_{max}|$)', fontsize=30, fontname='Arial')

# Set the limits for x and y axes based on the overall data
ax.set_ylim(df_filtered['reverse_Qmax'].min() - 5, df_filtered['reverse_Qmax'].max() + 5)

# Add legend
ax.legend(loc='lower right', fontsize=20)

# Adjust tick label size and font
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(20)
    label.set_fontname('Arial')

# Save the plot for all mobility types
filename = "./fig/reverse_Qmax_hcap_CI_all_mobility_types_all_regression_with_overall.png"
plt.savefig(filename, format='png')

# Show the plot
plt.show()

# Convert the regression results to a DataFrame and display it
regression_df = pd.DataFrame(regression_results, columns=['mobility_type', 'slope', 'intercept', 'r_squared'])
print(regression_df)

# Save the regression results to a CSV file
regression_df.to_csv('./outputs/Qmax_hcap_regression_results_with_R2.csv', index=False)

# Output overall regression slope, intercept, and R²
print(f'Overall Regression Slope: {overall_a_fit}, Intercept: {overall_b_fit}, R²: {overall_r_squared}')

# Save overall regression results to a CSV file
overall_regression_df = pd.DataFrame([{'slope': overall_a_fit, 'intercept': overall_b_fit, 'r_squared': overall_r_squared}])
overall_regression_df.to_csv('./outputs/overall_Qmax_hcap_regression_results_with_R2.csv', index=False)