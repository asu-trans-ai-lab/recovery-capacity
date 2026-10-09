import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.cm import get_cmap


def regression_model(x, a, b):
    return a * x + b


# Read the CSV file and include the 'state' column
df = pd.read_csv('./data/states_fitting.csv', usecols=["reverse_delta", "reverse_Qmax", 'state'])

# Compute the Z-scores for the 'reverse_Qmax' values
df['z_score'] = np.abs(stats.zscore(df['reverse_delta']))

# Set a threshold for the Z-score, typically 3.0
z_threshold = 3.0

# Filter the data to exclude rows with a Z-score above the threshold
df_filtered = df[df['z_score'] < z_threshold]

# Drop the 'z_score' column, as it's no longer needed
df_filtered = df_filtered.drop(columns=['z_score'])

# Get the list of unique states
states = df_filtered['state'].unique()

# Initialize an empty list to store results
regression_results = []

# Set up a colormap to assign different colors to each state
cmap = get_cmap('tab10')  # This colormap provides distinct colors for up to 10 categories
markers = ['o', '^', 's', 'd', 'P', 'X', '*', 'v', 'h', 'D']  # Add as many as needed

# Proceed with the rest of your regression analysis using the filtered data
for i, state in enumerate(states):
    # Create a new figure for each state
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter the data for the current state
    state_df = df_filtered[df_filtered['state'] == state]

    # Prepare data for fitting
    delta_values = state_df['reverse_delta'].values
    Qmax_values = state_df['reverse_Qmax'].values

    # Perform the curve fitting (regressing Qmax on delta)
    params, covariance = curve_fit(regression_model, delta_values, Qmax_values, p0=[1, 1])

    # Extracting the parameters a, b
    a_fit, b_fit = params

    # Store the results in a list (state, slope, intercept)
    regression_results.append([state, a_fit, b_fit])

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

    # Store R² value along with the state, slope, and intercept
    regression_results[-1].append(r_squared)

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
    color = cmap(i % 10)
    marker = markers[i % len(markers)]

    # Scatter plot for the current state with different markers
    ax.scatter(state_df['reverse_delta'], state_df['reverse_Qmax'], label=state,
               color=color, marker=marker, s=100, linewidth=3)

    # Plot the regression line for the current state
    ax.plot(delta_fit, Qmax_fit, color=color, linewidth=5)

    # Plot the confidence interval for the current state
    ax.fill_between(delta_fit, ci_lower, ci_upper, color=color, alpha=0.2)

    # Labels and formatting
    ax.set_xlabel('Reverse Delta', fontsize=20, fontname='Arial')
    ax.set_ylabel('Disruption Magnitude (|Qmax|)', fontsize=20, fontname='Arial')

    # Set the limits for x and y axes
    ax.set_xlim(state_df['reverse_delta'].min() - 5, state_df['reverse_delta'].max() + 5)
    ax.set_ylim(state_df['reverse_Qmax'].min() - 5, state_df['reverse_Qmax'].max() + 5)

    # Add legend
    ax.legend(loc='lower right', fontsize=15)

    # Adjust tick label size and font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)
        label.set_fontname('Arial')

    # Display the parameters a, b, and R² on the plot
    ax.text(0.05, 0.95, f'Slope E: {a_fit:.2f}\nIntercept B: {b_fit:.2f}\nR²: {r_squared:.2f}',
            transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0))

    # Save the plot for the current state
    filename = f"./fig/reverse_Qmax_delta_CI_all_states_all_regression_with_overall/reverse_Qmax_delta_CI_state_{state}.png"
    plt.savefig(filename, format='png')

    # Show the plot (optional, remove if only saving)
    plt.show()

# Convert regression results to a DataFrame and output to console and CSV
regression_df = pd.DataFrame(regression_results, columns=['state', 'slope', 'intercept', 'r_squared'])
print(regression_df)

# Save the results to a CSV file
regression_df.to_csv('./outputs/Qmax_delta_regression_results_by_state_with_R2.csv', index=False)