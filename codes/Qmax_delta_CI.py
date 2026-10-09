import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def regression_model(x, a, b):
    return a * x + b


# Read the CSV file and specify the columns to use
df = pd.read_csv('./data/states_fitting.csv', usecols=["reverse_delta", "reverse_Qmax", 'mobility_type'])

# Compute the Z-scores for the 'reverse_Qmax' values
df['z_score'] = np.abs(stats.zscore(df['reverse_delta']))

# Set a threshold for the Z-score, typically 3.0
z_threshold = 3.0

# Filter the data to exclude rows with a Z-score above the threshold
df_filtered = df[df['z_score'] < z_threshold]

# Drop the 'z_score' column, as it's no longer needed
df_filtered = df_filtered.drop(columns=['z_score'])


# Proceed with the rest of your regression analysis using the filtered data
mobility_types = df_filtered['mobility_type'].unique()

for mobility_type in mobility_types:
    # Filter the data for the current mobility type
    mobility_df = df_filtered[df_filtered['mobility_type'] == mobility_type]

    # Prepare data for fitting
    delta_values = mobility_df['reverse_delta'].values
    Qmax_values = mobility_df['reverse_Qmax'].values

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

    # Create a new figure for each mobility type
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.2)

    # Separate the data into negative and positive reverse_delta values
    negative_delta = mobility_df[mobility_df['reverse_delta'] < 0]
    positive_delta = mobility_df[mobility_df['reverse_delta'] >= 0]

    # Scatter plot for negative reverse_delta values (colored red)
    ax.scatter(negative_delta['reverse_delta'], negative_delta['reverse_Qmax'],
               color='red', linewidth=6)

    # Scatter plot for positive reverse_delta values (colored green)
    ax.scatter(positive_delta['reverse_delta'], positive_delta['reverse_Qmax'],
               color='green', linewidth=6)

    # Plot the regression line for the current mobility type
    ax.plot(delta_fit, Qmax_fit, color='blue', linewidth=5, label=f'{mobility_type}')

    # Plot the confidence interval for the current mobility type
    ax.fill_between(delta_fit, ci_lower, ci_upper, color='blue', alpha=0.2)

    # Labels and formatting
    ax.set_xlabel('\n\n\n($- \Delta$)', fontsize=30, fontname='Arial')
    ax.set_ylabel('Disruption magnitude ($|Q_{max}|$)', fontsize=30, fontname='Arial')

    ax.set_xlim(delta_values.min() - 5, delta_values.max() + 5)
    ax.set_ylim(Qmax_values.min() - 5, Qmax_values.max() + 5)


    # Add legend
    legend = ax.legend(loc='upper right', fontsize=30)
    # Get the position of the legend box and use it to define the position of the R^2 text
    legend_box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

    r2_x = legend_box.x0 - 0.18  # X position aligned with the left of the legend box
    # Adjust Y position based on mobility_type
    r2_y = legend_box.y0 - 0.05# Y position for 'GP'



    # Add R-squared value below the legend
    ax.text(r2_x, r2_y, f'$R^2 = {r_squared:.2f}$\nE = {a_fit:.2f}', color='blue', fontsize=30, fontname='Arial', transform=ax.transAxes)

    # Adjust tick label size and font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(30)
        label.set_fontname('Arial')

    # Save the plot for each mobility type
    # plt.tight_layout()
    filename = f"./fig/Qmax_delta_CI_{mobility_type}.png"
    plt.savefig(filename, format='png')

    # Show the plot
    plt.show()