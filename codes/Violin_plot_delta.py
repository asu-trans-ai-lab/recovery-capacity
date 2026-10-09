import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('./data/states_fitting.csv')

# Convert the 'mobility_type' column to a categorical dtype if it's not already
data['mobility_type'] = data['mobility_type'].astype('category')

# Create a numeric representation of the 'mobility_type' column
data['mobility_type_num'] = data['mobility_type'].cat.codes


sns.violinplot(x='mobility_type_num', y='delta', data=data, inner='box')
# plt.figure(figsize=(16, 12))
plt.subplots_adjust(left=0.15, bottom=0.2)  # Adjust the margins

# Group the data by mobility type
groups = data.groupby('mobility_type_num')

# # Iterate through each group
# for name, group in groups:
#     # Find the index of the highest and lowest delta value in the group
#     max_idx = group['delta'].idxmax()
#     min_idx = group['delta'].idxmin()
#
#     # Get the data for these indices
#     max_point = data.loc[max_idx]
#     min_point = data.loc[min_idx]
#
#     # Annotate the highest and lowest points
#     plt.annotate(max_point['state_code'], (max_point['mobility_type_num'], max_point['delta']), textcoords="offset points", xytext=(10,10), ha='center')
#     plt.annotate(min_point['state_code'], (min_point['mobility_type_num'], min_point['delta']), textcoords="offset points", xytext=(10,-10), ha='center')


# Iterate through each group
for name, group in groups:
    # Calculate the 5th and 95th percentile values
    q5_value = group['delta'].quantile(0.05)
    q95_value = group['delta'].quantile(0.95)

    # Find the nearest data points to the 5th and 95th percentiles
    q5_idx = (group['delta'] - q5_value).abs().idxmin()
    q95_idx = (group['delta'] - q95_value).abs().idxmin()

    # Get the data for these indices
    q5_point = data.loc[q5_idx]
    q95_point = data.loc[q95_idx]

    # Annotate the 5th and 95th percentile points
    plt.annotate(q5_point['state_code'], (q5_point['mobility_type_num'], q5_point['delta']), textcoords="offset points",
                 xytext=(10, -10), ha='center')
    plt.annotate(q95_point['state_code'], (q95_point['mobility_type_num'], q95_point['delta']), textcoords="offset points",
                 xytext=(10, 10), ha='center')


plt.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)  # Add a dashed line at y=0
# Set the labels and title
plt.xlabel('Mobility Type', fontsize=12, labelpad=8)
plt.ylabel('Transition Loss/Gain (%)', fontsize=12, labelpad=8)
# plt.title('Violin plot of mobility types and transition status')

# Set the x-axis tick labels to the category names
plt.xticks(ticks=range(len(data['mobility_type'].cat.categories)), labels=data['mobility_type'].cat.categories, fontsize=11)
plt.tick_params(axis='y', labelsize=11)
# Assuming 'RD' is for "Typical Transition Gain" and 'WP' is for "Typical Transition Loss"
typical_gain_code = data['mobility_type'].cat.categories.get_loc('RD')
typical_loss_code = data['mobility_type'].cat.categories.get_loc('WP')

# Adjusting text location by specifying offsets and alignment
# If the text is overlapping with plot elements, adjust the xytext values
plt.annotate("Typical\nTransition\nGain",
             (typical_gain_code, 0),
             textcoords="offset points",
             xytext=(0, 45),  # Moves text 10 points upward
             ha='center',
             va='bottom',
             fontsize=11,
             bbox=dict(facecolor='white', alpha=0.5))

plt.annotate("Typical\nTransition\nLoss",
             (typical_loss_code, 0),
             textcoords="offset points",
             xytext=(-3.5, 50),  # Moves text 10 points downward
             ha='center',
             va='top',
             fontsize=11,
             bbox=dict(facecolor='white', alpha=0.5))

# # Save the plot
# Save the plot
plt.tight_layout()
filename = f"./fig/Violin plot of delta.png"
plt.savefig(filename, format='png', dpi=1000)  # Increase DPI for higher resolution

# Show the plot
plt.show()
