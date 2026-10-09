import pandas as pd


# Load the prediction results for all states
df = pd.read_csv('./outputs/states_prediction.csv')
# Load the prediction results for US TS and GP
# df = pd.read_csv('./outputs/US_TS_prediction.csv')
# df = pd.read_csv('./outputs/US_RD_prediction.csv')
# df = pd.read_csv('./outputs/US_GP_prediction.csv')


# Filter columns needed for linear model
linear_rmse_columns = [
    'state', 'mobility_type',
    'RMSE_H(t)_linear_est', 'RMSE_H(t)_linear_pred', 'RMSE_F(t)_linear_est', 'RMSE_F(t)_linear_pred', 'RMSE_h(t)_linear_est', 'RMSE_h(t)_linear_pred',
    'RMSE_H(t)_dqq_est', 'RMSE_H(t)_dqq_pred', 'RMSE_F(t)_dqq_est', 'RMSE_F(t)_dqq_pred', 'RMSE_h(t)_dqq_est', 'RMSE_h(t)_dqq_pred',
    'RMSE_H(t)_lea_est', 'RMSE_H(t)_lea_pred', 'RMSE_F(t)_lea_est', 'RMSE_F(t)_lea_pred', 'RMSE_h(t)_lea_est', 'RMSE_h(t)_lea_pred',
    'RMSE_H(t)_exp_est', 'RMSE_H(t)_exp_pred', 'RMSE_F(t)_exp_est', 'RMSE_F(t)_exp_pred', 'RMSE_h(t)_exp_est', 'RMSE_h(t)_exp_pred',
    'RMSE_H(t)_weibull_est', 'RMSE_H(t)_weibull_pred', 'RMSE_F(t)_weibull_est', 'RMSE_F(t)_weibull_pred', 'RMSE_h(t)_weibull_est', 'RMSE_h(t)_weibull_pred',
    'RMSE_H(t)_log_est', 'RMSE_H(t)_log_pred', 'RMSE_F(t)_log_est', 'RMSE_F(t)_log_pred', 'RMSE_h(t)_log_est', 'RMSE_h(t)_log_pred',
    'RMSE_H(t)_gomp_est', 'RMSE_H(t)_gomp_pred', 'RMSE_F(t)_gomp_est', 'RMSE_F(t)_gomp_pred', 'RMSE_h(t)_gomp_est', 'RMSE_h(t)_gomp_pred',
    'RMSE_H(t)_lsm_est', 'RMSE_H(t)_lsm_pred', 'RMSE_F(t)_lsm_est', 'RMSE_F(t)_lsm_pred', 'RMSE_h(t)_lsm_est', 'RMSE_h(t)_lsm_pred',
    'delta_prime', 'delta_prime_linear', 'delta_prime_dqq', 'delta_prime_lea', 'delta_prime_exp', 'delta_prime_weibull', 'delta_prime_log', 'delta_prime_gomp', 'delta_prime_lsm'
]

# Filter dataframe to only include necessary columns for linear model
df_linear = df[linear_rmse_columns]

# Exclude rows where state is 'US'
df_linear_filtered = df_linear[df_linear['state'] != 'US']
# df_linear_filtered = df_linear[df_linear['state'] == 'US']

# Group data by 'mobility_type' and 'state' and calculate the mean RMSE for each
rmse_summary = df_linear_filtered.groupby(['mobility_type', 'state']).mean()
print(rmse_summary)

# Now calculate the average RMSE across all states for each mobility_type
average_rmse_by_mobility_type = rmse_summary.groupby('mobility_type').mean()

# Output the results
print("Average RMSE for H(t) and F(t) by mobility type:")
print(average_rmse_by_mobility_type)


# Save the RMSE summary and average RMSE by mobility type to CSV files
rmse_summary.to_csv('./outputs/average_rmse_by_state.csv')
average_rmse_by_mobility_type.to_csv('./outputs/average_rmse_by_mobility_type.csv')