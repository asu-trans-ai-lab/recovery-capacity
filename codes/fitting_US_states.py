import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.special as sp
import scipy.integrate as integrate
from scipy.special import gammainc, gamma
from matplotlib.ticker import FuncFormatter
from numpy.polynomial import Polynomial
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d


############# Linear F(t): F(t2)=0
def F_L(relative_t, linear_a, relative_t4):
    F_t = linear_a * relative_t
    Ft4_value = linear_a * relative_t4
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))

def H_L(relative_t, linear_a, relative_t4):
    H_t = -np.log(1 - linear_a * relative_t)
    Ht4_value = -np.log(1 - linear_a * relative_t4)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, H_t))

def h_L(relative_t, linear_a, relative_t4):
    h_t = linear_a / (1 - linear_a * relative_t)
    ht4_value = linear_a / (1 - linear_a * relative_t4)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, h_t))

def objective_L(params, relative_t, F_observed, h_observed, relative_t4, theta):
    linear_a = params
    F_estimated = F_L(relative_t, linear_a, relative_t4)
    h_estimated = h_L(relative_t, linear_a, relative_t4)
    return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)

############# DQQ
def F_DQQ(relative_t, _beta, relative_t4):
    F_t = - _beta * relative_t * relative_t * (relative_t4 / 2 - relative_t / 3)
    Ft4_value = - _beta * relative_t4 * relative_t4 * (relative_t4 / 2 - relative_t4 / 3)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))

def H_DQQ(relative_t, _beta, relative_t4):
    S_t = 1 + _beta * relative_t * relative_t * (relative_t4 / 2 - relative_t / 3)
    H_t = -np.log(S_t)
    St4_value = 1 + _beta * relative_t4 * relative_t4 * (relative_t4 / 2 - relative_t4 / 3)
    Ht4_value = -np.log(St4_value)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, Ht4_value))

def h_DQQ(relative_t, _beta, relative_t4):
    f_t = _beta * relative_t * (relative_t - relative_t4)
    S_t = 1 + _beta * relative_t * relative_t * (relative_t4 / 2 - relative_t / 3)
    h_t = f_t / S_t
    f_t4 = _beta * relative_t4 * (relative_t4 - relative_t4)
    S_t4 = 1 + _beta * relative_t4 * relative_t4 * (relative_t4 / 2 - relative_t4 / 3)
    ht4_value = f_t4 / S_t4
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, h_t))

def objective_DQQ(params, relative_t, F_observed, h_observed, relative_t4, theta):
    _beta = params
    F_estimated = F_DQQ(relative_t, _beta, relative_t4)
    h_estimated = h_DQQ(relative_t, _beta, relative_t4)
    return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)

# ############# LEA with c=1
# # Define the function h(t)
# def h_LEA(relative_t, a, b, relative_t4, h_b):
#     a = (h_b / b) * np.exp(1)
#     h_t = a * relative_t * np.exp(-relative_t/b)
#     return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, h_t))
#
# def H_LEA(relative_t, a, b, relative_t4, h_b):
#     a = (h_b / b) * np.exp(1)
#     H_t = a * b * b - (a * b * b + a * b * relative_t) * np.exp(-relative_t / b)
#     Ht4_value = a * b * b - (a * b * b + a * b * relative_t4) * np.exp(-relative_t4 / b)
#     return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, H_t))
#
# def F_LEA(relative_t, a, b, relative_t4, h_b):
#     a = (h_b / b) * np.exp(1)
#     F_t = 1 - np.exp(-a * b**2 + (a * b**2 + a * b * relative_t) * np.exp(-relative_t / b))
#     Ft4_value = 1 - np.exp(-a * b**2 + (a * b**2 + a * b * relative_t4) * np.exp(-relative_t4 / b))
#     return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))
#
# def objective_LEA(params, relative_t, F_observed, h_observed, relative_t4, theta, h_b):
#     b = params
#     a = (h_b / b) * np.exp(1)
#     F_estimated = F_LEA(relative_t, a, b, relative_t4, h_b)
#     h_estimated = h_LEA(relative_t, a, b, relative_t4, h_b)
#     return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)

############# LEA with c=2
# Define the function h(t)
def h_LEA(relative_t, a, b, relative_t4, h_b):
    a = (h_b / b) / np.exp(-1/2)
    h_t = a * relative_t * np.exp(-0.5 * (relative_t / b)**2)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, 0))

def H_LEA(relative_t, a, b, relative_t4, h_b):
    a = (h_b / b) / np.exp(-1 / 2)
    H_t = a * b**2 * (1 - np.exp(-0.5 * (relative_t / b)**2))
    Ht4_value = a * b**2 * (1 - np.exp(-0.5 * (relative_t4 / b)**2))
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, H_t))

def F_LEA(relative_t, a, b, relative_t4, h_b):
    a = (h_b / b) / np.exp(-1 / 2)
    F_t = 1 - np.exp(-a * b**2 * (1 - np.exp(-0.5 * (relative_t / b)**2)))
    Ft4_value = 1 - np.exp(-a * b**2 * (1 - np.exp(-0.5 * (relative_t4 / b)**2)))
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))

def objective_LEA(params, relative_t, F_observed, h_observed, relative_t4, theta, h_b):
    b = params
    a = (h_b / b) / np.exp(-1/2)
    F_estimated = F_LEA(relative_t, a, b, relative_t4, h_b)
    h_estimated = h_LEA(relative_t, a, b, relative_t4, h_b)
    return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)

# ############# LEA with c=3
# # Define the function h(t)
# def h_LEA(relative_t, a, b, relative_t4, h_b):
#     a = (h_b / b) / np.exp(-1/3)
#     h_t = a * relative_t * np.exp(-(1/3) * (relative_t / b)**3)
#     return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, h_t))
#
# def H_LEA(relative_t, a, b, relative_t4, h_b):
#     a = (h_b / b) / np.exp(-1 / 3)
#     H_t = (a * b**3 / 3) * gamma(2/3) * gammainc(2/3, ((relative_t / b) ** 3) / 3)
#     Ht4_value = (a * b**3 / 3) * gamma(2/3) * gammainc(2/3, ((relative_t4 / b) ** 3) / 3)
#     return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, H_t))
#
# def F_LEA(relative_t, a, b, relative_t4, h_b):
#     a = (h_b / b) / np.exp(-1 / 3)
#     F_t = 1 - np.exp(-(a * b**3 / 3) * gamma(2/3) * gammainc(2/3, ((relative_t / b) ** 3) / 3))
#     Ft4_value = 1 - np.exp(-(a * b**3 / 3) * gamma(2/3) * gammainc(2/3, ((relative_t4 / b) ** 3) / 3))
#     return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))
#
# def objective_LEA(params, relative_t, F_observed, h_observed, relative_t4, theta, h_b):
#     b = params
#     a = (h_b / b) / np.exp(-1/3)
#     F_estimated = F_LEA(relative_t, a, b, relative_t4, h_b)
#     h_estimated = h_LEA(relative_t, a, b, relative_t4, h_b)
#     return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)

# ############# LEA with a, b, c
# # Define the function h(t)
# def h_LEA(relative_t, a, b, c, relative_t4):
#     h_t = a * relative_t * np.exp(-1/c * (relative_t/b)**c)
#     return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, h_t))
#
# def H_LEA(relative_t, a, b, c, relative_t4):
#     H_t = (a * (c**(2/c - 1)) * b**2) * (gammainc(2/c, 0) - gammainc(2/c, (1/c)*(relative_t/b)**c))
#     return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, H_t))
#
# def F_LEA(relative_t, a, b, c, relative_t4):
#     F_t = 1 - np.exp(-a * b**2 + (a * b**2 + a * b * relative_t) * np.exp(-relative_t / b))
#     Ft4_value = 1 - np.exp(-a * b**2 + (a * b**2 + a * b * relative_t4) * np.exp(-relative_t4 / b))
#     return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))


# Define the polynomial function for H(t)
def H_smoothing (relative_t, smoothing_a, relative_t4, relative_t6):
    return smoothing_a * (relative_t**4 / 4 - (relative_t4 + relative_t6) * relative_t**3 / 3 + relative_t4 * relative_t6 * relative_t**2 / 2)

# def H_smoothing (relative_t, smoothing_a, relative_t4, relative_t6, relative_t7):
#     return smoothing_a * (relative_t**5 / 5 - (relative_t4 + relative_t6 + relative_t7) * (relative_t**4 / 4) + (relative_t4 * relative_t6 + relative_t4 * relative_t7 + relative_t6 * relative_t7) * relative_t**3 / 3 - relative_t4 * relative_t6 * relative_t7 * relative_t**2 / 2)


############# exp
def F_exp(relative_t, exp_lambda, relative_t4):
    F_t = 1 - np.exp( - exp_lambda * relative_t)
    Ft4_value = 1 - np.exp( - exp_lambda * relative_t4)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))

def H_exp(relative_t, exp_lambda, relative_t4):
    H_t = exp_lambda * relative_t
    Ht4_value = exp_lambda * relative_t4
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, H_t))

def h_exp(relative_t, exp_lambda, relative_t4):
    h_t = exp_lambda
    ht4_value = exp_lambda
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, h_t))

def objective_exp(params, relative_t, F_observed, h_observed, relative_t4, theta):
    exp_lambda = params
    F_estimated = F_exp(relative_t, exp_lambda, relative_t4)
    h_estimated = h_exp(relative_t, exp_lambda, relative_t4)
    return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)

############# weibull
def h_weibull(relative_t, weibull_lambda, weibull_beta, relative_t4):
    h_t = (weibull_beta/weibull_lambda) * (relative_t/weibull_lambda)**(weibull_beta - 1)
    ht4_value = (weibull_beta/weibull_lambda) * (relative_t4/weibull_lambda)**(weibull_beta - 1)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, h_t))

def H_weibull(relative_t, weibull_lambda, weibull_beta, relative_t4):
    H_t = (relative_t / weibull_lambda)**weibull_beta
    Ht4_value = (relative_t4 / weibull_lambda)**weibull_beta
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, H_t))

def F_weibull(relative_t, weibull_lambda, weibull_beta, relative_t4):
    F_t = 1 - np.exp(-(relative_t/weibull_lambda)**weibull_beta)
    Ft4_value = 1 - np.exp(-(relative_t4/weibull_lambda)**weibull_beta)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))

def objective_weibull(params, relative_t, F_observed, h_observed, relative_t4, theta):
    weibull_lambda, weibull_beta = params
    F_estimated = F_weibull(relative_t, weibull_lambda, weibull_beta, relative_t4)
    h_estimated = h_weibull(relative_t, weibull_lambda, weibull_beta, relative_t4)
    return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)

############# log-logistic
def h_log(relative_t, log_a, log_b, relative_t4):
    h_t = (log_b / log_a) * (relative_t / log_a)**(log_b - 1) / (1 + (relative_t / log_a)**log_b)
    ht4_value = (log_b / log_a) * (relative_t4 / log_a)**(log_b - 1) / (1 + (relative_t4 / log_a)**log_b)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, h_t))

def H_log(relative_t, log_a, log_b, relative_t4):
    H_t = np.log(1 + (relative_t / log_a)**log_b)
    Ht4_value = np.log(1 + (relative_t4 / log_a)**log_b)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, H_t))

def F_log(relative_t, log_a, log_b, relative_t4):
    F_t = (relative_t**log_b) / (log_a**log_b + relative_t**log_b)
    Ft4_value = (relative_t4**log_b) / (log_a**log_b + relative_t4**log_b)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))

def objective_log(params, relative_t, F_observed, h_observed, relative_t4, theta):
    log_a, log_b = params
    F_estimated = F_log(relative_t, log_a, log_b, relative_t4)
    h_estimated = h_log(relative_t, log_a, log_b, relative_t4)
    return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)

############# Gompertz
# def H_gomp(relative_t, gomp_lambda, gomp_beta, relative_t4):
#     H_t = (gomp_lambda / gomp_beta) * (np.exp(gomp_beta * relative_t) - 1)
#     Ht4_value = (gomp_lambda / gomp_beta) * (np.exp(gomp_beta * relative_t4) - 1)
#     return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, Ht4_value))
def h_gomp(relative_t, gomp_lambda, gomp_beta, relative_t4):
    scaled_relative_t = relative_t / np.max([1, np.abs(relative_t4)])
    scaled_relative_t4 = relative_t4 / np.max([1, np.abs(relative_t4)])
    h_t = gomp_lambda * np.exp(gomp_beta * scaled_relative_t)
    ht4_value = gomp_lambda * np.exp(gomp_beta * scaled_relative_t4)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, h_t))

def H_gomp(relative_t, gomp_lambda, gomp_beta, relative_t4):
    # To avoid overflow, scale relative_t if necessary
    scaled_relative_t = relative_t / np.max([1, np.abs(relative_t4)])
    scaled_relative_t4 = relative_t4 / np.max([1, np.abs(relative_t4)])
    H_t = (gomp_lambda / gomp_beta) * (np.exp(gomp_beta * scaled_relative_t) - 1)
    # H_t = (gomp_lambda / gomp_beta) * (np.exp(gomp_beta * relative_t) - 1)
    Ht4_value = (gomp_lambda / gomp_beta) * (np.exp(gomp_beta * scaled_relative_t4) - 1)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, H_t))

def F_gomp(relative_t, gomp_lambda, gomp_beta, relative_t4):
    scaled_relative_t = relative_t / np.max([1, np.abs(relative_t4)])
    scaled_relative_t4 = relative_t4 / np.max([1, np.abs(relative_t4)])
    F_t = 1 - np.exp(- (gomp_lambda / gomp_beta) * (np.exp(gomp_beta * scaled_relative_t) - 1))
    Ft4_value = 1 - np.exp(- (gomp_lambda / gomp_beta) * (np.exp(gomp_beta * scaled_relative_t4) - 1))
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))

def objective_gomp(params, relative_t, F_observed, h_observed, relative_t4, theta):
    gomp_lambda, gomp_beta = params
    F_estimated = F_gomp(relative_t, gomp_lambda, gomp_beta, relative_t4)
    h_estimated = h_gomp(relative_t, gomp_lambda, gomp_beta, relative_t4)
    return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)

############# LSM
# Define the function h(t)
def h_LSM(relative_t, lsm_a, lsm_b, lsm_c, relative_t4):
    h_t = lsm_a * (1 - (relative_t / lsm_b)**lsm_c)
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), h_t, h_t))

def H_LSM(relative_t, lsm_a, lsm_b, lsm_c, relative_t4):
    H_t = lsm_a * (relative_t - (relative_t**(lsm_c+1)) / (lsm_b**lsm_c * (lsm_c + 1)))
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), H_t, H_t))

def F_LSM(relative_t, lsm_a, lsm_b, lsm_c, relative_t4):
    F_t = 1 - np.exp(-lsm_a * (relative_t - (relative_t**(lsm_c+1)) / (lsm_b**lsm_c * (lsm_c + 1))))
    return np.where(relative_t < 0, 0, np.where((relative_t >= 0) & (relative_t <= relative_t4), F_t, F_t))

def objective_LSM(params, relative_t, F_observed, h_observed, relative_t4, theta):
    lsm_a, lsm_b, lsm_c = params
    F_estimated = F_LSM(relative_t, lsm_a, lsm_b, lsm_c, relative_t4)
    h_estimated = h_LSM(relative_t, lsm_a, lsm_b, lsm_c, relative_t4)
    return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)


# Function to calculate R²
def calculate_r2(observed, predicted):
    SSR = np.sum((observed - predicted) ** 2)
    SST = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (SSR / SST)

# Function to calculate Root Mean Square Error (RMSE)
# def calculate_rmse(observed, predicted):
#     return np.sqrt(np.mean((observed - predicted) ** 2))
def calculate_rmse(observed, predicted):
    return np.sum((observed - predicted)**2)

# Function to theta-weighed RMSE
def calculate_theta_weighted_rmse(F_observed, F_estimated, h_observed, h_estimated, theta):
    return theta * np.sum((F_observed - F_estimated) ** 2) + (1 - theta) * np.sum((h_observed - h_estimated) ** 2)


# Define a formatter function
def two_decimal_formatter(x, pos):
    return f'{x:.2f}'

def three_decimal_formatter(x, pos):
    return f'{x:.3f}'

def normalize_ridership(file_path, output_file_path):
    # Load data from CSV file
    df = pd.read_csv(file_path)

    # Convert 't', 't2', and 't4' to numeric types in case they are read as strings
    df[['t', 't2', 't4']] = df[['t', 't2', 't4']].apply(pd.to_numeric, errors='coerce')
    # workplaces_percent_change_from_baseline, transit_stations_percent_change_from_baseline, retail_and_recreation_percent_change_from_baseline,
    # residential_percent_change_from_baseline, grocery_and_pharmacy_percent_change_from_baseline, parks_percent_change_from_baseline
    # df = df[(df['state'].isin(['US'])) & (df['mobility_type'].isin(['transit_stations_percent_change_from_baseline', 'parks_percent_change_from_baseline']))]
    # df = df[(df['state'].isin(['US']))]
    # df = df[(df['mobility_type'].isin(['transit_stations_percent_change_from_baseline']))]

    # Group by 'state' and 'mobility_type'
    grouped = df.groupby(['state', 'mobility_type'])

    # Initialize a DataFrame to store results
    result_df = pd.DataFrame()

    for (state, mobility_type), group in grouped:
        # For each group, find max and min ridership between t2 and t4
        max_ridership = group.loc[(group['t'] >= group['t2'].iloc[0]) & (group['t'] <= group['t4'].iloc[0]), 'ridership'].max()
        min_ridership = group.loc[(group['t'] >= group['t2'].iloc[0]) & (group['t'] <= group['t4'].iloc[0]), 'ridership'].min()
        group['Qmax'] = min_ridership
        group['delta'] = group.loc[group['t'] == group['t4'].iloc[0], 'ridership'].values[0]  ### need update the delta value. should be from lea

        # # Find the time at which min_ridership occurs
        time_at_min_ridership = group.loc[group['ridership'] == min_ridership, 't'].min()

        group['UN_F(t)'] = (group['ridership'] + abs(min_ridership)) / abs(min_ridership)
        print(f"The maximum value of un-normalized F(t): {group['UN_F(t)'].max()}")

        # # Normalize ridership based on max_ridership value
        # if max_ridership < 0:
        #     w = 1
        # else:
        #     w = 1 / (int(group['UN_F(t)'].max()) + 1)

        w = 1 / (int(group['UN_F(t)'].max()) + 1)

        group['F(t)'] = w * (group['ridership'] + abs(min_ridership)) / abs(min_ridership)
        print(f"The scaling factor w: {w}")

        # Calculate S(t) based on F(t)
        group['S(t)'] = 1 - group['F(t)']

        # Calculate H(t) based on S(t)
        epsilon = 1e-10
        group['H(t)'] = -np.log(group['S(t)'] + epsilon)

        # Approximate h(t) using the difference in H(t)
        group['h(t)'] = group['H(t)'].diff().fillna(0) / group['t'].diff().fillna(1)

        # # Step 1: Smooth H(t) using Gaussian filter
        # group['smoothed_H(t)'] = gaussian_filter1d(group['H(t)'], sigma=6)  # You can adjust the sigma for more or less smoothing
        #
        # # Step 2: Compute the derivative of smoothed H(t) to get h(t)
        # group['smoothed_h(t)'] = np.gradient(group['smoothed_H(t)'], group['t'])  # Using np.gradient to calculate the derivative



        # Fit the polynomial function to F(t)
        # Filter data to only include points between t2 and t4
        t2 = time_at_min_ridership
        t4 = group['t4'].iloc[0]
        relative_t4 = t4 - t2
        relative_t5 = relative_t4 + 20
        cut_off_time = int(relative_t5)


        group['R'] = relative_t4
        # Update 't2' in the group
        group['t2'] = time_at_min_ridership
        group['relative_t'] = group['t'] - t2

        # Extract the F(t) value at relative_t4
        delta_prime = 1 - group.loc[group['relative_t'] == relative_t5, 'F(t)'].values[0]
        print("Delta prime is: ", delta_prime)
        group['delta_prime'] = delta_prime



        smoothing_group = group[(group['relative_t'] >= 0) & (group['relative_t'] <= relative_t5)]
        # filtered_group_pred = group[(group['relative_t'] > cut_off_time) & (group['relative_t'] <= relative_t5)]
        # Initial guess for parameters a, t4, t5 (you can adjust these if necessary)
        initial_guess = [1.0, 1000]
        # Fit the curve using curve_fit, H_smoothing (relative_t, smoothing_a, relative_t4, relative_t6):
        try:
            params, _ = curve_fit(lambda relative_t, smoothing_a, relative_t6: H_smoothing(relative_t, smoothing_a, relative_t4, relative_t6),
                                  smoothing_group['relative_t'], smoothing_group['H(t)'], p0=initial_guess,
                                  bounds= ([-np.inf, -np.inf], [np.inf, np.inf]), maxfev=2000)
        except RuntimeError as e:
            print(f"Failed to fit for group {(state, mobility_type)} with error: {e}")
            continue
        smoothing_a, relative_t6 = params
        print(f"Optimized parameters: smoothing_a = {smoothing_a}, relative_t6 = {relative_t6}")
        # Generate fitted values using the optimized parameters
        group['smoothed_H(t)'] = H_smoothing(group['relative_t'], smoothing_a, relative_t4,
                                                      relative_t6)
        group['smoothed_h(t)'] = np.gradient(group['smoothed_H(t)'], group[
            'relative_t'])  # Using np.gradient to calculate the derivative

        filtered_group = group[(group['relative_t'] >= 0) & (group['relative_t'] <= cut_off_time)]
        filtered_group_pred = group[(group['relative_t'] > cut_off_time) & (group['relative_t'] <= relative_t5)]

        # smoothed_h(t) = group['smoothed_h(t)'] when 0 < t <= relative_t4
        filtered_group.loc[(filtered_group['relative_t'] >= 0) & (filtered_group['relative_t'] <= relative_t4),
        'smoothed_h(t)'] = group['smoothed_h(t)']

        # smoothed_h(t) = 0 when relative_t4 < t <= relative_t5
        filtered_group.loc[(filtered_group['relative_t'] > relative_t4) & (filtered_group['relative_t'] <= relative_t5),
        'smoothed_h(t)'] = 0


        theta = filtered_group['smoothed_h(t)'].max() / (filtered_group['F(t)'].max() + filtered_group['smoothed_h(t)'].max())
        print("theta is: ", theta)

        # ###### Lear-based H(t) fitting
        # initial_linear = [10]
        #
        # try:
        #     params, _ = curve_fit(lambda relative_t, linear_a: H_L(relative_t, linear_a, relative_t4), filtered_group['relative_t'], filtered_group['H(t)'], p0=initial_linear, bounds=(-np.inf, np.inf), maxfev=2000)
        # except RuntimeError as e:
        #     print(f"Failed to fit for group {(state, mobility_type)} with error: {e}")
        #     continue
        #
        # linear_a = params[0]
        observed_Ht4_value = filtered_group.loc[filtered_group['relative_t'] == relative_t4, 'F(t)'].values[0]
        linear_a = observed_Ht4_value/relative_t4
        print(f"linear_a: {linear_a}")
        group['linear_a'] = linear_a

        ####### DQQ-based H(t) fitting
        initial_dqq = [1]

        res_dqq = minimize(objective_DQQ, initial_dqq, args=(
            filtered_group['relative_t'], filtered_group['F(t)'], filtered_group['smoothed_h(t)'], relative_t4, theta),
                           method='L-BFGS-B')

        _beta = res_dqq.x[0]
        print(f"DQQ beta: {_beta}")
        group['dqq_beta'] = _beta

        ####### LEA-based H(t) fitting
        h_b = filtered_group['smoothed_h(t)'].max()
        # Assign 'b' as the time corresponding to the maximum value of 'smoothed_h(t)'
        b = filtered_group['relative_t'][filtered_group['smoothed_h(t)'].idxmax()]
        # initial_lea = [10, 10]
        delta_prime_2 = 1 - group.loc[group['relative_t'] == relative_t4, 'F(t)'].values[0]
        # a_min = 4 * np.log(1-delta_prime_2) / ((3 * np.exp(-2) - 1) * relative_t4 * relative_t4)
        # a_mid = 9 * np.log(1-delta_prime_2) / ((4 * np.exp(-3) - 1) * relative_t4 * relative_t4)
        # a_max = 400 * np.log(1-delta_prime_2) / ((21 * np.exp(-20) - 1) * relative_t4 * relative_t4)
        # b_min = (relative_t4 / 20).item()
        # b_max = (relative_t4 / 2).item()
        # print(f"initial lea: {initial_lea}; {a_min}, {a_max}, {b_min}, {b_max}")
        b_min = b
        b_max = b

        # initial_lea = [max(a_min, min(a_mid, a_max)), b]
        initial_lea = [b]
        res_lea = minimize(objective_LEA, initial_lea, args=(
            filtered_group['relative_t'], filtered_group['F(t)'], filtered_group['smoothed_h(t)'], relative_t4, theta, h_b),
                           method='L-BFGS-B',bounds=[(b_min, b_max)])

        b = res_lea.x[0]
        print(f"LEA b: {b}")
        a = (h_b / b) / np.exp(-1 / 2)
        group['lea_a'] = a
        group['lea_b'] = b
        group['lea_c'] = 1


        ####### exp-based H(t) fitting
        initial_exp = [0.01]

        res_exp = minimize(objective_exp, initial_exp, args=(
            filtered_group['relative_t'], filtered_group['F(t)'], filtered_group['smoothed_h(t)'], relative_t4, theta),
                           method='L-BFGS-B', bounds=[(1e-5, np.inf)])

        exp_lambda = res_exp.x[0]
        print(f"Exp lambda: {exp_lambda}")
        group['exp_lambda'] = exp_lambda

        ####### weibull-based H(t) fitting
        initial_weibull = [relative_t4, 1.5]

        res_weibull = minimize(objective_weibull, initial_weibull, args=(
            filtered_group['relative_t'], filtered_group['F(t)'], filtered_group['smoothed_h(t)'], relative_t4, theta),
                           method='L-BFGS-B', bounds=[(1e-5, np.inf), (1e-5, np.inf)])

        weibull_lambda, weibull_beta = res_weibull.x
        print(f"Weibull lambda: {weibull_lambda}, beta: {weibull_beta}")
        group['weibull_lambda'] = weibull_lambda
        group['weibull_beta'] = weibull_beta

        ####### log logistic-based H(t) fitting
        initial_log = [relative_t4, 1.5]
        res_log = minimize(objective_log, initial_log, args=(
            filtered_group['relative_t'], filtered_group['F(t)'], filtered_group['smoothed_h(t)'], relative_t4, theta),
                           method='L-BFGS-B', bounds=[(1e-5, np.inf), (1e-5, np.inf)])

        log_a, log_b = res_log.x
        print(f"Log-logistic a: {log_a}, b: {log_b}")
        group['log_a'] = log_a
        group['log_b'] = log_b

        ####### gomptz-based H(t) fitting
        initial_gomp = [1, 1.5]
        res_gomp = minimize(objective_gomp, initial_gomp, args=(
            filtered_group['relative_t'], filtered_group['F(t)'], filtered_group['smoothed_h(t)'], relative_t4, theta),
                           method='L-BFGS-B', bounds=[(1e-5, np.inf), (1e-5, np.inf)])

        gomp_lambda, gomp_beta = res_gomp.x
        print(f"Gomptz lambda: {gomp_lambda}, beta: {gomp_beta}")
        group['gomp_lambda'] = gomp_lambda
        group['gomp_beta'] = gomp_beta


        ####### LSM-based H(t) fitting
        initial_lsm = [0.013, 80, 3.5]
        res_lsm = minimize(objective_LSM, initial_lsm, args=(
            filtered_group['relative_t'], filtered_group['F(t)'], filtered_group['smoothed_h(t)'], relative_t4, theta),
                           method='L-BFGS-B', bounds=[(1e-5, np.inf), (1, np.inf), (1e-5, np.inf)])

        lsm_a, lsm_b, lsm_c = res_lsm.x
        print(f"LSM a: {lsm_a}, b: {lsm_b}, c: {lsm_c}")
        group['lsm_a'] = lsm_a
        group['lsm_b'] = lsm_b
        group['lsm_c'] = lsm_c


        ####### linear-based H(t) caculation
        group['fitted_H(t)_linear'] = H_L(group['relative_t'], linear_a, relative_t4)
        group['R2_H(t)_linear_est'] = calculate_r2(filtered_group['H(t)'], group['fitted_H(t)_linear'])
        group['RMSE_H(t)_linear_est'] = calculate_rmse(filtered_group['H(t)'], group['fitted_H(t)_linear'])
        group['RMSE_H(t)_linear_pred'] = calculate_rmse(filtered_group_pred['H(t)'], group['fitted_H(t)_linear'])
        group['fitted_F(t)_linear'] = F_L(group['relative_t'], linear_a, relative_t4)
        group['RMSE_F(t)_linear_est'] = calculate_rmse(filtered_group['F(t)'], group['fitted_F(t)_linear'])
        group['RMSE_F(t)_linear_pred'] = calculate_rmse(filtered_group_pred['F(t)'], group['fitted_F(t)_linear'])
        group['fitted_h(t)_linear'] = h_L(group['relative_t'], linear_a, relative_t4)
        group['RMSE_h(t)_linear_est'] = calculate_rmse(filtered_group['smoothed_h(t)'], group['fitted_h(t)_linear'])
        group['RMSE_h(t)_linear_pred'] = calculate_rmse(filtered_group_pred['smoothed_h(t)'], group['fitted_h(t)_linear'])
        group['delta_prime_linear'] = 1 - F_L(relative_t5, linear_a, relative_t4)
        group['theta_weighted_RMSE_linear'] = calculate_theta_weighted_rmse(filtered_group['F(t)'], group['fitted_F(t)_linear'], filtered_group['smoothed_h(t)'], group['fitted_h(t)_linear'], theta)
        ####### DQQ-based H(t) caculation
        group['fitted_H(t)_dqq'] = H_DQQ(group['relative_t'], _beta, relative_t4)
        group['R2_H(t)_dqq_est'] = calculate_r2(filtered_group['H(t)'], group['fitted_H(t)_dqq'])
        group['RMSE_H(t)_dqq_est'] = calculate_rmse(filtered_group['H(t)'], group['fitted_H(t)_dqq'])
        group['RMSE_H(t)_dqq_pred'] = calculate_rmse(filtered_group_pred['H(t)'], group['fitted_H(t)_dqq'])
        group['fitted_F(t)_dqq'] = F_DQQ(group['relative_t'], _beta, relative_t4)
        group['RMSE_F(t)_dqq_est'] = calculate_rmse(filtered_group['F(t)'], group['fitted_F(t)_dqq'])
        group['RMSE_F(t)_dqq_pred'] = calculate_rmse(filtered_group_pred['F(t)'], group['fitted_F(t)_dqq'])
        group['fitted_h(t)_dqq'] = h_DQQ(group['relative_t'], _beta, relative_t4)
        group['RMSE_h(t)_dqq_est'] = calculate_rmse(filtered_group['smoothed_h(t)'], group['fitted_h(t)_dqq'])
        group['RMSE_h(t)_dqq_pred'] = calculate_rmse(filtered_group_pred['smoothed_h(t)'], group['fitted_h(t)_dqq'])
        group['delta_prime_dqq'] = 1 - F_DQQ(relative_t5, _beta, relative_t4)
        group['theta_weighted_RMSE_dqq'] = calculate_theta_weighted_rmse(filtered_group['F(t)'], group['fitted_F(t)_dqq'], filtered_group['smoothed_h(t)'], group['fitted_h(t)_dqq'], theta)
        ####### LEA-based H(t) caculation
        group['fitted_H(t)_lea'] = H_LEA(group['relative_t'], a, b, relative_t4, h_b)
        group['R2_H(t)_lea_est'] = calculate_r2(filtered_group['H(t)'], group['fitted_H(t)_lea'])
        group['RMSE_H(t)_lea_est'] = calculate_rmse(filtered_group['H(t)'], group['fitted_H(t)_lea'])
        group['RMSE_H(t)_lea_pred'] = calculate_rmse(filtered_group_pred['H(t)'], group['fitted_H(t)_lea'])
        group['fitted_F(t)_lea'] = F_LEA(group['relative_t'], a, b, relative_t4, h_b)
        group['RMSE_F(t)_lea_est'] = calculate_rmse(filtered_group['F(t)'], group['fitted_F(t)_lea'])
        group['RMSE_F(t)_lea_pred'] = calculate_rmse(filtered_group_pred['F(t)'], group['fitted_F(t)_lea'])
        group['fitted_h(t)_lea'] = h_LEA(group['relative_t'], a, b, relative_t4, h_b)
        group['RMSE_h(t)_lea_est'] = calculate_rmse(filtered_group['smoothed_h(t)'], group['fitted_h(t)_lea'])
        group['RMSE_h(t)_lea_pred'] = calculate_rmse(filtered_group_pred['smoothed_h(t)'], group['fitted_h(t)_lea'])
        group['delta_prime_lea'] = 1 - F_LEA(relative_t5, a, b, relative_t4, h_b)
        group['theta_weighted_RMSE_lea'] = calculate_theta_weighted_rmse(filtered_group['F(t)'], group['fitted_F(t)_lea'], filtered_group['smoothed_h(t)'], group['fitted_h(t)_lea'], theta)
        # Define the x-axis (relative_t) and the y-axis values (fitted_F(t)_lea) within the specified range
        x_values = group['relative_t'][
            (group['relative_t'] >= 0) & (group['relative_t'] <= relative_t4)]
        y_values = group['fitted_F(t)_lea'][
            (group['relative_t'] >= 0) & (group['relative_t'] <= relative_t4)]

        # Calculate the area using the trapezoidal rule
        area1 = relative_t4 * group.loc[group['relative_t'] == relative_t4, 'fitted_F(t)_lea'].values[0]
        area = area1 - np.trapz(y_values, x_values)
        group['RTA_lea'] = area

        ####### exp-based H(t) caculation
        group['fitted_H(t)_exp'] = H_exp(group['relative_t'], exp_lambda, relative_t4)
        group['R2_H(t)_exp_est'] = calculate_r2(filtered_group['H(t)'], group['fitted_H(t)_exp'])
        group['RMSE_H(t)_exp_est'] = calculate_rmse(filtered_group['H(t)'], group['fitted_H(t)_exp'])
        group['RMSE_H(t)_exp_pred'] = calculate_rmse(filtered_group_pred['H(t)'], group['fitted_H(t)_exp'])
        group['fitted_F(t)_exp'] = F_exp(group['relative_t'], exp_lambda, relative_t4)
        group['RMSE_F(t)_exp_est'] = calculate_rmse(filtered_group['F(t)'], group['fitted_F(t)_exp'])
        group['RMSE_F(t)_exp_pred'] = calculate_rmse(filtered_group_pred['F(t)'], group['fitted_F(t)_exp'])
        group['fitted_h(t)_exp'] = h_exp(group['relative_t'], exp_lambda, relative_t4)
        group['RMSE_h(t)_exp_est'] = calculate_rmse(filtered_group['smoothed_h(t)'], group['fitted_h(t)_exp'])
        group['RMSE_h(t)_exp_pred'] = calculate_rmse(filtered_group_pred['smoothed_h(t)'], group['fitted_h(t)_exp'])
        group['delta_prime_exp'] = 1 - F_exp(relative_t5, exp_lambda, relative_t4)
        group['theta_weighted_RMSE_exp'] = calculate_theta_weighted_rmse(filtered_group['F(t)'], group['fitted_F(t)_exp'], filtered_group['smoothed_h(t)'], group['fitted_h(t)_exp'], theta)
        ####### weibull-based H(t) caculation
        group['fitted_H(t)_weibull'] = H_weibull(group['relative_t'], weibull_lambda, weibull_beta, relative_t4)
        group['R2_H(t)_weibull_est'] = calculate_r2(filtered_group['H(t)'], group['fitted_H(t)_weibull'])
        group['RMSE_H(t)_weibull_est'] = calculate_rmse(filtered_group['H(t)'], group['fitted_H(t)_weibull'])
        group['RMSE_H(t)_weibull_pred'] = calculate_rmse(filtered_group_pred['H(t)'], group['fitted_H(t)_weibull'])
        group['fitted_F(t)_weibull'] = F_weibull(group['relative_t'], weibull_lambda, weibull_beta, relative_t4)
        group['RMSE_F(t)_weibull_est'] = calculate_rmse(filtered_group['F(t)'], group['fitted_F(t)_weibull'])
        group['RMSE_F(t)_weibull_pred'] = calculate_rmse(filtered_group_pred['F(t)'], group['fitted_F(t)_weibull'])
        group['fitted_h(t)_weibull'] = h_weibull(group['relative_t'], weibull_lambda, weibull_beta, relative_t4)
        group['RMSE_h(t)_weibull_est'] = calculate_rmse(filtered_group['smoothed_h(t)'], group['fitted_h(t)_weibull'])
        group['RMSE_h(t)_weibull_pred'] = calculate_rmse(filtered_group_pred['smoothed_h(t)'], group['fitted_h(t)_weibull'])
        group['delta_prime_weibull'] = 1 - F_weibull(relative_t5, weibull_lambda, weibull_beta, relative_t4)
        group['RMSE_delta_prime_weibull'] = calculate_rmse(group['delta_prime'], group['delta_prime_weibull'])
        group['theta_weighted_RMSE_weibull'] = calculate_theta_weighted_rmse(filtered_group['F(t)'], group['fitted_F(t)_weibull'], filtered_group['smoothed_h(t)'], group['fitted_h(t)_weibull'], theta)
        ####### log-based H(t) caculation
        group['fitted_H(t)_log'] = H_log(group['relative_t'], log_a, log_b, relative_t4)
        group['R2_H(t)_log_est'] = calculate_r2(filtered_group['H(t)'], group['fitted_H(t)_log'])
        group['RMSE_H(t)_log_est'] = calculate_rmse(filtered_group['H(t)'], group['fitted_H(t)_log'])
        group['RMSE_H(t)_log_pred'] = calculate_rmse(filtered_group_pred['H(t)'], group['fitted_H(t)_log'])
        group['fitted_F(t)_log'] = F_log(group['relative_t'], log_a, log_b, relative_t4)
        group['RMSE_F(t)_log_est'] = calculate_rmse(filtered_group['F(t)'], group['fitted_F(t)_log'])
        group['RMSE_F(t)_log_pred'] = calculate_rmse(filtered_group_pred['F(t)'], group['fitted_F(t)_log'])
        group['fitted_h(t)_log'] = h_log(group['relative_t'], log_a, log_b, relative_t4)
        group['RMSE_h(t)_log_est'] = calculate_rmse(filtered_group['smoothed_h(t)'], group['fitted_h(t)_log'])
        group['RMSE_h(t)_log_pred'] = calculate_rmse(filtered_group_pred['smoothed_h(t)'], group['fitted_h(t)_log'])
        group['delta_prime_log'] = 1 - F_log(relative_t5, log_a, log_b, relative_t4)
        group['theta_weighted_RMSE_log'] = calculate_theta_weighted_rmse(filtered_group['F(t)'], group['fitted_F(t)_log'], filtered_group['smoothed_h(t)'], group['fitted_h(t)_log'], theta)
        ####### gomp-based H(t) caculation
        group['fitted_H(t)_gomp'] = H_gomp(group['relative_t'], gomp_lambda, gomp_beta, relative_t4)
        group['R2_H(t)_gomp_est'] = calculate_r2(filtered_group['H(t)'], group['fitted_H(t)_gomp'])
        group['RMSE_H(t)_gomp_est'] = calculate_rmse(filtered_group['H(t)'], group['fitted_H(t)_gomp'])
        group['RMSE_H(t)_gomp_pred'] = calculate_rmse(filtered_group_pred['H(t)'], group['fitted_H(t)_gomp'])
        group['fitted_F(t)_gomp'] = F_gomp(group['relative_t'], gomp_lambda, gomp_beta, relative_t4)
        group['RMSE_F(t)_gomp_est'] = calculate_rmse(filtered_group['F(t)'], group['fitted_F(t)_gomp'])
        group['RMSE_F(t)_gomp_pred'] = calculate_rmse(filtered_group_pred['F(t)'], group['fitted_F(t)_gomp'])
        group['fitted_h(t)_gomp'] = h_gomp(group['relative_t'], gomp_lambda, gomp_beta, relative_t4)
        group['RMSE_h(t)_gomp_est'] = calculate_rmse(filtered_group['smoothed_h(t)'], group['fitted_h(t)_gomp'])
        group['RMSE_h(t)_gomp_pred'] = calculate_rmse(filtered_group_pred['smoothed_h(t)'], group['fitted_h(t)_gomp'])
        group['delta_prime_gomp'] = 1 - F_gomp(relative_t5, gomp_lambda, gomp_beta, relative_t4)
        group['theta_weighted_RMSE_gomp'] = calculate_theta_weighted_rmse(filtered_group['F(t)'], group['fitted_F(t)_gomp'], filtered_group['smoothed_h(t)'], group['fitted_h(t)_gomp'], theta)
        ####### LSM-based H(t) caculation
        group['fitted_H(t)_lsm'] = H_LSM(group['relative_t'], lsm_a, lsm_b, lsm_c, relative_t4)
        group['R2_H(t)_lsm_est'] = calculate_r2(filtered_group['H(t)'], group['fitted_H(t)_lsm'])
        group['RMSE_H(t)_lsm_est'] = calculate_rmse(filtered_group['H(t)'], group['fitted_H(t)_lsm'])
        group['RMSE_H(t)_lsm_pred'] = calculate_rmse(filtered_group_pred['H(t)'], group['fitted_H(t)_lsm'])
        group['fitted_F(t)_lsm'] = F_LSM(group['relative_t'], lsm_a, lsm_b, lsm_c, relative_t4)
        group['RMSE_F(t)_lsm_est'] = calculate_rmse(filtered_group['F(t)'], group['fitted_F(t)_lsm'])
        group['RMSE_F(t)_lsm_pred'] = calculate_rmse(filtered_group_pred['F(t)'], group['fitted_F(t)_lsm'])
        group['fitted_h(t)_lsm'] = h_LSM(group['relative_t'], lsm_a, lsm_b, lsm_c, relative_t4)
        group['RMSE_h(t)_lsm_est'] = calculate_rmse(filtered_group['smoothed_h(t)'], group['fitted_h(t)_lsm'])
        group['RMSE_h(t)_lsm_pred'] = calculate_rmse(filtered_group_pred['smoothed_h(t)'], group['fitted_h(t)_lsm'])
        group['delta_prime_lsm'] = 1 - F_LSM(relative_t5, lsm_a, lsm_b, lsm_c, relative_t4)
        group['theta_weighted_RMSE_lsm'] = calculate_theta_weighted_rmse(filtered_group['F(t)'], group['fitted_F(t)_lsm'], filtered_group['smoothed_h(t)'], group['fitted_h(t)_lsm'], theta)



        # Append to the result DataFrame
        result_df = pd.concat([result_df, group], ignore_index=True)

        filtered_group_plot = group[(group['relative_t'] >= 0) & (group['relative_t'] <= relative_t5)]
        # Ensure filtered_group_plot has only relevant data
        filtered_group_plot = filtered_group_plot[['relative_t']].merge(
            group[['relative_t',
                   'F(t)', 'fitted_F(t)_lea', 'fitted_F(t)_linear', 'fitted_F(t)_log', 'fitted_F(t)_weibull', 'fitted_F(t)_exp', 'fitted_F(t)_dqq', 'fitted_F(t)_gomp', 'fitted_F(t)_lsm',
                   'smoothed_H(t)', 'fitted_H(t)_lea', 'fitted_H(t)_linear', 'fitted_H(t)_log', 'fitted_H(t)_weibull', 'fitted_H(t)_exp', 'fitted_H(t)_dqq', 'fitted_H(t)_gomp', 'fitted_H(t)_lsm',
                   'smoothed_h(t)', 'fitted_h(t)_lea', 'fitted_h(t)_linear', 'fitted_h(t)_log', 'fitted_h(t)_weibull', 'fitted_h(t)_exp', 'fitted_h(t)_dqq', 'fitted_h(t)_gomp', 'fitted_h(t)_lsm']],
            on='relative_t', how='inner')

        # Create a single figure with 2 subplots
        # fig_1, axs = plt.subplots(2, 1, figsize=(16, 24), sharex=True)
        fig, axs = plt.subplots(3, 1, figsize=(24, 28))
        plt.subplots_adjust(hspace=0.3)  # Adjust vertical spacing between subplots
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)  # Adjust the margins

        # Create predictions for plotting
        axs[0].set_xlim(0, relative_t5)
        # axs[0].set_ylim(0, 1)
        axs[1].set_xlim(0, relative_t5)
        axs[2].set_xlim(0, relative_t5)
        # axs[2].set_ylim(filtered_group_plot['fitted_h(t)_lea'].min() * 1.5, filtered_group_plot['fitted_h(t)_lea'].max() * 1.5)
        # axs[2].set_ylim(filtered_group_plot['smoothed_h(t)'].min() * 1.5, filtered_group_plot['smoothed_h(t)'].max() * 1.5)
        axs[2].set_ylim(0, filtered_group_plot['smoothed_h(t)'].max() * 1.5)

        # Subplot 1: F
        axs[0].plot(filtered_group_plot['relative_t'], filtered_group_plot['F(t)'], 'o-', label='Observed Data', markersize=15, linewidth=15, alpha=0.3)
        axs[0].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_F(t)_lea'], 'x-', label='LEA', linewidth=8)
        # axs[0].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_F(t)_linear'], 'x-', label='V-Shape')
        axs[0].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_F(t)_log'], 'x-', label='Log Logistic')
        axs[0].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_F(t)_weibull'], 'x-', label='Weibull')
        axs[0].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_F(t)_exp'], 'x-', label='Exponential')
        # axs[0].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_F(t)_dqq'], 'x-', label='DQQ')
        # axs[0].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_F(t)_gomp'], 'x-', label='Gomptz')
        # axs[0].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_F(t)_lsm'], 'x-', label='LSM')
        # axs[0].set_xlabel('Time (Days)\n\n(a)', fontsize=30, labelpad=20)
        axs[0].set_xlabel('Time (Days)', fontsize=30, labelpad=20)
        # Apply the formatter to the y-axis
        axs[0].yaxis.set_major_formatter(FuncFormatter(two_decimal_formatter))
        axs[0].set_ylabel('F(t)', fontsize=30, labelpad=20)
        axs[0].tick_params(axis='x', labelsize=27.5)  # Set font size for x-axis labels
        axs[0].tick_params(axis='y', labelsize=27.5)  # Set font size for y-axis labels
        axs[0].set_title(f'Observed vs. Modeled for {state}, {mobility_type}', fontsize=30)
        axs[0].legend(fontsize=27.5)
        axs[0].grid(True)

        # Subplot 2: H
        axs[1].plot(filtered_group_plot['relative_t'][:relative_t4], filtered_group_plot['smoothed_H(t)'][:relative_t4], 'o-', label='Observed Data', markersize=15, linewidth=15, alpha=0.3)
        axs[1].plot(filtered_group_plot['relative_t'][:relative_t4], filtered_group_plot['fitted_H(t)_lea'][:relative_t4], 'x-', label='LEA', linewidth=8)
        # axs[1].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_H(t)_linear'], 'x-', label='V-Shape')
        axs[1].plot(filtered_group_plot['relative_t'][:relative_t4], filtered_group_plot['fitted_H(t)_log'][:relative_t4], 'x-', label='Log Logistic')
        axs[1].plot(filtered_group_plot['relative_t'][:relative_t4], filtered_group_plot['fitted_H(t)_weibull'][:relative_t4], 'x-', label='Weibull')
        axs[1].plot(filtered_group_plot['relative_t'][:relative_t4], filtered_group_plot['fitted_H(t)_exp'][:relative_t4], 'x-', label='Exponential')
        # axs[1].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_H(t)_dqq'], 'x-', label='DQQ')
        # axs[1].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_H(t)_gomp'], 'x-', label='Gomptz')
        # axs[1].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_H(t)_lsm'], 'x-', label='LSM')
        # axs[1].set_xlabel('Time (Days)\n\n(b)\n\n(2) Residential in Washington', fontsize=30, labelpad=20)
        axs[1].set_xlabel('Time (Days)', fontsize=30, labelpad=20)
        axs[1].set_ylabel('H(t)', fontsize=30, labelpad=20)
        axs[1].tick_params(axis='x', labelsize=27.5)  # Set font size for x-axis labels
        # Apply the formatter to the y-axis
        axs[1].yaxis.set_major_formatter(FuncFormatter(three_decimal_formatter))
        axs[1].tick_params(axis='y', labelsize=27.5)  # Set font size for y-axis labels
        # axs[1].set_title(f'Observed vs. Modeled for {state}, {mobility_type}', fontsize=30)
        # axs[1].legend(fontsize=27.5)
        axs[1].grid(True)

        # Subplot 3: h
        axs[2].plot(filtered_group_plot['relative_t'][:relative_t4], filtered_group_plot['smoothed_h(t)'][:relative_t4], 'o-', label='Observed Data', markersize=15, linewidth=15, alpha=0.3)
        axs[2].plot(filtered_group_plot['relative_t'][:relative_t4], filtered_group_plot['fitted_h(t)_lea'][:relative_t4], 'x-', label='LEA', linewidth=8)
        # axs[2].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_h(t)_linear'], 'x-', label='V-Shape')
        axs[2].plot(filtered_group_plot['relative_t'][:relative_t4], filtered_group_plot['fitted_h(t)_log'][:relative_t4], 'x-', label='Log Logistic')
        axs[2].plot(filtered_group_plot['relative_t'][:relative_t4], filtered_group_plot['fitted_h(t)_weibull'][:relative_t4], 'x-', label='Weibull')
        axs[2].plot(filtered_group_plot['relative_t'][:relative_t4], filtered_group_plot['fitted_h(t)_exp'][:relative_t4], 'x-', label='Exponential')
        # axs[2].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_h(t)_dqq'], 'x-', label='DQQ')
        # axs[2].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_h(t)_gomp'], 'x-', label='Gomptz')
        # axs[2].plot(filtered_group_plot['relative_t'], filtered_group_plot['fitted_h(t)_lsm'], 'x-', label='LSM')
        # axs[1].set_xlabel('Time (Days)\n\n(b)\n\n(2) Residential in Washington', fontsize=30, labelpad=20)
        axs[2].set_xlabel('Time (Days)', fontsize=30, labelpad=20)
        axs[2].set_ylabel('h(t)', fontsize=30, labelpad=20)
        axs[2].tick_params(axis='x', labelsize=27.5)  # Set font size for x-axis labels
        # Apply the formatter to the y-axis
        axs[2].yaxis.set_major_formatter(FuncFormatter(three_decimal_formatter))
        axs[2].tick_params(axis='y', labelsize=27.5)  # Set font size for y-axis labels
        # axs[2].set_title(f'Observed vs. Modeled for {state}, {mobility_type}', fontsize=30)
        # axs[2].legend(fontsize=27.5)
        axs[2].grid(True)

        # Save the plot
        filename = f"./fig/fitting_{state}_{mobility_type}.png"
        plt.savefig(filename, format='png')

        # plt.tight_layout()
        plt.show()




    # Filter result DataFrame for time range from time_at_min_ridership to t4+20 for each group
    filtered_result_df = pd.DataFrame()
    for _, group in result_df.groupby(['state', 'mobility_type']):
        # Filter result DataFrame for time range t2 to t4+20
        t2_minus_5 = time_at_min_ridership - 5
        t4_plus_20 = group['t4'].iloc[0] + 20
        group = group[(group['t'] >= t2_minus_5) & (group['t'] <= t4_plus_20)]
        filtered_result_df = pd.concat([filtered_result_df, group], ignore_index=True)

    # Select only the necessary columns to output
    filtered_result_df = filtered_result_df[['state', 'mobility_type', 'ridership', 't', 't0', 't2', 't4', 'R', 'Qmax', 'delta', 'delta_prime', 'RTA_lea', 'date', 'relative_t', 'UN_F(t)', 'F(t)', 'S(t)', 'H(t)', 'h(t)', 'smoothed_h(t)',
                                             'fitted_H(t)_linear', 'linear_a', 'R2_H(t)_linear_est', 'RMSE_H(t)_linear_est', 'RMSE_H(t)_linear_pred', 'fitted_F(t)_linear', 'RMSE_F(t)_linear_est', 'RMSE_F(t)_linear_pred', 'fitted_h(t)_linear', 'RMSE_h(t)_linear_est', 'RMSE_h(t)_linear_pred', 'delta_prime_linear', 'theta_weighted_RMSE_linear',
                                             'fitted_H(t)_dqq', 'dqq_beta', 'R2_H(t)_dqq_est', 'RMSE_H(t)_dqq_est', 'RMSE_H(t)_dqq_pred', 'fitted_F(t)_dqq', 'RMSE_F(t)_dqq_est', 'RMSE_F(t)_dqq_pred', 'fitted_h(t)_dqq', 'RMSE_h(t)_dqq_est', 'RMSE_h(t)_dqq_pred', 'delta_prime_dqq', 'theta_weighted_RMSE_dqq',
                                             'fitted_H(t)_lea', 'lea_a', 'lea_b', 'lea_c', 'R2_H(t)_lea_est', 'RMSE_H(t)_lea_est', 'RMSE_H(t)_lea_pred', 'fitted_F(t)_lea', 'RMSE_F(t)_lea_est', 'RMSE_F(t)_lea_pred', 'fitted_h(t)_lea', 'RMSE_h(t)_lea_est', 'RMSE_h(t)_lea_pred', 'delta_prime_lea', 'theta_weighted_RMSE_lea',
                                             'fitted_H(t)_exp', 'exp_lambda', 'R2_H(t)_exp_est', 'RMSE_H(t)_exp_est', 'RMSE_H(t)_exp_pred', 'fitted_F(t)_exp', 'RMSE_F(t)_exp_est', 'RMSE_F(t)_exp_pred', 'fitted_h(t)_exp', 'RMSE_h(t)_exp_est', 'RMSE_h(t)_exp_pred', 'delta_prime_exp', 'theta_weighted_RMSE_exp',
                                             'fitted_H(t)_weibull', 'weibull_lambda', 'weibull_beta', 'R2_H(t)_weibull_est', 'RMSE_H(t)_weibull_est', 'RMSE_H(t)_weibull_pred', 'fitted_F(t)_weibull', 'RMSE_F(t)_weibull_est', 'RMSE_F(t)_weibull_pred', 'RMSE_h(t)_weibull_est', 'RMSE_h(t)_weibull_pred', 'fitted_h(t)_weibull', 'delta_prime_weibull', 'theta_weighted_RMSE_weibull',
                                             'fitted_H(t)_log', 'log_a', 'log_b', 'R2_H(t)_log_est', 'RMSE_H(t)_log_est', 'RMSE_H(t)_log_pred', 'fitted_F(t)_log', 'RMSE_F(t)_log_est', 'RMSE_F(t)_log_pred', 'fitted_h(t)_log', 'RMSE_h(t)_log_est', 'RMSE_h(t)_log_pred', 'delta_prime_log', 'theta_weighted_RMSE_log',
                                             'fitted_H(t)_gomp', 'gomp_lambda', 'gomp_beta', 'R2_H(t)_gomp_est', 'RMSE_H(t)_gomp_est', 'RMSE_H(t)_gomp_pred', 'fitted_F(t)_gomp', 'RMSE_F(t)_gomp_est', 'RMSE_F(t)_gomp_pred', 'fitted_h(t)_gomp', 'RMSE_h(t)_gomp_est', 'RMSE_h(t)_gomp_pred', 'delta_prime_gomp', 'theta_weighted_RMSE_gomp',
                                             'fitted_H(t)_lsm', 'lsm_a', 'lsm_b', 'lsm_c', 'R2_H(t)_lsm_est', 'RMSE_H(t)_lsm_est', 'RMSE_H(t)_lsm_pred', 'fitted_F(t)_lsm', 'RMSE_F(t)_lsm_est', 'RMSE_F(t)_lsm_pred', 'fitted_h(t)_lsm', 'RMSE_h(t)_lsm_est', 'RMSE_h(t)_lsm_pred', 'delta_prime_lsm',  'theta_weighted_RMSE_lsm']]

    # Save the results to a new CSV file
    filtered_result_df.to_csv(output_file_path, index=False)

# Example usage
file_path = './data/mobility_t_values_states.csv'
output_file_path = './outputs/states_fitting.csv'
normalize_ridership(file_path, output_file_path)