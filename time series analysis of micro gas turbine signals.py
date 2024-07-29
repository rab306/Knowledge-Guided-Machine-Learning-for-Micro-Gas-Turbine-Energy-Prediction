# This is a project of Knowledge Guided Machine Learning for dynamical system (micro gas turbine) signals

# The idea is to integrate the knowledge of permissible system states with the machine learning model
# By adding constraints of the system to the loss function of the model
# These constraints are the constant change rates of the system in the stationary and transition states

# The data is a time series of the input voltage, output energy, and the time

# The project is divided into the following steps:
# 1. Read the data
# 2. Visualize the data
# 3. Calculate the change rate of the output energy
# 4. Slice the transition phases based on the change rate
# 5. Applying signal processing techniques (Fourier transform and Wavelet transform)
# 6. Preprocess the data
# 7. Define the custom loss function
# 8. Define the model
# 9. Train the model




# Importing the libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pywt


# Function to read data from a folder
def read_data_from_folder(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            data.append(df)
    return data


# Read train and test data
train_folder = 'train'
test_folder = 'test'

train_data = read_data_from_folder(train_folder)
test_data = read_data_from_folder(test_folder)

# Print data shapes and number of experiments
for i, df in enumerate(train_data):
    print(f'Training data shape is: {df.shape}')
print(f'We have {len(train_data)} experiments for training')

for i, df in enumerate(test_data):
    print(f'Test data shape is: {df.shape}')
print(f'We have {len(test_data)} experiments for testing')

# Print data info for each experiment
for i, df in enumerate(train_data):
    print(f'Experiment {i + 1} data info:')
    df.info()
    print("\n")


# Function to plot variable against the target in time domain
def plot_experiment(data, experiment_number, variable_name, label):
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 1, 1)
    plt.plot(data['time'], data[variable_name], label=label)
    plt.xlabel('Time')
    plt.ylabel(label)
    plt.title(f'Experiment {experiment_number}: Time vs {label}')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(data['time'], data['el_power'], label='Output Energy', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Output Energy')
    plt.title(f'Experiment {experiment_number}: Time vs Output Energy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Plot the data for each experiment
for i, df in enumerate(train_data):
    plot_experiment(df, i + 1, 'input_voltage', 'Input Voltage')

# Plot the test data
for i, df in enumerate(test_data):
    plot_experiment(df, i + 1,'input_voltage', 'Input Voltage')

# Calculate the change rate
for df in train_data:
    df['change_rate'] = df['el_power'].diff() / df['time'].diff()

# Get min, max, and mean values for the change rate
for i, df in enumerate(train_data):
    max_value = df['change_rate'].max()
    min_value = df['change_rate'].min()
    mean_value = df['change_rate'].mean()

    print(f"The maximum value in experiment {i + 1} is: {max_value}")
    print(f"The minimum value in experiment {i + 1} is: {min_value}")
    print(f"The mean value in experiment {i + 1} is: {mean_value}")

# Define change rate conditions for transition phases
# These thresholds were chosen after examining almost stationary phases
# based on visualizations and their statistics. Each pair of values represents
# the upper and lower bounds for the change rate during transition phases.
change_rate_conditions = [
    (38.49283827722983, -39.736841780402614),
    (100.22632874488022, -99.68670647456997),
    (98.75887692407991, -101.19320880015994),
    (99.98408542252992, -99.00885872435474),
    (97.14314833572728, -98.11770430004003),
    (154.0098171962502, -172.14620598439)
]


# Function to slice transition phases based on change rate conditions
def slice_transition_phases(df, condition):
    return df[(df['change_rate'] >= condition[0]) | (df['change_rate'] <= condition[1])]['change_rate']


# Slice based on change rate for transition phases
sliced_trans_diff = [slice_transition_phases(df, change_rate_conditions) for i, df in enumerate(train_data)]

# Print statistics for transition phases
for i, sub_series in enumerate(sliced_trans_diff):
    print(
        f'For experiment {i + 1} transition phases: the max is {max(sub_series)}, the min is {min(sub_series)}, and the mean is {np.mean(sub_series)}')



# Calculate and print mean value
trans_mean = [np.mean(sub_series) for sub_series in sliced_trans_diff]

print(f"Mean of transition change rates: {np.mean(trans_mean)}")   # The change rate used in earier experimentation for transition phases

# Applying Signal Transformers
# At first, I tried with fft but it did not help improve the model performance
# Uncomment to try applying the features extracted from fft

# # Function to calculate the Fourier transform of the input signal
# def fourier_transform_processing(data):
#     fft_data = []
#     for df in data:
#         signal = df['input_voltage'].values
#         fft_sig = fftpack.fft(signal)
#         df['magnitude'] = np.abs(fft_sig)
#         df['angle'] = np.angle(fft_sig)

#         fft_data.append(df)
#     return fft_data


# # Apply the Fourier transform to the data
# train_data = fourier_transform_processing(train_data)
# test_data = fourier_transform_processing(test_data)


# # Plotting the new features against the target in the training data
# for i, df in enumerate(train_data):
#     plot_experiment(df, i + 1,'magnitude', 'Magnitude of the Signal')

# # Plotting the new features against the target
# for i, df in enumerate(train_data):
#     plot_experiment(df, i + 1,'angle', 'Phase of the Signal')


# # Plotting the new features against the target in the test data
# for i, df in enumerate(test_data):
#     plot_experiment(df, i + 1,'magnitude', 'Magnitude of the Signal')

# # Plotting the new features against the target
# for i, df in enumerate(test_data):
#     plot_experiment(df, i + 1,'angle', 'Phase of the Signal')

# # Check the new resulted data
# for i, df in enumerate(train_data):
#     print(f'Experiment {i + 1}')
#     print(df[['magnitude', 'angle']].describe())
#     print(df.corr())


# # I also tried Applying Wavelets Transfrom
# # Unlike Fourier Transform, Wavelets transform preseves the resolution in both time and frequency domain
# def wavelets_transform_and_integrate(data_list, wavelets):
#     """
#     Apply wavelet transform to each DataFrame in the list and integrate the features into each DataFrame.

#     Parameters:
#     - data_list: List of DataFrames to which wavelet transform will be applied.
#     - wavelets: List of wavelet names for each DataFrame.

#     Returns:
#     - Updated data_list with wavelet features integrated.
#     """
#     updated_data_list = []

#     for df, wavelet in zip(data_list, wavelets):
#         coeffs = pywt.wavedec(df['input_voltage'], wavelet, level=4)

#         coeffs_flattened = np.concatenate([np.ravel(coef) for coef in coeffs])
#         coeffs_df = pd.DataFrame([coeffs_flattened], columns=[f'{wavelet}_coef_{i}' for i in range(len(coeffs_flattened))])

#         updated_df = pd.concat([df.reset_index(drop=True), coeffs_df.reset_index(drop=True)], axis=1)
#         updated_data_list.append(updated_df)

#     return updated_data_list


# # Define wavelets for train and test data
# train_wavelets = ['db1', 'db1', 'db1', 'db2', 'db2', 'db2']
# test_wavelets = ['db2', 'db1']

# # Extract and integrate wavelet features
# train_data_with_features = wavelets_transform_and_integrate(train_data, train_wavelets)
# test_data_with_features = wavelets_transform_and_integrate(test_data, test_wavelets)


# # Check the new data
# for i, df in enumerate(train_data_with_features):
#     print(f'Updated Training Data {i + 1} with wavelet features:')
#     print(df.head())
#     print(df.shape)

# for i, df in enumerate(test_data_with_features):
#     print(f'Updated Test Data {i + 1} with wavelet features:')
#     print(df.head())
#     print(df.shape)



# Data preprocessing
def preprocess_data(data_list, scaler=None, fit_scaler=True, seq_length=451):
    X = []
    y = []

    for data in data_list:
        if fit_scaler:
            data_scaled = scaler.fit_transform(data['input_voltage'])
        else:
            data_scaled = scaler.transform(data['input_voltage'])

        for i in range(seq_length, len(data_scaled)):
            X.append(data_scaled[i - seq_length:i, :])
            y.append(data.loc[data.index[i], 'el_power'])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    return X, y



# Initialize the scaler
scaler = MinMaxScaler()

# Preprocess the data
X_train, y_train = preprocess_data(train_data, scaler=scaler, fit_scaler=True)
X_test, y_test = preprocess_data(test_data, scaler=scaler, fit_scaler=False)

# Reshape for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')


def loss_function(y_true, y_pred, beta_trans=1.0, beta_stat=0.6, lambda_k=1000):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    y_pred_diff = y_pred[1:] - y_pred[:-1]

    delta = 6.388  # I tried with the mean change rate I got above 17.299 but it did not help improve the model performance
    trans_penalty = tf.square(tf.abs(y_pred_diff) - delta)
    stat_penalty = tf.square(y_pred_diff)
    state_loss = tf.reduce_min([beta_trans * trans_penalty, beta_stat * stat_penalty], axis=0)

    # Compute mean loss and add regularization
    state_loss_mean = tf.reduce_mean(state_loss)
    mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
    total_loss = mse_loss + lambda_k * state_loss_mean

    return total_loss


# Define the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(451, 1)),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=lambda y_true, y_pred: loss_function(y_true, y_pred),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)


# # Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])


# Plot the RMSE
plt.plot(history.history['root_mean_squared_error'], label='RMSE')
plt.plot(history.history['val_root_mean_squared_error'], label='Val RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()







