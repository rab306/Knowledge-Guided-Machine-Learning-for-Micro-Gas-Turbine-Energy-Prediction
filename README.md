Here is a README file for your project:

---

# Knowledge Guided Machine Learning for Micro Gas Turbine Electrical Energy Prediction

## Project Overview

This project focuses on predicting the electrical power output of a micro gas turbine using time series data of input voltage and output energy. The approach integrates knowledge of permissible system states with a machine learning model by adding constraints to the model's loss function. The constraints are based on the constant change rates of the system in stationary and transition states.

## Data

The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/994/micro+gas+turbine+electrical+energy+prediction). It consists of time series data for input voltage and output energy of a micro gas turbine.

## Introductory Paper

The methodology and theoretical foundations for the model are based on the paper titled ["Knowledge Guided Machine Learning: Combining System Knowledge with Data for Dynamical System Prediction"](https://dl.acm.org/doi/10.1145/3632775.3661967).

## Project Steps

1. **Read the Data**
   - Function to read CSV files from specified folders for training and test datasets.

2. **Visualize the Data**
   - Plot input voltage and output energy over time for both training and test datasets.

3. **Calculate Change Rate**
   - Compute the change rate of output energy to analyze the dynamics of the system.

4. **Slice Transition Phases**
   - Define and slice transition phases based on change rate conditions.

5. **Signal Processing Techniques**
   - Apply Fourier Transform and Wavelet Transform to extract features from the signal.

6. **Preprocess the Data**
   - Normalize data and prepare it for model training.

7. **Define Custom Loss Function**
   - Implement a loss function incorporating system constraints to guide the learning process.

8. **Define the Model**
   - Build and compile an LSTM model for time series prediction.

9. **Train the Model**
   - Train the model with early stopping and plot training history.

## Code Overview

### Import Libraries
The code imports necessary libraries including `pandas`, `numpy`, `matplotlib`, `tensorflow`, and `pywt`.

### Data Reading
Functions for reading data from CSV files and printing dataset shapes and information.

### Data Visualization
Functions for plotting input voltage and output energy.

### Change Rate Calculation
Calculation of the change rate of output energy and printing of its statistics.

### Transition Phase Slicing
Function to slice transition phases based on predefined change rate conditions.

### Signal Processing
Commented-out sections for applying Fourier Transform and Wavelet Transform.

### Data Preprocessing
Function to preprocess data, including scaling and reshaping for LSTM input.

### Custom Loss Function
Definition of a custom loss function with constraints on change rates.

### Model Definition and Training
Construction and training of an LSTM model, with plotting of Root Mean Squared Error (RMSE) over epochs.

## Installation

To run the code, make sure to install the required packages:
```bash
pip install pandas matplotlib numpy scipy tensorflow scikit-learn pywt
```

## Usage

1. Place the CSV files in `train` and `test` folders.
2. Run the script to perform all steps from data reading to model training.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

