# Stock Price Prediction with LSTM

## Project Description
This project demonstrates how to use Long Short-Term Memory (LSTM) neural networks to predict high stock prices for Microsoft (MSFT) based on historical stock data. We use historical stock data, preprocess it, and build an LSTM model to make predictions. The project aims to predict high stock prices, but the code can be easily adapted for other stock-related predictions.

## Dependencies
- Python 3
- Required Python libraries:
  - pandas
  - numpy
  - seaborn
  - scikit-learn (for train-test split and mean_squared_error)
  - keras (for deep learning)
  
You can install these dependencies using `pip`:
```bash
pip install pandas numpy seaborn scikit-learn keras
```

## Getting Started
1. Clone this repository or download the project files.
2. Download historical stock data (e.g., MSFT.csv)  from kaggle[https://www.kaggle.com/datasets/varpit94/microsoft-stock-data]
3. Open and run the Jupyter Notebook or Python script in your development environment.

## Preprocessing
- The Date column is removed from the dataset as it doesn't affect predictions.
- The data is normalized using Min-Max scaling to bring all features within the range [0, 1].

## Train-Test Split
- The dataset is split into training and testing sets.
- Features (X) are created by dropping the "High" column, and the target variable (Y) is set as the "High" column.

## LSTM Model
- The model architecture consists of multiple LSTM layers with dropout to prevent overfitting.
- The final layer is a dense layer to predict the high stock prices.
- The model is compiled with the Adam optimizer and Mean Squared Error loss.
- It's trained for 20 epochs with a batch size of 32.

## Results
- Predictions are made on the test set.
- The Mean Squared Error (MSE) is calculated, and a 1 - MSE score is presented as a metric for model performance.

## Usage
1. Make sure you have all the required dependencies installed.
2. download the dataset from kaggle[https://www.kaggle.com/datasets/varpit94/microsoft-stock-data]
3. Run the Jupyter Notebook or Python script.
4. Check the model's performance with the 1 - MSE score.
