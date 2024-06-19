# False Alarm Detection

This project implements a fire alarm system using machine learning to predict whether a detected situation is a true alarm or a false alarm based on various environmental factors.

## Description

The system takes into account several environmental parameters to make predictions. These parameters include:
- Ambient Temperature
- Calibration
- Unwanted Substance Deposition
- Humidity
- H2S Content
- Detected by (% of sensors)

The system is built using Flask, which provides a web interface for users to input data and receive predictions.

## Usage

To use the system, follow these steps:

1. Clone the repository to your local machine.

2. Ensure you have Python installed along with the required dependencies listed in `requirements.txt`.

3. Run the Flask application by executing the `app.py` file. You can do this by running the following command in your terminal:

    ```bash
    python app.py
    ```

4. Once the application is running, navigate to `http://localhost:5000` in your web browser.

5. Input the required environmental parameters into the form and click the "Predict" button.

6. The system will provide a prediction indicating whether it is a true alarm or a false alarm.

## Files

The project consists of the following files:

- `app.py`: Contains the Flask application code responsible for handling user input, making predictions, and displaying results.
- `model.py`: Contains the code for training and saving the machine learning model used by the system.
- `Historical Alarm Cases.xlsx`: Dataset containing historical alarm cases used for training the model.
- `model.pkl`: Pickle file containing the trained machine learning model.

## Installation

To install the required dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```
## Requirements

The project requires the following dependencies:

- Flaks
- Pandas
- Numpy
- Scikit-Learn

