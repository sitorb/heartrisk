# Heart Attack Risk Prediction

## Overview

This project aims to predict the risk of heart attack using machine learning, specifically the K-Nearest Neighbors (KNN) algorithm. It involves data preprocessing, model training, evaluation, and hyperparameter tuning to achieve optimal performance.

## Code Structure

The project code is structured as follows:

1. **Data Loading and Preprocessing:**
    - Imports necessary libraries (`pandas`, `numpy`, `sklearn`, etc.).
    - Loads the heart attack risk dataset using `pd.read_csv`.
    - Handles missing values (if any).
    - Performs one-hot encoding for categorical features using `pd.get_dummies`.
    - Splits the data into features (X) and target (y).
    - Scales numerical features using `StandardScaler`.
2. **Model Building and Evaluation:**
    - Splits the data into training and testing sets using `train_test_split`.
    - Creates a KNN classifier with an initial value of `n_neighbors`.
    - Trains the model using the training data.
    - Makes predictions on the testing data.
    - Evaluates the model's performance using metrics like accuracy, classification report, and confusion matrix.
3. **Hyperparameter Tuning:**
    - Uses cross-validation to find the optimal value of `n_neighbors` for the KNN classifier.
    - Iterates through a range of `k` values and calculates the average cross-validation score.
    - Selects the `k` value that yields the highest accuracy.
4. **Visualization:**
    - Visualizes the data using `seaborn` and `matplotlib`.
    - Plots the relationship between `k` values and accuracy during hyperparameter tuning.
![image](https://github.com/user-attachments/assets/76ae2ed9-2c64-4dbb-bb0b-17e8066c32ae)


## Logic

The project follows a standard machine learning workflow:

1. **Data Preparation:** Cleaning, transforming, and preparing the data for model training.
2. **Model Selection:** Choosing the KNN algorithm for its simplicity and effectiveness in classification tasks.
3. **Training:** Fitting the model to the training data to learn patterns.
4. **Evaluation:** Assessing the model's performance on unseen data using relevant metrics.
5. **Tuning:** Optimizing the model's parameters to improve its performance.
6. **Visualization:** Presenting the results and insights through visual representations.

## Technology

The project utilizes the following technologies:

- **Python:** The primary programming language used for data processing, model building, and evaluation.
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical computations.
- **Scikit-learn:** For machine learning algorithms, model selection, and evaluation metrics.
- **Matplotlib and Seaborn:** For data visualization.

## Other Relevant Aspects

- **Dataset:** The project uses a heart attack risk dataset, which should be provided or described in detail.
- **Assumptions:** Any assumptions made during data preprocessing or model building should be documented.
- **Limitations:** Potential limitations of the model or the dataset should be acknowledged.
- **Future Work:** Possible improvements or extensions to the project can be outlined.

## Conclusion

This project demonstrates the application of machine learning for heart attack risk prediction. By leveraging the KNN algorithm and following a structured workflow, the model achieves reasonable accuracy. The project also highlights the importance of data preprocessing, hyperparameter tuning, and visualization in building effective machine learning models.
