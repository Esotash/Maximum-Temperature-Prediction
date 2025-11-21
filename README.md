# Maximum Temperature Prediction

This project provides a way to predict the maximum temperature.  The current version focuses on data processing and model training using PySpark.

## Features and Functionality

*   **Data Ingestion:** Reads weather data from a specified CSV file.
*   **Data Cleaning:**  Handles missing values by imputation with the mean.
*   **Feature Engineering:**  Creates features suitable for machine learning.
*   **Model Training:** Trains a Linear Regression model using PySpark's MLlib.
*   **Model Evaluation:** Evaluates the trained model using Root Mean Squared Error (RMSE).
*   **Logging:**  Logs model training progress, metrics, and relevant information.

## Technology Stack

*   **Python:** Primary programming language.
*   **PySpark:** Distributed data processing and machine learning framework.
*   **pandas:** Data manipulation and analysis.
*   **scikit-learn:** Used for model evaluation metrics.
*   **logging:**  Python's built-in logging module for recording events.

## Prerequisites

Before running this project, ensure you have the following installed and configured:

*   **Python:** Version 3.6 or higher.  It's recommended to use a virtual environment.
*   **Java Development Kit (JDK):** Required by Spark.  Version 8 or 11 is recommended.  Ensure `JAVA_HOME` environment variable is set correctly.
*   **Apache Spark:**  Download and install Apache Spark from the official website. Set the `SPARK_HOME` environment variable to the Spark installation directory and add `$SPARK_HOME/bin` to your `PATH`.
*   **PySpark:** Install PySpark using pip:
    ```bash
    pip install pyspark pandas scikit-learn
    ```

## Installation Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Esotash/Maximum-Temperature-Prediction.git
    cd Maximum-Temperature-Prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install pyspark pandas scikit-learn
    ```

## Usage Guide

To run the maximum temperature prediction script:

1.  **Prepare your data:** Place your weather data in a CSV file.

2.  **Configure the script:**  Modify the script to use the correct input file path and other configurations.

3.  **Run the script:**
    ```bash
    python main.py
    ```

    The script will:
    *   Read and process the data.
    *   Train a Linear Regression model.
    *   Evaluate the model.
    *   Log the results.

    The logs will show the progress of training and the final RMSE value.

## Contributing Guidelines

Contributions are welcome!  Here's how you can contribute:

1.  **Fork the repository.**
2.  **Create a new branch for your feature or bug fix:**
    ```bash
    git checkout -b feature/your-feature-name
    ```
3.  **Make your changes and commit them:**
    ```bash
    git add .
    git commit -m "Add your descriptive commit message"
    ```
4.  **Push your changes to your forked repository:**
    ```bash
    git push origin feature/your-feature-name
    ```
5.  **Create a pull request to the `main` branch of the original repository.**

Please ensure your code adheres to the following guidelines:

*   Use descriptive variable names.
*   Write clear and concise comments.
*   Test your code thoroughly.

## License Information

License information is not specified in the repository.  All rights are reserved unless otherwise specified.

## Contact/Support Information

For questions or support, please contact the repository owner through GitHub.