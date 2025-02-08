# Recommendation System with Apache Spark, Python, and Machine Learning

This project demonstrates how to build a recommendation system using Apache Spark for distributed data processing, Python for machine learning, and common data science libraries.  It uses the Alternating Least Squares (ALS) algorithm for collaborative filtering.

## Project Overview

The recommendation system analyzes user interaction data (e.g., ratings, purchases, views) to predict user preferences and generate personalized recommendations.  It leverages Spark's distributed computing capabilities to handle large datasets efficiently.

## Data Files

The project uses three CSV files:

* **`users.csv`:** Contains user information (e.g., `user_id`, `age`, `gender`, `location`).
* **`interactions.csv`:** Records user interactions with items (e.g., `user_id`, `item_id`, `rating`, `timestamp`).
* **`items.csv`:** Contains item metadata (e.g., `item_id`, `name`, `category`, `price`).

Example data files are provided in the repository.  You can replace these with your own datasets.

## Code Files

* **`main.py`:** The main Python script that implements the recommendation system. It loads the data, preprocesses it, trains the ALS model, evaluates the model, generates recommendations, and saves the results to a CSV file.

## Libraries

* `pyspark`
* `pandas`
* `numpy`
* `scikit-learn` (for evaluation metrics)

## Running the Project (Local)

1. **Prerequisites:**
    * Install Python 3.x.
    * Install Apache Spark.  You can download a pre-built version from the Apache Spark website or use a package manager like `brew` (on macOS). Make sure `SPARK_HOME` is set.
    * Install the required Python libraries:

    ```bash
    pip install pyspark pandas numpy scikit-learn
    ```

2. **Data:** Place the `users.csv`, `interactions.csv`, and `items.csv` files in the same directory as `main.py`.

3. **Run the Spark Application:** Use `spark-submit` to run the `main.py` script:

    ```bash
    spark-submit main.py
    ```

    This will start a local Spark session, load the data, train the model, evaluate it, generate recommendations, and save the results to `recommendations.csv` in the same directory.

## Running the Project (Docker - Spark Container Separate)

1. **Build the Spark Docker Image (If you don't have one):**

   ```bash
   docker pull apache/spark:latest  # Or a specific version
   ```
2. Run the Spark Container:

    ```bash
    docker run -it -p 8080:8080 -p 7077:7077 -v $(pwd):/home/jovyan apache/spark:latest
    ```

    Find Spark Master URL: In the Docker logs, find the Spark master URL `(e.g., spark://<container_id>:7077 or spark://localhost:7077)`.

    Set `SPARK_MASTER_URL` and run `main.py`:

    Bash
    ```bash
    export SPARK_MASTER_URL=spark://localhost:7077  # Replace with actual URL
    spark-submit main.py # or python main.py
    ```

    Make sure your data files (users.csv, interactions.csv, items.csv) are in the directory you mounted with the -v flag (your current directory).

    Running the Project (Docker - Spark and Application in One Container):
    Build the Docker Image:
    Create a Dockerfile in the same directory as your main.py and CSV files. Example:

    ```Dockerfile

    FROM jupyter/pyspark-notebook:latest

    USER root

    RUN pip install scikit-learn

    COPY main.py /home/jovyan/
    COPY users.csv /home/jovyan/
    COPY interactions.csv /home/jovyan/
    COPY items.csv /home/jovyan/

    WORKDIR /home/jovyan

    USER $NB_UID

    CMD ["spark-submit", "main.py"] # or CMD ["python", "main.py"]
    Build the image:
    ```
    ```Bash
    docker build -t my-spark-app .
    Run the Docker Container:
    ```
    ```Bash
    docker run -v $(pwd):/home/jovyan -p 8888:8888 my-spark-app # -p 8888:8888 is only if you want to use jupyter
    ```

## Output
The recommendations are saved in a CSV file named recommendations.csv in the same directory where you run the script.

## Further Enhancements
- Parameter Tuning: Experiment with different ALS parameters (e.g., maxIter, regParam) to improve model performance.
- More Metrics: Evaluate the model using additional metrics (e.g., precision@k, recall@k).
- Implicit Feedback: Adapt the code to handle implicit feedback data.
- Real-time Recommendations: Explore using Spark Streaming or Structured - Streaming for real-time recommendation generation.
- Production Deployment: Consider how to deploy the model and generate recommendations in a production environment.