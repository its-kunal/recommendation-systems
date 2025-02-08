from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator  # For evaluating the model
from pyspark.sql.functions import col  # For DataFrame column operations

def main():
    # 1. Initialize SparkSession
    spark = SparkSession.builder.appName("RecommendationSystem").getOrCreate()

    # 2. Load Data (Adjust paths as needed)
    users_df = spark.read.csv("data/users.csv", header=True, inferSchema=True)
    interactions_df = spark.read.csv("data/interactions.csv", header=True, inferSchema=True)
    items_df = spark.read.csv("data/items.csv", header=True, inferSchema=True)

    # 3. Data Preprocessing (Example: Joining and filtering)
    interactions_with_user_item = interactions_df.join(users_df, "user_id").join(items_df, "item_id")
    filtered_interactions = interactions_with_user_item.filter("rating > 0")  # Remove interactions with invalid ratings

    # 4. Split Data into Training and Test Sets
    (training_data, test_data) = filtered_interactions.randomSplit([0.8, 0.2])

    # 5. Train the ALS Model
    als = ALS(
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        coldStartStrategy="drop",  # Handle new users/items
        # Other important parameters to tune:
        # maxIter=10,  # Maximum iterations
        # regParam=0.01,  # Regularization parameter
        # alpha=1.0 # implicit preference strength
    )
    model = als.fit(training_data)

    # 6. Evaluate the Model (using RMSE)
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-squared error = {rmse}")

    # 7. Generate Recommendations (Example: for all users)
    user_ids = users_df.select("user_id").distinct() # Get all distinct user ids
    item_ids = items_df.select("item_id").distinct() # Get all distinct item ids

    # Create all possible user-item pairs (for generating batch predictions)
    users_items_df = user_ids.crossJoin(item_ids)

    all_predictions = model.transform(users_items_df)

    # Filter out already interacted items, if you only want to recommend new items.
    # recommendations = all_predictions.filter(~col("item_id").isin([row.item_id for row in interactions_df.collect()])).orderBy("prediction", ascending=False)
    recommendations = all_predictions.orderBy("prediction", ascending=False)
    recommendations.show(10) # Show top 10 recommendations

    recommendations.coalesce(1).write.csv("outputs/recommendations.csv", header=True, mode="overwrite")  # mode="overwrite" will replace the file if it exists

    # 8. Save the Model (Optional)
    # model.save("recommendation_model")  # Save the trained model

    # 9. Stop SparkSession
    spark.stop()

if __name__ == "__main__":
    main()