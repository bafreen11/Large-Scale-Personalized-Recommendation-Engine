from pyspark.sql import SparkSession, functions as F, Window

P, OUT = "data/parquet", "data/splits"

spark = (SparkSession.builder.appName("split")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "16")
         .getOrCreate())

ratings = spark.read.parquet(f"{P}/ratings")
movies = spark.read.parquet(f"{P}/movies")

# 1. Unix-seconds timestamp for easy comparison and percentiles
ratings = ratings.withColumn("ts", F.col("timestamp").cast("long"))

# 2. Dedupe: keep the latest rating per (user, movie)
w = Window.partitionBy("userId", "movieId").orderBy(F.col("ts").desc())
ratings = ratings.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")

# 3. Movie cleaning: year from title, genres as an array
movies_clean = (movies
    .withColumn("year_str", F.regexp_extract("title", r"\((\d{4})\)\s*$", 1))
    .withColumn("year", F.when(F.col("year_str") != "", F.col("year_str").cast("int")))
    .drop("year_str")
    .withColumn("genre_list",
        F.when(F.col("genres") == "(no genres listed)", F.array())
         .otherwise(F.split("genres", r"\|"))))
movies_clean.write.mode("overwrite").parquet(f"{OUT}/movies_clean")
# 4. Label
ratings = ratings.withColumn("label", (F.col("rating") >= 4.0).cast("int"))

# 5. Global time split
c1, c2 = ratings.approxQuantile("ts", [0.90, 0.95], 0.0005)
train = ratings.filter(F.col("ts") < c1)
val = ratings.filter((F.col("ts") >= c1) & (F.col("ts") < c2))
test = ratings.filter(F.col("ts") >= c2)

# 6. Min-5 filter, counted on TRAIN only
good_users = (train.groupBy("userId").count().filter("count >= 5").select("userId"))
good_movies = (train.groupBy("movieId").count().filter("count >= 5").select("movieId"))
train = (train.join(good_users, "userId", "left_semi")
              .join(good_movies, "movieId", "left_semi"))

# 7. Keep only val/test rows whose user AND movie are in (filtered) train
train_users = train.select("userId").distinct()
train_movies = train.select("movieId").distinct()

def split_warm_cold(df):
    in_movies = df.join(train_movies, "movieId", "left_semi")
    warm = in_movies.join(train_users, "userId", "left_semi")
    cold = in_movies.join(train_users, "userId", "left_anti")
    return warm, cold

val, val_cold = split_warm_cold(val)
test, test_cold = split_warm_cold(test)
cold = val_cold.unionByName(test_cold)

for name, df in [("train", train), ("val", val), ("test", test), ("cold_start", cold)]:
    df.write.mode("overwrite").parquet(f"{OUT}/{name}")

# 8. Summary for your notebook
print("cutoffs (unix):", c1, c2)
for name in ["train", "val", "test", "cold_start"]:
    d = spark.read.parquet(f"{OUT}/{name}")
    d.agg(F.count("*").alias("rows"),
          F.countDistinct("userId").alias("users"),
          F.countDistinct("movieId").alias("movies"),
          F.min(F.col("ts").cast("timestamp")).alias("first"),
          F.max(F.col("ts").cast("timestamp")).alias("last"),
          F.avg("label").alias("share_positive")).show(truncate=False)