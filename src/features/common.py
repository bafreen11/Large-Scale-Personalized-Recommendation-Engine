from pyspark.sql import SparkSession, functions as F
from src.features import config as C


def get_spark(app="phase2-features"):
    return (SparkSession.builder.appName(app)
            .config("spark.sql.shuffle.partitions", "32")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def load_train(spark):
    """Training ratings + label. Everything downstream is built from THIS table only
    (never from validation/test), which is what prevents leakage."""
    df = spark.read.parquet(C.TRAIN_PATH).select(
        F.col(C.USER).cast("int").alias("userId"),
        F.col(C.ITEM).cast("int").alias("movieId"),
        F.col(C.RATING).cast("double").alias("rating"),
        F.col(C.TS).cast("long").alias("ts"),
    )
    return df.withColumn("label", (F.col("rating") >= C.POSITIVE_THRESHOLD).cast("int"))


def get_cutoff(train):
    if C.CUTOFF_TS is not None:
        return int(C.CUTOFF_TS)
    return train.agg(F.max("ts")).first()[0] + 1


def load_movies(spark):
    """movieId, year, genres multi-hot (array<int> in C.GENRES order). Reads Phase 1's movies_clean."""
    m = spark.read.parquet(C.MOVIES_PATH)   # movieId, title, genres, year, genre_list
    hot = F.array(*[F.when(F.array_contains(F.col("genre_list"), name), 1).otherwise(0)
                    for name in C.GENRES])
    return m.select("movieId", "year", hot.alias("genres_multihot"))
