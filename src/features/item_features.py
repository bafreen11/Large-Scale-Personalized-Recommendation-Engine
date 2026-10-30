from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA, StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.functions import vector_to_array
from src.features import config as C


def genome_pca(spark):
    """Reads the PCA vectors written by precompute_genome_pca.py (movieId, genome_pca)."""
    return spark.read.parquet(C.GENOME_PCA_OUT)


def build_item_features(spark, train, cutoff, movies, genome):
    """One row per movie, a SNAPSHOT as of `cutoff` (use for serving / val / test)."""
    stats = train.groupBy("movieId").agg(
        F.count("*").alias("item_pop_all"),
        F.sum(F.when(F.col("ts") >= cutoff - 30 * C.DAY, 1).otherwise(0)).alias("item_pop_30d"),
        F.sum(F.when(F.col("ts") >= cutoff - 90 * C.DAY, 1).otherwise(0)).alias("item_pop_90d"),
        F.avg("rating").alias("item_avg_rating"),
    )
    global_mean = train.agg(F.avg("rating")).first()[0]

    items = (movies.join(stats, "movieId", "left")
                   .join(genome, "movieId", "left"))
    zeros = F.array_repeat(F.lit(0.0), C.PCA_K)
    items = (items
        .withColumn("has_genome", F.col("genome_pca").isNotNull().cast("int"))
        .withColumn("item_genome_pca", F.coalesce("genome_pca", zeros))
        .drop("genome_pca")
        .withColumnRenamed("year", "item_year")
        .withColumnRenamed("genres_multihot", "item_genres")
        .fillna({"item_pop_all": 0, "item_pop_30d": 0, "item_pop_90d": 0,
                 "item_avg_rating": global_mean}))
    return items
