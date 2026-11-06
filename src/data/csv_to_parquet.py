"""Day 1: convert the 6 MovieLens CSVs to Parquet and print a quick data profile.

Run from the project root:
    python -m src.data.csv_to_parquet
"""
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, FloatType, IntegerType, LongType, StringType, StructField, StructType,
)

from src.common.config import PARQUET_DIR, RAW_DIR, TRAIN_QUANTILE, VAL_QUANTILE
from src.common.spark import get_spark

# Explicit schemas: faster than inferSchema (which reads the file twice) and no type surprises.
SCHEMAS = {
    "ratings": StructType([
        StructField("userId", IntegerType()),
        StructField("movieId", IntegerType()),
        StructField("rating", FloatType()),
        StructField("timestamp", LongType()),
    ]),
    "movies": StructType([
        StructField("movieId", IntegerType()),
        StructField("title", StringType()),
        StructField("genres", StringType()),
    ]),
    "tags": StructType([
        StructField("userId", IntegerType()),
        StructField("movieId", IntegerType()),
        StructField("tag", StringType()),
        StructField("timestamp", LongType()),
    ]),
    "links": StructType([
        StructField("movieId", IntegerType()),
        StructField("imdbId", StringType()),   # string keeps leading zeros
        StructField("tmdbId", StringType()),
    ]),
    "genome-scores": StructType([
        StructField("movieId", IntegerType()),
        StructField("tagId", IntegerType()),
        StructField("relevance", DoubleType()),
    ]),
    "genome-tags": StructType([
        StructField("tagId", IntegerType()),
        StructField("tag", StringType()),
    ]),
}


def main():
    spark = get_spark("csv_to_parquet")
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    for name, schema in SCHEMAS.items():
        df = (
            spark.read
            .option("header", True)
            .option("quote", '"')
            .option("escape", '"')      # movie titles contain commas inside quotes
            .schema(schema)
            .csv(str(RAW_DIR / f"{name}.csv"))
        )
        out = PARQUET_DIR / name.replace("-", "_")
        df.write.mode("overwrite").parquet(str(out))
        print(f"{name:15s} rows={spark.read.parquet(str(out)).count():>12,}  -> {out}")

    # Profile of the main table: these numbers decide the Day 2 time split.
    ratings = spark.read.parquet(str(PARQUET_DIR / "ratings"))
    ratings.select(
        F.countDistinct("userId").alias("users"),
        F.countDistinct("movieId").alias("movies"),
        F.from_unixtime(F.min("timestamp")).alias("first_rating"),
        F.from_unixtime(F.max("timestamp")).alias("last_rating"),
        F.avg((F.col("rating") >= 4.0).cast("int")).alias("share_positive"),
    ).show(truncate=False)

    t_train, t_val = ratings.approxQuantile("timestamp", [TRAIN_QUANTILE, VAL_QUANTILE], 0.001)
    print("Planned time split cutoffs:")
    print(f"  train <  {spark.sql(f'select from_unixtime({int(t_train)})').first()[0]}")
    print(f"  val   <  {spark.sql(f'select from_unixtime({int(t_val)})').first()[0]}")
    print("  test  =  everything after")

    ratings.groupBy("rating").count().orderBy("rating").show()
    spark.stop()


if __name__ == "__main__":
    main()
