"""Step 3 prep: keep SAMPLE_PCT% of users and write model-ready tables to data/training/.

Run from the project root:
    SAMPLE_PCT=5 python -m src.training.sample        (SAMPLE_PCT=100 = all users)

The same hash bucket is used for every table, so a sampled user has train, val and test rows.
"""
from pyspark.sql import functions as F

from src.common.config import FEATURES_DIR, SAMPLE_PCT, SPLITS_DIR, TRAINING_DIR
from src.common.spark import get_spark


def keep(df):
    return df.filter(F.pmod(F.hash("userId"), F.lit(100)) < SAMPLE_PCT)


def main():
    spark = get_spark("training-sample")
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    out = lambda name: str(TRAINING_DIR / f"{name}.parquet")

    keep(spark.read.parquet(str(FEATURES_DIR / "train_interactions.parquet"))) \
        .repartition(8).write.mode("overwrite").parquet(out("train"))
    keep(spark.read.parquet(str(FEATURES_DIR / "user_features.parquet"))) \
        .coalesce(1).write.mode("overwrite").parquet(out("users"))
    for split in ["val", "test", "cold_start"]:
        keep(spark.read.parquet(str(SPLITS_DIR / split))) \
            .select("userId", "movieId", "rating", "label", "ts") \
            .coalesce(1).write.mode("overwrite").parquet(out(split))

    # items are not sampled: the whole catalogue is retrievable
    spark.read.parquet(str(FEATURES_DIR / "item_features.parquet")) \
        .join(spark.read.parquet(str(SPLITS_DIR / "movies_clean")).select("movieId", "title"), "movieId", "left") \
        .coalesce(1).write.mode("overwrite").parquet(out("items"))

    for name in ["train", "users", "val", "test", "cold_start", "items"]:
        df = spark.read.parquet(out(name))
        print(f"{name:10s} rows={df.count():>10,}  users={df.select('userId').distinct().count() if 'userId' in df.columns else '-'}")
    spark.stop()


if __name__ == "__main__":
    main()
