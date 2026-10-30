"""Step 2 entry point (run from the project root):  scripts/run_phase2.sh   (or PYTHONPATH=. spark-submit src/features/run_phase2.py)"""
import os
from pyspark.sql import Window, functions as F
from src.features import config as C
from src.features.common import get_spark, load_train, get_cutoff, load_movies
from src.features.item_features import genome_pca, build_item_features
from src.features.user_features import build_user_features
from src.features.interactions import build_train_interactions


def main():
    # Quick test:  SAMPLE_FRAC=0.02 spark-submit ... run_phase2.py   (2% of users, separate output folder)
    frac = float(os.environ.get("SAMPLE_FRAC", "0") or 0)
    if frac:
        C.OUT_DIR = "data/features_sample"
        C.USER_FEATURES_OUT = f"{C.OUT_DIR}/user_features.parquet"
        C.ITEM_FEATURES_OUT = f"{C.OUT_DIR}/item_features.parquet"
        C.TRAIN_INTERACTIONS_OUT = f"{C.OUT_DIR}/train_interactions.parquet"
    tmp_dir = os.path.expanduser("~/phase2_tmp" + ("_sample" if frac else ""))   # Linux disk = fast

    spark = get_spark()
    train = load_train(spark)
    if frac:
        train = train.filter(F.pmod(F.hash("userId"), F.lit(1000)) < int(frac * 1000))
        print(f"SAMPLE MODE: ~{frac:.1%} of users -> {C.OUT_DIR}")
    train = train.cache()
    cutoff = get_cutoff(train)
    movies = load_movies(spark)
    print(f"train rows={train.count():,}  cutoff_ts={cutoff}")

    # 1) item snapshot (also holds the static genre/year/genome columns)
    item_features = build_item_features(spark, train, cutoff, movies, genome_pca(spark)).cache()
    item_features.write.mode("overwrite").parquet(C.ITEM_FEATURES_OUT)
    print("item_features written")

    # 2) user snapshot
    build_user_features(train, movies).write.mode("overwrite").parquet(C.USER_FEATURES_OUT)
    print("user_features written")

    # 3) as-of-time interaction table (staged)
    inter = build_train_interactions(spark, train, movies, item_features, tmp_dir)
    inter.write.mode("overwrite").parquet(C.TRAIN_INTERACTIONS_OUT)
    print("train_interactions written")

    validate(spark, train)


def validate(spark, train):
    """Cheap sanity checks. If any assert fails, fix before Phase 3."""
    inter = spark.read.parquet(C.TRAIN_INTERACTIONS_OUT)
    users = spark.read.parquet(C.USER_FEATURES_OUT)
    items = spark.read.parquet(C.ITEM_FEATURES_OUT)

    n_train, n_inter = train.count(), inter.count()
    assert n_train == n_inter, f"row count changed: {n_train} -> {n_inter} (duplicate movieIds in movies.csv?)"

    # no feature leakage: a user's first rating must see zero history
    w = Window.partitionBy("userId").orderBy("timestamp", "movieId")
    first = inter.withColumn("rn", F.row_number().over(w)).filter("rn = 1")
    assert first.filter("user_n_ratings != 0 or size(user_seq) != 0").count() == 0, "history leaked into first row"

    # the as-of count of each user's LAST rating == snapshot count - 1
    last = (inter.withColumn("rn", F.row_number().over(w.orderBy(F.col("timestamp").desc(), F.col("movieId").desc())))
                 .filter("rn = 1").select("userId", "user_n_ratings"))
    bad = last.join(users.select("userId", "user_n_ratings").withColumnRenamed("user_n_ratings", "snap"), "userId") \
              .filter(F.col("user_n_ratings") != F.col("snap") - 1).count()
    assert bad == 0, f"{bad} users where as-of count != snapshot - 1"

    print("label balance:"); inter.groupBy("label").count().show()
    print("users:", users.count(), " items:", items.count(), " interactions:", n_inter)
    inter.printSchema()


if __name__ == "__main__":
    main()
