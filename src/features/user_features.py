from pyspark.sql import Window, functions as F
from src.features import config as C


def affinity_array(sum_cols, n_pos_col):
    """share of positives per genre = (#positives having genre g) / (#positives). 0 if no positives.
    NOTE: a movie has several genres, so the vector does not sum to 1 (it's per-genre, not a distribution)."""
    return F.array(*[F.when(n_pos_col > 0, c / n_pos_col).otherwise(F.lit(0.0)) for c in sum_cols])


def build_user_features(train, movies):
    """One row per user, a SNAPSHOT as of the cutoff (use for serving / val / test)."""
    # --- last N movieIds, oldest -> newest ---
    w = Window.partitionBy("userId").orderBy(F.col("ts").desc(), F.col("movieId").desc())
    last = (train.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= C.SEQ_LEN)
                 .groupBy("userId")
                 .agg(F.collect_list(F.struct("rn", "movieId")).alias("s")))
    last = last.select(
        "userId",
        F.transform(F.sort_array("s", asc=False), lambda x: x["movieId"]).alias("user_seq"))
    # sort rn DESC => biggest rn (oldest) first => chronological order

    # --- scalar stats ---
    stats = train.groupBy("userId").agg(
        F.count("*").alias("user_n_ratings"),
        F.avg("rating").alias("user_avg_rating"),
        F.countDistinct(F.floor(F.col("ts") / C.DAY)).alias("user_days_active"),
        F.sum("label").alias("user_n_pos"),
    )

    # --- genre affinity from positives ---
    pos = train.filter("label = 1").join(movies.select("movieId", "genres_multihot"), "movieId", "left")
    pos = pos.withColumn("genres_multihot",
                         F.coalesce("genres_multihot", F.array_repeat(F.lit(0), len(C.GENRES))))
    gsums = pos.groupBy("userId").agg(
        *[F.sum(F.col("genres_multihot")[k]).alias(f"g{k}") for k in range(len(C.GENRES))])

    out = (stats.join(last, "userId", "left").join(gsums, "userId", "left")
                .fillna(0, subset=[f"g{k}" for k in range(len(C.GENRES))]))
    out = out.withColumn(
        "user_genre_affinity",
        affinity_array([F.col(f"g{k}") for k in range(len(C.GENRES))], F.col("user_n_pos")))
    return out.drop(*[f"g{k}" for k in range(len(C.GENRES))], "user_n_pos")
