from pyspark.sql import Window, functions as F
from src.features import config as C
from src.features.user_features import affinity_array

G = len(C.GENRES)


def user_asof(train, movies):
    """Per rating: user features from that user's EARLIER ratings only. Narrow rows, row-based windows
    (order: ts, then movieId). Output keys: userId, movieId."""
    global_mean = train.agg(F.avg("rating")).first()[0]
    df = (train.join(F.broadcast(movies.select("movieId", "genres_multihot")), "movieId", "left")
               .withColumn("genres_multihot",
                           F.coalesce("genres_multihot", F.array_repeat(F.lit(0), G)))
               .withColumn("day", F.floor(F.col("ts") / C.DAY))
               .select("userId", "movieId", "ts", "day", "rating", "label", "genres_multihot"))

    u = Window.partitionBy("userId").orderBy("ts", "movieId")
    prior = u.rowsBetween(Window.unboundedPreceding, -1)
    lastn = u.rowsBetween(-C.SEQ_LEN, -1)

    day_w = Window.partitionBy("userId", "day").orderBy("ts", "movieId")
    df = df.withColumn("is_new_day", (F.row_number().over(day_w) == 1).cast("int"))

    df = (df.withColumn("user_seq", F.collect_list("movieId").over(lastn))
            .withColumn("user_n_ratings", F.count(F.lit(1)).over(prior))
            .withColumn("_sum", F.sum("rating").over(prior))
            .withColumn("user_days_active", F.coalesce(F.sum("is_new_day").over(prior), F.lit(0)))
            .withColumn("_npos", F.coalesce(F.sum("label").over(prior), F.lit(0))))
    for k in range(G):
        df = df.withColumn(f"_g{k}", F.coalesce(
            F.sum(F.col("label") * F.col("genres_multihot")[k]).over(prior), F.lit(0)))
    df = (df.withColumn("user_avg_rating",
                        F.when(F.col("user_n_ratings") > 0, F.col("_sum") / F.col("user_n_ratings"))
                         .otherwise(F.lit(global_mean)))
            .withColumn("user_genre_affinity",
                        affinity_array([F.col(f"_g{k}") for k in range(G)], F.col("_npos"))))
    return df.select("userId", "movieId", "user_seq", "user_n_ratings", "user_avg_rating",
                     "user_genre_affinity", "user_days_active")


def item_asof(train):
    """Item popularity/avg rating as of the START of each day (previous full days only).
    Built on a small daily table (movie, day) so window frames are tiny. Keys: movieId, day."""
    global_mean = train.agg(F.avg("rating")).first()[0]
    daily = (train.withColumn("day", F.floor(F.col("ts") / C.DAY))
                  .groupBy("movieId", "day")
                  .agg(F.count("*").alias("n"), F.sum("rating").alias("s")))
    w = Window.partitionBy("movieId").orderBy("day")

    def over(col, lo, hi):
        return F.coalesce(F.sum(col).over(w.rangeBetween(lo, hi)), F.lit(0))

    daily = (daily
        .withColumn("item_pop_all", over("n", Window.unboundedPreceding, -1))
        .withColumn("item_pop_30d", over("n", -30, -1))
        .withColumn("item_pop_90d", over("n", -90, -1))
        .withColumn("_s_all", over("s", Window.unboundedPreceding, -1)))
    daily = daily.withColumn("item_avg_rating",
                             F.when(F.col("item_pop_all") > 0, F.col("_s_all") / F.col("item_pop_all"))
                              .otherwise(F.lit(global_mean)))
    return daily.select("movieId", "day", "item_pop_all", "item_pop_30d", "item_pop_90d", "item_avg_rating")


def build_train_interactions(spark, train, movies, item_features, tmp_dir):
    """Stage 1 (user) and stage 2 (item) are written to tmp_dir, then joined. One row per rating."""
    u_path, i_path = f"{tmp_dir}/user_asof", f"{tmp_dir}/item_asof"
    user_asof(train, movies).write.mode("overwrite").parquet(u_path)
    print("[stage 1/2] user as-of features written")
    item_asof(train).write.mode("overwrite").parquet(i_path)
    print("[stage 2/2] item as-of features written")

    u = spark.read.parquet(u_path)
    i = spark.read.parquet(i_path)
    static = F.broadcast(item_features.select("movieId", "item_year", "item_genres",
                                              "item_genome_pca", "has_genome"))
    base = train.select("userId", "movieId", "label", F.col("ts").alias("timestamp"),
                        F.floor(F.col("ts") / C.DAY).alias("day"))
    # (userId, movieId) is unique in MovieLens; run_phase2's row-count check would catch duplicates
    out = (base.join(u, ["userId", "movieId"]).join(i, ["movieId", "day"]).join(static, "movieId", "left"))
    return out.select("userId", "movieId", "label", "timestamp",
                      "user_seq", "user_n_ratings", "user_avg_rating", "user_genre_affinity",
                      "user_days_active",
                      "item_year", "item_genres", "item_pop_all", "item_pop_30d", "item_pop_90d",
                      "item_avg_rating", "item_genome_pca", "has_genome")
