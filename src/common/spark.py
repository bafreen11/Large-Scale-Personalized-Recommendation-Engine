import os

from pyspark.sql import SparkSession


def get_spark(app_name: str = "recsys", driver_memory: str = "4g") -> SparkSession:
    """Local Spark session for this 4-core / 7 GB WSL machine.

    "local[3]" means no cluster: the driver and the workers all run inside
    this one Python process's JVM, using 3 cores (one left for Windows/VS Code).
    Shuffle/spill goes to ~/spark-tmp because /tmp in WSL is RAM (tmpfs).
    """
    local_dir = os.path.expanduser("~/spark-tmp")
    os.makedirs(local_dir, exist_ok=True)
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[3]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.local.dir", local_dir)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "64")  # default 200 is too many for one laptop
        .getOrCreate()
    )
