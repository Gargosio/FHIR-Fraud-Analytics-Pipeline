import os
import json
import tempfile
from fhirclient import client
from fhirclient.models.observation import Observation
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date,  col, explode,split

# ── CREATE FHIR CLIENT HERE ── (top of file)
settings = {
    'app_id': 'fhir_etl_app',
    'api_base': 'http://localhost:8091/fhir'
}
smart = client.FHIRClient(settings=settings)

# Make sure client is ready (optional but good)
if not smart.ready:
    smart.prepare()
print("FHIR client initialized successfully")

# ── Function definition ──
def extract_fhir_resources(resource_class, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    search = resource_class.where({})  # add filters if needed

    for resource in search.perform_resources_iter(smart.server):   # ← now smart is visible
        file_path = os.path.join(output_dir, f"{resource.resource_type}_{count}.json")
        with open(file_path, 'w') as f:
            json.dump(resource.as_json(), f)
        count += 1

    print(f"Extracted {count} {resource_class.resource_type} resources to {output_dir}")

# Use absolute path to avoid resolution issues
doris_jar_path = os.path.abspath("jars/spark-doris-connector-spark-3.5-25.2.0.jar")


# ── Spark setup ──
spark = SparkSession.builder \
    .appName("FHIR_ETL_to_Doris") \
    .master("local[*]") \
    .config("spark.jars",doris_jar_path) \
    .config("spark.driver.extraClassPath",doris_jar_path) \
    .config("spark.executor.extraClassPath",doris_jar_path) \
    .getOrCreate()


print("Spark started with JAR:", doris_jar_path)


# ── Main ETL ──
with tempfile.TemporaryDirectory() as temp_dir:
    extract_fhir_resources(Observation, temp_dir)

    if os.listdir(temp_dir):
        df = spark.read.json(temp_dir + "/*.json")
        df.printSchema()

        # Transformation example
        transformed_df = df.select(
            col("id"),
	    col("subject.reference").alias("patient_id"),
           # col("effectiveDateTime").alias("observation_date"),
        to_date(col("effectiveDateTime")).alias("observation_date"),
 
	explode(col("code.coding")).alias("coding"),
            col("valueQuantity.value").alias("value"),
            col("valueQuantity.unit").alias("unit"),
		to_date(col("meta.lastUpdated")).alias("updated_date"),
        split(col("encounter.reference"),"/")[1].alias("encounter_id")
        ).select(
            "id", 
            split(col("patient_id"), "/")[1].alias("patient_id"), 
            "observation_date", "value", "unit",
            col("coding.code").alias("code"),
            col("coding.display").alias("code_display"),"updated_date","encounter_id"
        ).filter(col("value").isNotNull())

        transformed_df.show(10, truncate=False)

        # Write to Doris
        transformed_df.write \
            .format("doris") \
            .option("doris.fenodes", "127.0.0.1:8030") \
            .option("doris.table.identifier", "fhir_db.observations") \
            .option("user", "root") \
            .option("password", "") \
            .mode("append") \
            .save()
    else:
        print("No resources extracted – check if HAPI has data.")

spark.stop()
print("ETL finished.")
