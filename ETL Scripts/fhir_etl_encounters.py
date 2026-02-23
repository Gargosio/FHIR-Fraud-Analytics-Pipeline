import os
import json
import tempfile
import logging
from fhirclient import client
from fhirclient.models.encounter import Encounter
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, explode_outer, when,array_contains,split



# ── Logging setup ──
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── FHIR Client ──
settings = {
    'app_id': 'fhir_etl_app',
    'api_base': 'http://localhost:8091/fhir'
}
smart = client.FHIRClient(settings=settings)

if not smart.ready:
    smart.prepare()
logger.info("FHIR client initialized successfully")

# ── Extraction function ──
def extract_fhir_resources(resource_class, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    search = resource_class.where({})  # full load
    # search = resource_class.where({'_lastUpdated': 'gt2025-01-01'})  # incremental load

    for resource in search.perform_resources_iter(smart.server):
        file_path = os.path.join(output_dir, f"{resource.resource_type}_{count}.json")
        with open(file_path, 'w') as f:
            json.dump(resource.as_json(), f)
        count += 1

    logger.info(f"Extracted {count} {resource_class.resource_type} resources to {output_dir}")
    return count

# ── Spark setup ──
doris_jar_path = os.path.abspath("jars/spark-doris-connector-spark-3.5-25.2.0.jar")

spark = SparkSession.builder \
    .appName("FHIR_ETL_to_Doris_Encounters") \
    .master("local[*]") \
    .config("spark.jars", doris_jar_path) \
    .config("spark.driver.extraClassPath", doris_jar_path) \
    .config("spark.executor.extraClassPath", doris_jar_path) \
    .getOrCreate()

logger.info(f"Spark started with JAR: {doris_jar_path}")
logger.info(f"Spark version: {spark.version}")

# ── Main ETL ──
with tempfile.TemporaryDirectory() as temp_dir:
    # Extract Encounters resources
    extracted_count = extract_fhir_resources(Encounter, temp_dir)

    if extracted_count > 0 and os.listdir(temp_dir):
        try:
            # Read all JSON files
            df = spark.read.json(os.path.join(temp_dir, "*.json"))

            # Filter only Encounters resources
            encounters_df = df.filter(col("resourceType") == "Encounter")

            if encounters_df.count() > 0:
                logger.info("Raw Encounters schema:")
                encounters_df.printSchema()

                # ── Transformation: only SSN identifier ──
                transformed_encounters_df = encounters_df.select(
                    col("id"),
                    col("meta.lastUpdated").alias("last_updated"),
                    col("identifier")[0].use.alias("identifier_use"),
                    col("identifier")[0].value.alias("identifier_value"),
                    col("status"),
                    explode_outer("type").alias("type"),
                    split(col("subject.reference"), "/")[1].alias("patient_id"),
                    explode_outer("participant").alias("participant"),
                    split(col("participant")[0].individual.reference, "/")[1].alias("practitioner_id"),
                    col("participant")[0].individual.display.alias("practitioner"),
                    col("period.start").alias("encounter_start"),
                    col("period.end").alias("encounter__end"),
                    split(col("serviceProvider.reference"), "/")[1].alias("service_provider_id"),
                    split(col("location")[0].location.reference, "/")[1].alias("service_provider_location")    
                ).select(
                    col("id"),
                    col("last_updated"),
                    col("identifier_use"),
                    col("identifier_value"),
                    col("status"),
                    col("practitioner_id"),
                    col("practitioner"),
                    col("encounter_start"),
                    col("encounter__end"),
                    col("service_provider_id"),
                    col("service_provider_location"),
                    col("type.text").alias("encounter_type"),
                    col("patient_id"),
                    col("participant.type")[0].text.alias("practitioner_type")                    
                )#.filter(
                #    col("id").isNotNull()
                #)

                # Preview
                transformed_encounters_df.show(10, truncate=False)

                # Write to Doris – simple SSN table
                transformed_encounters_df.write \
                    .format("doris") \
                    .option("doris.fenodes", "127.0.0.1:8030") \
                    .option("doris.table.identifier", "fhir_db.encounters") \
                    .option("user", "root") \
                    .option("password", "") \
                    .mode("append") \
                    .save()

                logger.info("Encounters data successfully written to Doris!")
            else:
                logger.warning("No Encounters resources found in extracted data.")
        except Exception as e:
            logger.error(f"Encounters ETL failed: {e}")
            raise
    else:
        logger.warning("No resources extracted at all – check HAPI server.")

spark.stop()
logger.info("ETL finished.")
