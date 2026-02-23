import os
import json
import tempfile
import logging
from fhirclient import client
from fhirclient.models.patient import Patient
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
    .appName("FHIR_ETL_to_Doris_Patients") \
    .master("local[*]") \
    .config("spark.jars", doris_jar_path) \
    .config("spark.driver.extraClassPath", doris_jar_path) \
    .config("spark.executor.extraClassPath", doris_jar_path) \
    .getOrCreate()

logger.info(f"Spark started with JAR: {doris_jar_path}")
logger.info(f"Spark version: {spark.version}")

# ── Main ETL ──
with tempfile.TemporaryDirectory() as temp_dir:
    # Extract Patient resources
    extracted_count = extract_fhir_resources(Patient, temp_dir)

    if extracted_count > 0 and os.listdir(temp_dir):
        try:
            # Read all JSON files
            df = spark.read.json(os.path.join(temp_dir, "*.json"))

            # Filter only Patient resources
            patient_df = df.filter(col("resourceType") == "Patient")

            if patient_df.count() > 0:
                logger.info("Raw Patient schema:")
                patient_df.printSchema()

                # ── Transformation: only SSN identifier ──
                transformed_patient_df = patient_df.select(
                    col("id"),
                    col("meta.lastUpdated").alias("last_updated"),
                    to_date(col("birthDate")).alias("birth_date"),
                    col("gender"),
                    col("maritalStatus.coding")[0].code.alias("marital_status_code"),
                    col("maritalStatus.text").alias("marital_status_text"),
                    col("name")[0].family.alias("family_name"),
                    explode_outer(col("name")[0].given).alias("given_name"),
                    col("name")[0].prefix[0].alias("prefix"),
                    col("address")[0].line[0].alias("address_line"),
                    col("address")[0].city.alias("city"),
                    col("address")[0].state.alias("state"),
                    col("address")[0].country.alias("country"),
                    col("extension").getItem(3).valueCode.alias("birth_sex"),
                    explode_outer(col("identifier")).alias("identifier")
                ).filter(
                    array_contains(col("identifier.type.coding.code"),"SS")   # only keep rows with SSN
                ).select(
                    col("id"),
                    col("last_updated"),
                    col("birth_date"),
                    col("gender"),
                    col("marital_status_code"),
                    col("marital_status_text"),
                    col("family_name"),
                    col("given_name"),
                    col("prefix"),
                    col("address_line"),
                    col("city"),
                    col("state"),
                    col("country"),
                    col("identifier.type.coding")[0].code.alias("identifier_type"),  # 'SS'
                    col("identifier.value").alias("identifier_value"),
                    col("birth_sex")
                ).filter(
                    col("identifier_value").isNotNull()
                )

                # Preview
                transformed_patient_df.show(10, truncate=False)

                # Write to Doris – simple SSN table
                transformed_patient_df.write \
                    .format("doris") \
                    .option("doris.fenodes", "127.0.0.1:8030") \
                    .option("doris.table.identifier", "fhir_db.patients") \
                    .option("user", "root") \
                    .option("password", "") \
                    .mode("append") \
                    .save()

                logger.info("Patient data successfully written to Doris!")
            else:
                logger.warning("No Patient resources found in extracted data.")
        except Exception as e:
            logger.error(f"Patient ETL failed: {e}")
            raise
    else:
        logger.warning("No resources extracted at all – check HAPI server.")

spark.stop()
logger.info("ETL finished.")
