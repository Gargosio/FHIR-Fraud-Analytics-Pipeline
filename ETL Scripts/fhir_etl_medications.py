import os
import json
import tempfile
import logging
from fhirclient import client
from fhirclient.models.medicationrequest import MedicationRequest
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
    .appName("FHIR_ETL_to_Doris_Medication") \
    .master("local[*]") \
    .config("spark.jars", doris_jar_path) \
    .config("spark.driver.extraClassPath", doris_jar_path) \
    .config("spark.executor.extraClassPath", doris_jar_path) \
    .getOrCreate()

logger.info(f"Spark started with JAR: {doris_jar_path}")
logger.info(f"Spark version: {spark.version}")

# ── Main ETL ──
with tempfile.TemporaryDirectory() as temp_dir:
    # Extract Medication resources
    extracted_count = extract_fhir_resources(MedicationRequest, temp_dir)

    if extracted_count > 0 and os.listdir(temp_dir):
        try:
            # Read all JSON files
            df = spark.read.json(os.path.join(temp_dir, "*.json"))

            # Filter only Medication resources
            medication_df = df.filter(col("resourceType") == "MedicationRequest")

            if medication_df.count() > 0:
                logger.info("Raw MedicationRequest schema:")
                medication_df.printSchema()

                # ── Transformation: only SSN identifier ──
                transformed_medication_df = medication_df.select(
                    col("id"),
                    col("meta.lastUpdated").alias("last_updated"),
                    col("status"),
                    col("intent").alias("intent"),
                    explode_outer("medicationCodeableConcept.coding").alias("medication"),
                    split(col("subject.reference"), "/")[1].alias("patient_id"),
                    split(col("encounter.reference"), "/")[1].alias("encounter_id"),
                    col("authoredOn").alias("issued_on"),
                    split(col("requester.reference"), "/")[1].alias("practitioner_id"),
                    col("requester.display").alias("practitioner")
                    ).select(
                    col("id"),
                    col("last_updated"),
                    col("status"),
                    col("intent"),
                    col("medication.display").alias("medication"),
                    col("patient_id"),
                    col("encounter_id"),
                    col("issued_on"),
                    col("practitioner_id"),
                    col("practitioner")             
                )#.filter(
                #    col("id").isNotNull()
                #)

                # Preview
                transformed_medication_df.show(10, truncate=False)

                # Write to Doris – simple SSN table
                transformed_medication_df.write \
                    .format("doris") \
                    .option("doris.fenodes", "127.0.0.1:8030") \
                    .option("doris.table.identifier", "fhir_db.medication") \
                    .option("user", "root") \
                    .option("password", "") \
                    .mode("append") \
                    .save()

                logger.info("Medication data successfully written to Doris!")
            else:
                logger.warning("No Medication resources found in extracted data.")
        except Exception as e:
            logger.error(f"Medication ETL failed: {e}")
            raise
    else:
        logger.warning("No resources extracted at all – check HAPI server.")

spark.stop()
logger.info("ETL finished.")
