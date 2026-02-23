import os
import json
import tempfile
import logging
from fhirclient import client
from fhirclient.models.claim import Claim
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
    .appName("FHIR_ETL_to_Doris_Claims") \
    .master("local[*]") \
    .config("spark.jars", doris_jar_path) \
    .config("spark.driver.extraClassPath", doris_jar_path) \
    .config("spark.executor.extraClassPath", doris_jar_path) \
    .getOrCreate()

logger.info(f"Spark started with JAR: {doris_jar_path}")
logger.info(f"Spark version: {spark.version}")

# ── Main ETL ──
with tempfile.TemporaryDirectory() as temp_dir:
    # Extract Claims resources
    extracted_count = extract_fhir_resources(Claim, temp_dir)

    if extracted_count > 0 and os.listdir(temp_dir):
        try:
            # Read all JSON files
            df = spark.read.json(os.path.join(temp_dir, "*.json"))

            # Filter only Claims resources
            claims_df = df.filter(col("resourceType") == "Claim")

            if claims_df.count() > 0:
                logger.info("Raw Claims schema:")
                claims_df.printSchema()

                # ── Transformation: only SSN identifier ──
                transformed_claims_df = claims_df.select(
                    col("id"),
                    col("meta.lastUpdated").alias("last_updated"),
                    col("status"),
                    col("patient.reference").alias("patient_id"),
                    to_date(col("billablePeriod.start")).alias("billable_period_start"),
                    to_date(col("billablePeriod.end")).alias("billable_period_end"),
                    col("created").alias("create_date"),
                    split(col("provider.reference"), "/")[1].alias("organization"),
                    col("facility.display").alias("facility"),
                    col("supportingInfo")[0].valueReference.reference.alias("supporting_info"),                  
                    col("insurance")[0].coverage.display.alias("insurance"),
                    explode_outer("item").alias("item"),
                    col("total.value").alias("total_value")
                ).select(
                    col("id"),
                    col("last_updated"),
                    col("status"),
                    split(col("patient_id"), "/")[1].alias("patient_id"),
                    col("billable_period_start"),
                    col("billable_period_end"),
                    col("create_date"),
                    col("organization"),
                    col("facility"),
                    col("supporting_info"),
                    col("insurance"),
                    col("item.productOrService.text").alias("product_service"),
                    split(col("item.encounter")[0].reference, "/")[1].alias("encounter"),
                    col("total_value")
                )#.filter(
                #    col("id").isNotNull()
                #)

                # Preview
                transformed_claims_df.show(10, truncate=False)

                # Write to Doris – simple SSN table
                transformed_claims_df.write \
                    .format("doris") \
                    .option("doris.fenodes", "127.0.0.1:8030") \
                    .option("doris.table.identifier", "fhir_db.claims") \
                    .option("user", "root") \
                    .option("password", "") \
                    .mode("append") \
                    .save()

                logger.info("Claims data successfully written to Doris!")
            else:
                logger.warning("No Claims resources found in extracted data.")
        except Exception as e:
            logger.error(f"Claims ETL failed: {e}")
            raise
    else:
        logger.warning("No resources extracted at all – check HAPI server.")

spark.stop()
logger.info("ETL finished.")
