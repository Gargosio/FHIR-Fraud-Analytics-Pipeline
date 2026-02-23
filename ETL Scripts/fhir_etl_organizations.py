import os
import json
import tempfile
import logging
from fhirclient import client
from fhirclient.models.organization import Organization
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
    .appName("FHIR_ETL_to_Doris_Organizations") \
    .master("local[*]") \
    .config("spark.jars", doris_jar_path) \
    .config("spark.driver.extraClassPath", doris_jar_path) \
    .config("spark.executor.extraClassPath", doris_jar_path) \
    .getOrCreate()

logger.info(f"Spark started with JAR: {doris_jar_path}")
logger.info(f"Spark version: {spark.version}")

# ── Main ETL ──
with tempfile.TemporaryDirectory() as temp_dir:
    # Extract Organizations resources
    extracted_count = extract_fhir_resources(Organization, temp_dir)

    if extracted_count > 0 and os.listdir(temp_dir):
        try:
            # Read all JSON files
            df = spark.read.json(os.path.join(temp_dir, "*.json"))

            # Filter only Organizations resources
            organizations_df = df.filter(col("resourceType") == "Organization")

            if organizations_df.count() > 0:
                logger.info("Raw Organizations schema:")
                organizations_df.printSchema()

                # ── Transformation: only SSN identifier ──
                transformed_organizations_df = organizations_df.select(
                    col("id"),
                    col("meta.lastUpdated").alias("last_updated"),
                    col("active"),
                    explode_outer("type").alias("type"),
                    col("name").alias("name"),
                    explode_outer("telecom").alias("telecom"),
                    col("address")[0].line[0].alias("address"),
                    col("address")[0].city.alias("city"),
                    col("address")[0].state.alias("state"),
                    col("address")[0].postalCode.alias("postalCode"),
                    col("address")[0].country.alias("country")
                    ).select(
                    col("id"),
                    col("last_updated"),
                    col("active"),
                    col("type.coding")[0].display.alias("type"),
                    col("name"),
                    col("telecom.value").alias("phone"),
                    col("address"),
                    col("city"),
                    col("state"),
                    col("postalCode"),
                    col("country")                  
                )#.filter(
                #    col("id").isNotNull()
                #)

                # Preview
                transformed_organizations_df.show(10, truncate=False)

                # Write to Doris – simple SSN table
                transformed_organizations_df.write \
                    .format("doris") \
                    .option("doris.fenodes", "127.0.0.1:8030") \
                    .option("doris.table.identifier", "fhir_db.organizations") \
                    .option("user", "root") \
                    .option("password", "") \
                    .mode("append") \
                    .save()

                logger.info("Organizations data successfully written to Doris!")
            else:
                logger.warning("No Organizations resources found in extracted data.")
        except Exception as e:
            logger.error(f"Organizations ETL failed: {e}")
            raise
    else:
        logger.warning("No resources extracted at all – check HAPI server.")

spark.stop()
logger.info("ETL finished.")
