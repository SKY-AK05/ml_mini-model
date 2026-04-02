from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment, AmlCompute
from azure.ai.ml.constants import AssetTypes
from azure.identity import InteractiveBrowserCredential


SUBSCRIPTION_ID = "e2d35239-5448-45ae-b156-d3309a9052a9"
RESOURCE_GROUP  = "Aakash_ML"
WORKSPACE_NAME  = "Aakash_ML"
COMPUTE_NAME = "cpu-testing"


ml_client = MLClient(
    InteractiveBrowserCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME
)


# Define a custom environment from your requirements.txt
custom_env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest", # Standard CPU base image
    conda_file="c:/Users/Aakash/Documents/PROJECT WEB/orchvate/ML/Mini model/configs/conda.yaml",
    name="lp-mini-cpu-env",
    description="Custom environment for CPU training"
)

# ─────────────────────────────────────────
# SUBMIT JOB
# ─────────────────────────────────────────
job = command(
    code="./src",                          # ONLY upload the code
    command=(
        "python train.py "
        "--data_path ${{inputs.data}} "
        "--epochs 2"                       # Only 2 epochs for CPU test!
    ),
    inputs={
        "data": Input(
            type=AssetTypes.URI_FOLDER, 
            path="tagging_project_data:1" # Use your existing cloud asset
        )
    },
    environment=custom_env,
    compute=COMPUTE_NAME,
    display_name="lp-cpu-test-run",
    experiment_name="lp-mini-model"
)


print("Submitting job...")
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted! Studio URL: {returned_job.studio_url}")
print(f"Status : {returned_job.status}")
print(f"Studio : {returned_job.studio_url}")
