import json
import os
import shutil
import warnings
from typing import List, TypedDict


# Suppress warnings
warnings.filterwarnings("ignore")

import onnxruntime as ort
# Import Flask-ML and other components for creating an ML server
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    BatchFileInput,
    FileInput,
    FileResponse,
    FileType,
    InputSchema,
    InputType,
    NewFileInputType,
    ResponseBody,
    TaskSchema,
)

from IndoorOutdoorClassifier.onnx_helper import indoorOutdoorModel

# Initialize Flask-ML server
server = MLServer(__name__)

# Add application metadata
server.add_app_metadata(
    name="IndoorOutdoorClassifier",
    author="Shalom",
    version="0.3.8",
    info=load_file_as_string("./README.md"),
)


# Define schema for image processing task inputs and outputs
def image_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="image_input", label="Upload Images", input_type=InputType.BATCHFILE
            ),
            InputSchema(
                key="output_path",
                label="Output JSON Path",
                input_type=NewFileInputType(
                    default_name="output.json",
                    default_extension=".json",
                    allowed_extensions=[".json"],
                ),
            ),
        ],
        parameters=[],
    )


# Define types for the image inputs and parameters
class ImageInputs(TypedDict):
    image_input: BatchFileInput
    output_path: FileInput


class ImageParameters(TypedDict):
    pass


# Function to append data to an existing JSON file, or create it if it doesn't exist
def append_to_json(file_path, data):
    if os.path.exists(file_path):
        with open(file_path, "r+") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
            if isinstance(existing_data, list):
                existing_data.append(data)
            else:
                existing_data = [existing_data, data]
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    else:
        with open(file_path, "w") as file:
            json.dump([data], file, indent=4, ensure_ascii=False)


model_path = "indoor_outdoor.onnx"
indoor_outdoor_classifier = indoorOutdoorModel(model_path)

# Define route for processing images
@server.route(
    "/process_images", task_schema_func=image_task_schema, short_title="IndoorOutdoor Result"
)
def process_images(inputs: ImageInputs, parameters: ImageParameters) -> ResponseBody:
    results = []  # Store results for each processed image
    output_path = inputs["output_path"].path

    # Ensure output path is clean
    if os.path.exists(output_path):
        os.remove(output_path)

    for img_file in inputs["image_input"].files:
        print(f"Processing image: {img_file.path}")
        # Run Indoor/Outdoor detector
        print("Predicting Indoor/Outdoor and Scene Type")
        io_result = indoor_outdoor_classifier.predict(img_file.path)
        print(f"IO Detection Result: {io_result}")
        results.append(io_result)

    # Write results to the output JSON file
    try:
        append_to_json(output_path, results)
        print(f"Results written to: {output_path}")
    except Exception as e:
        print(f"Failed to write results: {e}")

    # Return the output JSON file to the user as a download
    return ResponseBody(
        root=FileResponse(
            file_type=FileType.JSON,
            path=output_path,
            title="Report",
        )
    )


# Run the server if this script is executed directly
if __name__ == "__main__":
    server.run()
