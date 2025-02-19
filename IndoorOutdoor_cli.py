import onnxruntime as ort
import numpy as np
from IndoorOutdoorClassifier.onnx_helper import indoorOutdoorModel
import argparse
from pathlib import Path
import json
from pprint import pprint

# Arguments
parser = argparse.ArgumentParser(description="Run Indoor/Outdoor ONNX Model on Images")
parser.add_argument(
    "--image_folder",
    help="folder to be tested for locations",
    type=str,
    default="testImages",
)
parser.add_argument(
    "--output_csv",
    help="folder to be tested for locations",
    type=str,
    default="op.json",
)
args = parser.parse_args()
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

model = indoorOutdoorModel("indoor_outdoor.onnx")
outputs = model.predict_dir(input_dir)
pprint(outputs)

with open(output_dir / "output.json", "w") as f:
    json.dump(outputs, f, indent=4, ensure_ascii=False)
