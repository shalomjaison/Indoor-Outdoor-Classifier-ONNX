import torchvision.transforms.v2 as T
from PIL import Image
import torch
import os
import onnxruntime as ort
import numpy as np
from torchvision import transforms as trn
from iodetector import load_labels

class indoorOutdoorProcessing:
    def __init__(self):
        """Resize, Convert to Tensor and then Normalize with standard deviation and average of pixels"""
        self.transform = trn.Compose([
            trn.Resize((224, 224)),  # Resize to match ONNX model input
            trn.ToTensor(),  # Convert image to tensor first (needed for normalization)
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
        ])
        self.classes, self.labels_IO, _, _ = load_labels()
        self.img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    
    def find_images_in_dir(self, directory):
        """Find all valid images in a directory."""
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) 
            and f.lower().endswith(self.valid_extensions)
        ]

    def preprocess_single(self, img_path):
        """Loads and preprocesses an image for ONNX model"""
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img).numpy() # ONNX models only accept Numpy
        img = np.expand_dims(img, axis=0) # Numpy equivalent of unsqueeze(0), adds batch dimension
        return img
    
    
    def postprocess_single(self, model_output):
        """Postprocessing ONNX MODEL OUTPUT to determine Indoor/Outdoor and scene categories."""
        probs = np.exp(model_output[0]) / np.sum(np.exp(model_output[0]))  # Apply softmax manually in NumPy
        sorted_indices = np.argsort(probs)[::-1]  # Sort in descending order

        io_img = np.average(self.labels_IO[sorted_indices[:10]], weights=probs[sorted_indices[:10]])
        environment = "Indoor" if io_img < 0.5 else "Outdoor"

        scene_preds = [
            {"Description": self.classes[idx], "Confidence": round(probs[idx], 4)}
            for idx in sorted_indices[:5]
        ] # Similar to scene in iodetector.py

        return {"Environment Type": environment, "Scene Category": scene_preds}
    
    # Below methods are not necessary

    # def preprocess_dir(self, dir_path):
    #     """Loads and preprocesses all images in the directory for an ONNX model"""
    #     img_paths = self.find_images_in_dir(dir_path)
    #     return [self.preprocess_single(path) for path in img_paths]

    # def postprocess_multiple(self, model_outputs):
    #     return [self.postprocess_single(output) for output in model_outputs]


class indoorOutdoorModel:
    def __init__(self, model_path):
        self.iop = indoorOutdoorProcessing()
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    
    def predict(self, img_path):
        preprocessed_img = self.iop.preprocess_single(img_path)
        model_output = self.session.run(None, {"input": preprocessed_img})
        postprocessed_output = self.iop.postprocess_single(model_output)

        postprocessed_output["image_path"] = img_path
        return postprocessed_output
    
    def predict_dir(self, input_dir):
        results = []
        for img_path in self.iop.find_images_in_dir(input_dir):
            results.append(self.predict(img_path))
        
        return results

