# Indoor Outdoor Classifier

## Overview
The **Indoor-Outdoor Classifier ONNX** application is designed to classify whether an image represents an **indoor** or **outdoor** scene using an optimized ONNX model. This standalone module was extracted from the **GeoLocator** project and focuses solely on environment classification.

## Features
**Indoor/Outdoor Scene Recognition**: Classifies images into `Indoor` or `Outdoor` categories
**ONNX Model for Fast Inference**: Uses an ONNX model for fast inference and interoperability

## Installation and Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shalomjaison/Indoor-Outdoor-Classifier-ONNX.git
cd Indoor-Outdoor-Classifier-ONNX
```

### Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate  # For Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Running the Server
To start the Flask-ML server:
```bash
python flaskml-server.py
```
The server will run on http://127.0.0.1:5000 by default.

### Using the Frontend (RescueBox)
- Open the RescueBox interface.
- Register the model with the server's IP address (127.0.0.1) and port (5000).
- Upload images to the "GeoLocator" model.
- Provide an output JSON file path (e.g., /Users/username/Desktop/output.json).
- Click "Run Model" to process the images and retrieve the results.

### Output
The output is a JSON file containing:
- Environment Type: Whether it was taken indoor/outdoor
- Scene Categories: Top 5 possible scenes it could be each with a confidence level
- Image Path: The Path of the current image the analysis was conducted upon

#### Detailed Results Explanation
**1. Indoor/Outdoor Classification**
- Accuracy: 0.90

The model performs exceptionally well at classifying whether an image was taken indoors or outdoors, achieving 90% accuracy.

**2. Scene Detection**
- Precision: 0.37
- Recall: 0.64
- F1-Score: 0.45
- Accuracy: 0.64
  
The precision for scene detection is relatively low (0.37), suggesting the model has difficulty correctly identifying specific scenes. However, the recall is higher at 0.64, meaning it is able to find most correct scenes but with some incorrect predictions.



**Conclusion**
- Indoor/Outdoor Accuracy: **90%**

### Phase 1: Indoor Outdoor Classification
The dataset used is Places 365 (365 categories), out of which Indoor → 160 categories and Outdoor → 205 categories.
To test 

    !cd IndoorOutdoorClassifier
    !python
    iodetector.test_iodetector()


## Citation
If you find our work useful in your research, please consider citing:  

    @article{mmocr2021,
    title={MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding},
    author={Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin, Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang, Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua},
    journal= {arXiv preprint arXiv:2108.06543},
    year={2021}
    }

    @inproceedings{10.5555/3666122.3666501,
    author = {Cepeda, Vicente Vivanco and Nayak, Gaurav Kumar and Shah, Mubarak},
    title = {GeoCLIP: clip-inspired alignment between locations and images for effective worldwide geo-localization},
    year = {2024},
    publisher = {Curran Associates Inc.},
    address = {Red Hook, NY, USA},
    booktitle = {Proceedings of the 37th International Conference on Neural Information Processing Systems},
    articleno = {379},
    numpages = {12},
    location = {New Orleans, LA, USA},
    series = {NIPS '23}
    }

    @inproceedings{baek2019character,
    title={Character Region Awareness for Text Detection},
    author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={9365--9374},
    year={2019}
    }

    @article{zhou2017places,
    title={Places: A 10 million Image Database for Scene Recognition},
    author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2017},
    publisher={IEEE}
    }

