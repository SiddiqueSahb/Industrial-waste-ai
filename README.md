# Industrial Waste Classification

Industrial Waste Classification is a computer vision and deep learning project designed to classify industrial waste images into 28 different WaRP (Waste Recycling Project) categories. The project focuses on building an intelligent and scalable waste management solution that can automate waste segregation and improve recycling efficiency using AI.

The system combines image preprocessing, object detection, classification models, explainability techniques, and visualization tools to create a complete industrial waste analysis pipeline.

---

## Project Objectives

* Automate industrial waste classification using AI
* Improve recycling and waste segregation processes
* Reduce manual effort in waste identification
* Support sustainable and smart waste management systems
* Build an end-to-end computer vision pipeline for industrial applications

---

## Features

* Industrial waste image classification
* Deep learning-based detection and recognition
* Dataset preprocessing and augmentation
* YOLO-based detection pipeline
* Streamlit web application support
* Model training and evaluation scripts
* Visualization and experiment notebooks
* Explainable AI experiments
* Modular and scalable project structure

---

## Tech Stack

* Python
* Deep Learning
* Computer Vision
* YOLOv11
* Vision Transformers (ViT)
* ConvNeXt
* Swin Transformer
* MaxViT
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Streamlit
* Jupyter Notebook
* PyTorch

---

## Project Structure

```bash
Industrial-waste-ai/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   ├── classification_config.yaml
│   └── yolo_dataset.yaml
│
├── notebooks/
│   ├── dataset_analysis.ipynb
│   └── model_experiments.ipynb
│
├── scripts/
│   ├── train_classifier.sh
│   └── train_detector.sh
│
├── src/
│   ├── classification/
│   │   ├── augmentations.py
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   ├── models.py
│   │   └── train.py
│   │
│   ├── config/
│   ├── detection/
│   ├── preprocessing/
│   ├── utils/
│   └── visualization/
│
├── main.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SiddiqueSahb/Industrial-waste-ai.git
```

Navigate to the project directory:

```bash
cd Industrial-waste-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

### Train Classification Model

```bash
python src/classification/train.py
```

### Train Detection Model

```bash
bash scripts/train_detector.sh
```

### Train Classification Pipeline

```bash
bash scripts/train_classifier.sh
```

---

## Dataset

The project uses industrial waste image datasets categorized into 28 WaRP classes for training and evaluation. The dataset is preprocessed and augmented to improve model performance and generalization.

---

## Classification Models and Contributors

| Model | Contributor |
|---|---|
| Vision Transformer (ViT-B/16) | Mohammad Asim Siddique |
| ConvNeXt V1 | Mohammad Arshad Siddique |
| ConvNeXt V2 | Unmesh Pawar |
| Swin Transformer (Swin-T) | Suhaib Ahmed Khan |
| MaxViT | Mohd Yasir Ansari |

---

## Additional Experiments

### A. Per-Class Behaviour and Confusion Analysis
**Contributor:** Mohammad Asim Siddique

* Fine-grained class-level performance analysis
* Confusion matrix analysis
* Per-class F1-score behaviour evaluation

---

### B. Explainability via Occlusion Sensitivity on ViT-B/16
**Contributors:** Mohammad Arshad Siddique, Unmesh Pawar

* Occlusion sensitivity analysis
* Visual attention interpretation for Vision Transformers

---

### C. Detection Extension on WaRP-D Using YOLOv11-m (5 Super-Classes)
**Contributors:** Mohammad Arshad Siddique, Unmesh Pawar, Mohammad Asim Siddique

* Object detection on WaRP-D dataset
* Hierarchical detection pipeline
* Super-class based detection experiments
* Detection-to-classification cascade analysis

---

## Model and Notebook Resources

### Classification Experiment Notebooks

[Classification Notebook Folder](https://drive.google.com/drive/folders/17DMT-zl4z_Hl5N3TIOYRxG-blCWvld8q?usp=share_link)

---

### Detection Model Resources

[Detection Model Folder](https://drive.google.com/drive/folders/1njWOQmICC5T3CKZgopP_u3h0Dtr3IOxg?usp=drive_link)

---

## Applications

* Smart recycling systems
* Automated industrial waste segregation
* Sustainable waste management
* AI-powered recycling plants
* Environmental monitoring systems
* Research in computer vision and sustainability

---

## Future Enhancements

* Real-time detection using webcams
* Cloud deployment support
* Mobile application integration
* Improved model accuracy
* IoT-enabled smart recycling systems
* Waste analytics dashboard
* Real-time industrial monitoring systems

---

## Contributors

| Student Name | Student ID |
|---|---|
| Mohammad Asim Siddique | 6946816 |
| Unmesh Pawar | 6947243 |
| Mohammad Arshad Siddique | 6947016 |
| Suhaib Ahmed Khan | 6948902 |
| Mohd Yasir Ansari | 6950129 |

---

## Repository

[Industrial Waste Classification GitHub Repository](https://github.com/SiddiqueSahb/Industrial-waste-ai.git)

---

## License

This project is developed for academic, research, and educational purposes.