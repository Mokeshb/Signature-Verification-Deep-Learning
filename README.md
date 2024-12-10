Signature Verification Using Deep Learning and Digital Forensics

Introduction:
In today's digital landscape, signature verification is a critical process for validating the authenticity of individuals in sectors such as banking, legal documentation, and financial transactions. This project addresses the challenges of detecting forged signatures by combining deep learning and digital forensics.
Objectives:
The primary objectives of this project are:
Develop a robust system to classify signatures as genuine or forged.
Utilize Convolutional Neural Networks (CNNs) to extract and analyze signature features.
Integrate digital forensic tools, such as The Sleuth Kit (TSK), to ensure the integrity of the dataset.
Scope
Included:
Deep learning-based signature classification.
Forensic validation of datasets using TSK.
Analysis of training and testing performance.
Excluded:
Real-time signature capture systems.
Handwritten live data input.
Multi-language or symbolic signature datasets.
Methodology

Data Preparation:
The dataset consists of genuine and forged signature samples sourced from Kaggle.
Preprocessing includes resizing images, normalizing pixel values, and labeling.
Deep Learning Model:
A Convolutional Neural Network (CNN) with the following architecture:
Input Layer: Accepts grayscale signature images.
Convolutional and Pooling Layers: Extract features such as edges and shapes.
Dense Layers: Classify signatures as genuine or forged.
Optimizer: RMSProp.
Loss Function: Binary Cross-Entropy.
Forensic Validation:
The Sleuth Kit (TSK) was used to verify dataset integrity by checking for tampering or unauthorized modifications.
Model Training and Evaluation:
Training on 80% of the dataset and validation on 20%.
Testing performed on unseen samples to evaluate accuracy.
Results
The trained model achieved the following metrics:
Training Accuracy: 98%
Validation Accuracy: 85%
Test Accuracy: 88.57%
The CNN demonstrated high reliability in distinguishing genuine from forged signatures, though further optimization is required to improve real-time performance.
Challenges
Dataset Imbalance:
The dataset had fewer forged samples compared to genuine ones.
Mitigated using data augmentation techniques.
Computational Requirements:
Training the CNN required substantial time on CPU.
Leveraged Google Colab with GPU acceleration to reduce training time.
Forensic Integration:
Ensuring TSK compatibility with the Python environment posed initial challenges.
Future Work
Incorporate real-time signature capture and verification.
Enhance the dataset with diverse samples, including multilingual signatures.
Optimize the CNN model for faster inference and deployment in live systems.
Link to Tool:
The project's code, documentation, and dataset preprocessing scripts are available on GitHub.
Installation and Usage Manual
System Requirements
Hardware: At least 8GB RAM; GPU recommended.
Software:
Python 3.7 or higher
TensorFlow, Keras, NumPy, Matplotlib

Installation Steps:
Clone the repository:
git clone https://github.com/Mokeshb/Signature-Verification-Deep-Learning.git
cd Signature-Verification-Deep-Learning
Install dependencies:
pip install -r requirements.txt

Usage Instruc:
Place the Train and Test folders in the project root.
Run the Untitled13 (1).ipynb notebook in Google Colab or Jupyter Notebook.
Follow the notebook's instructions for preprocessing, training, and evaluation.

Troubleshooting:
Missing Dependencies: Ensure all libraries are installed using requirements.txt.
Dataset Issues: Verify the dataset folder structure.
GPU Errors: Confirm TensorFlow recognizes your GPU.

Conclusion:
This project successfully demonstrates the integration of deep learning and digital forensics for signature verification. It provides a solid foundation for further research and development in fraud detection and document authentication systems.
