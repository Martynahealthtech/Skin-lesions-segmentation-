# Skin Lesion Segmentation using SegNet
This project uses a SegNet-style convolutional neural network, built with PyTorch, to perform semantic segmentation on skin lesions from the ISIC 2016 dataset.

The main goal is to create a model that can accurately generate a binary mask, separating the lesion from the surrounding healthy skin.

💻 Project Notebook
Skin_lesions_segmentation.ipynb: 

The main Jupyter Notebook containing all steps:

Data loading and preprocessing

Model definition (SegNet)

Training and validation loops

Model evaluation and visualization of predictions

🚀 Setup & Data
To run this project, you must first download the dataset (as it is not stored in GitHub) and then install the required libraries.

1. Dataset
This project uses the ISBI 2016: Skin Lesion Analysis Towards Melanoma Detection dataset.

Download the data from the official ISIC Challenge website: [https://challenge.isic-archive.com/data/]

Unzip the files.

Place the unzipped data folders into the main directory of this project. The final folder structure must look like this:

```
SKIN-LESIONS-SEGMENTATION/
├── Skin_lesions_segmentation.ipynb
├── README.md
├── .gitignore
├── ISBI2016_ISIC_Part1_Training_Data/
├── ISBI2016_ISIC_Part1_Training_GroundTruth/
├── ISBI2016_ISIC_Part1_Test_Data/
└── ISBI2016_ISIC_Part1_Test_GroundTruth/
```
2. Installation
This project was developed in Python. You can install the main dependencies using pip:

Bash

pip install torch torchvision
pip install numpy
pip install matplotlib
pip install opencv-python-headless

🏃‍♀️ How to Run
After completing the setup steps above, open the Skin_lesions_segmentation.ipynb notebook (in VS Code, Jupyter Lab, or Google Colab).

Run all cells from top to bottom to load the data, train the model, and see the results.

🛠️ Technologies Used
Python

PyTorch: For building and training the SegNet model.

NumPy: For numerical operations.

Matplotlib: For plotting images and results.

OpenCV (cv2): For loading and processing images.
