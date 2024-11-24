# Image Captioning using VGG16 and LSTM

## Description
This project implements an image captioning model that generates descriptive captions for images using a combination of VGG16 and Long Short-Term Memory (LSTM) networks. The model uses the pre-trained VGG16 network for feature extraction from images and employs an LSTM for sequence modeling to generate human-readable captions.

The dataset used in this project is the Mini COCO 2014 dataset, which contains 18.7K images with 5 captions per image. The goal of this project is to train a deep learning model that can automatically generate captions for images, which is useful in applications such as accessibility tools, automated content generation, and image search.

## Dataset
- **Name**: Mini COCO 2014 dataset
- **Size**: 18.7K images with 5 captions per image
- **Source**: [Mini COCO 2014 dataset for image captioning on Kaggle](https://www.kaggle.com/datasets/nagasai524/mini-coco2014-dataset-for-image-captioning)

The dataset contains a diverse collection of everyday images annotated with captions describing the content.

## Project Structure
```
Image-Captioning-VGG16-LSTM/
│
├── Image_Captioning.ipynb      # Main Jupyter notebook for the image captioning model
├── requirements.txt           # Python dependencies required to run the project
└── README.md                  # Project description and instructions
```
- **Image_Captioning.ipynb**: The Jupyter notebook containing the code for building and training the image captioning model, using VGG16 for feature extraction and LSTM for generating captions.

- **requirements.txt**: A list of required libraries to run the project, such as TensorFlow, Keras, Numpy, etc.

- **README.md**: A markdown file that provides an overview of the project, including the purpose, dataset used, and how to run the model.

