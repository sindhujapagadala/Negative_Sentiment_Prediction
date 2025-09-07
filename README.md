# Negative Sentiment Prediction

This project is a machine learning application that predicts negative sentiment in text data. It uses natural language processing (NLP) techniques to classify whether a given text expresses negative sentiment.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Evaluation](#evaluation)

## Usage  


import pandas as pd  


## Overview
Negative Sentiment Prediction is designed to analyze text and identify if the sentiment is negative. This can be useful for applications like social media monitoring, customer feedback analysis, and opinion mining.

## Dataset
The dataset used in this project contains textual data labeled with sentiment. The dataset is preprocessed to clean the text and prepare it for model training.

## Features
- Text preprocessing (lowercasing, punctuation removal, tokenization)
- Vectorization using `TextVectorization`
- Multi-label classification using neural networks
- Model training and evaluation
- Prediction on new text data

## Installation
Clone the repository:
   
git clone https://github.com/sindhujapagadala/Negative_Sentiment_Prediction.git

Install required packages:

pip install -r requirements.txt

Usage
1.Load the dataset and preprocess the text.

2.Train the model:

model.fit(train_dataset, validation_data=val_dataset, epochs=5)

3.Predict sentiment for new text:

predictions = model.predict(new_text_vectorized)

Model:
The project uses a Bidirectional LSTM neural network with dense layers. The model architecture includes:
Embedding layer
Bidirectional LSTM
Dense layers with ReLU activation
Output layer with sigmoid activation for multi-label classification

Evaluation:
The model is evaluated using metrics such as:
Accuracy
Precision
Recall
F1-score


