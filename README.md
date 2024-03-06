# Disorder Analysis Classification
This repository contains code for a Disorder Analysis Classification Machine Learning project. The goal of this project is to develop a model that can classify the disorder of a person based on textual input into one of several categories such as Anger/ Intermittent Explosive Disorder,Anxiety Disorder,Depression, Narcissistic Disorder,Panic Disorder.

Understanding the sentiment or mood behind text data is a crucial task in Natural Language Processing (NLP). This project focuses on building a machine learning model that can effectively classify the disorder expressed in textual input. The model can have various applications such as sentiment analysis for customer reviews, social media sentiment analysis, etc.

Dataset
The dataset used for training and evaluation is a collection of textual data labeled with different disorders. The dataset is not included in this repository due to licensing reasons. However, you can use any dataset of your choice or collect your own data for training the model.

Model Architecture
The model architecture used in this project are SVM(Support Vector Machines),MultilayerPerceptron and Decision tree to identify the best possible model for this task.

Dependencies
Python 3.x
TensorFlow
NumPy
Pandas
scikit-learn
nltk

You can install these dependencies using pip:

```python
pip install pandas
pip install scikit-learn
pip install nltk
pip install matplotlib
```
Additionally, you might need to download NLTK resources such as stopwords and WordNet. You can do this by running the following Python code after installing NLTK:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```
Usage
To train the model, you need to provide the dataset in the appropriate format. Once the dataset is prepared, you can use the provided scripts to preprocess the data, train the model, and evaluate its performance. Here's a general outline of the steps:

Prepare Dataset: Prepare your dataset and preprocess it if necessary.
Train Model: Train the mood classification model using the prepared dataset.
Evaluate Model: Evaluate the performance of the trained model using test data.
Predictions: Make predictions on new textual data to classify the mood or disorder.
