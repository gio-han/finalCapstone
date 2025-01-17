# Capstone Project - Natural Language Processing (NLP) Applications


## Table of Contents

* Project Description
* Installation
* Usage


## Project Description

The Python program in this repository performs sentiment analysis on a dataset of product reviews.

The dataset used by the program is the CSV file named 'amazon_product_reviews.csv' in the same repository.

The program contains three functions:
* A function that reads CSV files and includes proper error handling.
* A function that preprocesses single text reviews.
* A function hat takes a product review as input and predicts its sentiment.

In the Python script, the model is then tested on a sample of product reviews.
Lastly, the similarity of two product reviews is compared.

The repository also contains a brief report/summary in PDF format, which includes the following:
* A description of the dataset used.
* Details of the preprocessing steps.
* An evaluation of the results.
* Insights into the model's strengths and limitations.


## Installation

Ensure you have the following packages installed:
* spaCy, including the small (en_core_web_sm) and medium (en_core_web_md) English language models
* spacytextblob, including additional data 'corpora' (python -m textblob.download_corpora)
* pandas


## Usage

Once you have downloaded the PY as well as the CSV files, and installed all the necessary packages, you are ready to run the code.

Below is a screenshot of part of the output of the code.

Note:
* Polarity scores range from -1 to 1, where -1 is negative, 0 is neutral and 1 is positive.
* Subjectivity scores range from 0 to 1, where 0 is objective (factual) and 1 is subjective (opinionated).

![A screenshot of part of the output of the code](https://github.com/gio-han/finalCapstone/assets/151397333/647c5a22-6fe6-4873-9c2d-d74544eba339)
