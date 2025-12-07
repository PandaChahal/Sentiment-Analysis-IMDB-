# Sentiment Analysis ( IMDb Reviews ) 
This project is a deep learning–based sentiment classification system built using the IMDB Movie Reviews dataset.
The goal is to automatically determine whether a movie review expresses a positive or negative sentiment.

Using an LSTM (Long Short-Term Memory) neural network, the model learns patterns in natural language and predicts sentiments with strong accuracy.
The entire workflow — from text preprocessing to training and evaluation — is implemented in Python using TensorFlow/Keras.

## Project Overview

This project demonstrates a practical approach to NLP-based sentiment analysis.

**It includes:**
- Preparing and cleaning raw movie review text
- Converting text into numerical sequences
- Training an LSTM network to classify reviews
- Evaluating accuracy and performance
- Predicting sentiment for new, custom-written reviews

**Sentiment analysis of movie reviews is widely used in:**
- Recommendation systems
- Customer feedback analytics
- Social media understanding
- NLP model benchmarking and research

This project delivers a simple, effective, and end-to-end pipeline for binary text sentiment classification.

## Dataset

The dataset is a well-known benchmark for sentiment analysis, containing:
- 25,000 labeled training reviews
- 25,000 labeled test reviews

Due to its large size, the dataset has been uploaded on Google Drive.

Here is the link to download the Dataset: 🔗[IMDb Dataset](https://drive.google.com/file/d/1JMkLOWiqXGzZiuRcdJsVyQ5Yr1a_vwqB/view?usp=sharing)

**Dataset Structure :**
```perl
IMDB-Dataset/
├── train/
│   ├── pos/
│   └── neg/
└── test/
    ├── pos/
    └── neg/
```
Each review is a raw text file containing a single movie review.

## Content Overview
Apart form the dataset used this project contains two Jupyter notebooks. 

- The imdbSentimentAnalysis.ipynb notebook is the main workflow, covering dataset loading, text preprocessing, tokenization, model training, evaluation, and saving the final LSTM model along with the tokenizer. It serves as the complete end-to-end implementation of the sentiment analysis pipeline.
  
- imdbLSTM.ipynb, is used for experimentation—testing different LSTM architectures, tweaking hyperparameters, trying alternate embedding sizes, and running quick validation tests. It allows you to explore model variations without altering the main training workflow.
  
## Technologies Used 

- Python
- Google Colab / Kaggle GPU (for accelerated training)
- TensorFlow
- Keras (Sequential API)
- NLTK
- Keras Tokenizer & Pad Sequences
- NumPy
- Pandas
- Matplotlib

## Model Architecture 

            ┌─────────────────────┐
            │   Input Review      │
            │ (Tokenized + Padded)│
            └─────────┬───────────┘
                      ▼
            ┌──────────────────────┐
            │   Embedding Layer    │
            │ (Word Vector Mapping)│
            └─────────┬────────────┘
                      ▼
            ┌────────────────────┐
            │     LSTM Layer     │
            │ (Sequence Learning)│
            └─────────┬──────────┘
                      ▼
            ┌────────────────────┐
            │   Dropout Layer    │
            │ (Regularization)   │
            └─────────┬──────────┘
                      ▼
            ┌────────────────────┐
            │   Dense Layer      │
            │ (Sigmoid Output)   │
            └─────────┬──────────┘
                      ▼
            ┌─────────────────────┐
            │  Sentiment Result   │
            │ (Positive/Negative) │
            └─────────────────────┘

## Model Evaluation

The performance of the model is evaluated using these key metrics:

**> Accuracy**
(Measures the overall percentage of correct predictions made by the model)

**> Confusion matrix**
(A matrix that shows how many predictions were correct or incorrect for each class)

**> F1 Score**
(The harmonic mean of precision and recall, giving a balanced measure of a model’s performance)

## To run :
- Clone the repository and open the project folder.
- Download the IMDB dataset from the provided Google Drive link and place it in the project directory.
- Install the required libraries.
- Open the notebook imdbSentimentAnalysis.ipynb in Jupyter/Colab/Kaggle.
- Run all cells to preprocess data, train the LSTM model, and evaluate performance.
- The notebook will save the trained model and tokenizer automatically.
- Use the final section of the notebook to test the model on custom text inputs.

## Conclusion

This project was completed as part of our Deep Learning and Applications (UEC642) coursework, undertaken by our group under the guidance of Dr. Gaganpreet Kaur. 

Through this project, we implemented an end-to-end sentiment analysis model using LSTM networks, gaining hands-on experience with text preprocessing, sequence modeling, and neural network training. The final model achieved strong performance, demonstrating that LSTM-based architectures are effective for binary sentiment classification tasks such as IMDB movie reviews. Overall, this project enhanced our practical understanding of natural language processing and deep learning, while providing valuable exposure to real-world model development and evaluation.

## Contributors

This work was carried out by the following team members :
- Bhamya Gupta
- Hashandeep Kaur
- Sanya Arora
- Sharanya Sharma 
