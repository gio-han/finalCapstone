""" This is a Python program/script that performs
sentiment analysis on a dataset of product reviews. """


# Importing the spaCy natural language processing library.
import spacy

# Importing the pipeline component 'spacttextblob',
# which enables sentiment analysis using the 'TextBlob' library.
from spacytextblob.spacytextblob import SpacyTextBlob

# Importing the pandas library for data manipulation and analysis.
import pandas as pd


# Loading the small spaCy model and assigning it to a variable.
nlp = spacy.load('en_core_web_sm')

# Adding the spacytextblob pipeline component.
nlp.add_pipe('spacytextblob')


def read_csv_with_error_handling(filename, encoding='utf-8'):
    """ A function to read in a CSV file and handle any potential errors. """
    try:
        df = pd.read_csv('amazon_product_reviews.csv', encoding=encoding)
        return df
    except FileNotFoundError:
        print(f'\nError: The file "{filename}" was not found.\n')
        return None
    except Exception as e:
        print(f'\nAn error has occurred while reading the CSV file: {e}\n')
        return None


# Loading the dataset by calling the user-defined function.
dataframe = read_csv_with_error_handling('amazon_product_reviews.csv', 'utf-8')


"""
# Getting descriptions of the dataset for the report/summary.
print(f'\n{dataframe.info()}')
print(f'\n{dataframe.isna().sum()}\n')
"""


# Selecting the 'reviews.text' column and removing all missing values from it.
# Removing missing values wouldn't really be needed as we know there are none.
reviews_data = dataframe['reviews.text'].dropna()


def preprocessing(text):
    """ A function to preprocess single text reviews."""
    doc = nlp(text)
    # Removing stop words, punctuation and whitespace &
    # lemmatizing as well as converting to lowercase and string.
    tokens = [str(token.lemma_.lower()) for token in doc if not token.is_stop
              and not token.is_punct and not token.is_space]
    return ' '.join(tokens)


def sentiment_analysis(review):
    """ A function that takes a product review
    as input and predicts its sentiment. """
    # Tokenizing the review.
    doc = nlp(review)
    # Using the polarity attribute and rounding the output to 3 decimal points.
    polarity = round(doc._.blob.polarity, 3)
    # Using the subjectivity attribute and rounding the output to 3 decimal points.
    subjectivity = round(doc._.blob.subjectivity, 3)
    return f'Sentiment: Polarity = {polarity} & Subjectivity = {subjectivity}'
    # Used 'polarity' and 'subjectivity' above instead of 'sentiment' as
    # 'sentiment' inherently provides both scores but not in a user-friendly form.


# Testing the model on a sample of product reviews.
# 'random_state' ensures the reproducibility of the test samples.
sample_of_reviews = reviews_data.sample(25, random_state=99)

# Printing out the results of the test using a for loop.
for review in sample_of_reviews:
    print(f'\n{review}')
    print(sentiment_analysis(preprocessing(review)))
print() # Statement used for a prettier output.


""" Comparing the similarity of two product reviews. """

# Loading the medium model spaCy model as the small one has no word vectors.
nlp = spacy.load('en_core_web_md')

# Choosing two reviews from the data using square bracket indexing.
review_of_choice_1 = reviews_data[263]
review_of_choice_2 = reviews_data[49]

# Tokenizing both reviews and calculating the similarity between them.
similarity = nlp(review_of_choice_1).similarity(nlp(review_of_choice_2))

# Printing the results using an f-docstring.
print(f'''
Review A: "{review_of_choice_1}"
Review B: "{review_of_choice_2}"

The similiarity between reviews A and B is : {similarity:.3f}
''')


# End-of-file (EOF)
