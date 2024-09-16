The objective is to take a sentence,  slice it into individual text and produce a label that summarizes the sentiment of this text, e.g., positive and negative. 

Classify the tweets of Elon musk from 2010-2022 and whether they had a positive or negative impact

1) importing necessary libraries and downloading stopwords Corpus
 
2) downloading all tweets of Elon musk from 2010-2022 and creating a dataset using that

3) fetch the sentiment type of every tweet to classify it as a positive or a negative tweet

4) removing '@names' from tweets and keeping cleaned tweets in a new column called tidy-tweets

5) removing links like 'HTTP/HTTPS from tidy-tweets

6) Removing tweets with empty text and dropping duplicates rows 

7) Remove punctuation, numbers, and special characters and keep them in a new column, 'absolute_tidy_tweets.'

8)Tokenize the tweets in absolute_tidy_tweets. Convert words to lemma and join all tokens to sentences.

9) preprocess the phrase terms and set grammatical rules to identify the phrase

10) convert textual representation in the form of Numeric Features

11) predict using a naive_classifier and find the accuracy score

Developed a model based on a naive Bayes classifier which predicts whether the tweets have positive or negative impacts based on keywords and critical phrases extracted from the tweets.

Fetching the sentiment using NLTK's SIA

Converting words to lemma using wordnet lemmatizer

Use phrase extract helper in preprocessing phrase terms.

Feature extraction TF-IDF (Term Frequency - Inverse Document Frequency)

After preprocessing all the tweets of Elon Musk from 2010-2022, Tokenizing and lemmatization for keyword-extract and key-phrase extract, predict whether it has a positive or negative impact.