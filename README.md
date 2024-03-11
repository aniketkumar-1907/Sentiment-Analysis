# Sentiment-Analysis

This code uses a pre-trained BERT model for sentiment analysis to predict the sentiment scores of Yelp reviews obtained from a specific URL. The results are stored in a DataFrame for further analysis.


# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
Here, you're importing the required libraries, including the transformers library for using the BERT model, and other libraries for web scraping (requests and BeautifulSoup) as well as for data manipulation (pandas and numpy).

# Load pre-trained BERT model and tokenizer for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
This part of the code initializes a pre-trained BERT model (bert-base-multilingual-uncased-sentiment) along with its tokenizer for sentiment analysis.

# Tokenize a sample text and obtain sentiment prediction
tokens = tokenizer.encode('I loved this, absolutely the best', return_tensors='pt')
result = model(tokens)
predicted_sentiment = int(torch.argmax(result.logits)) + 1
print(predicted_sentiment)
Here, you're testing the model by encoding a sample text and obtaining its sentiment prediction. The sentiment scores are typically in the range [1, 5].

# Scraping Yelp reviews from a specific URL
r = requests.get('https://www.yelp.com/biz/harris-restaurant-san-francisco')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class': regex})
reviews = [result.text for result in results]
This part of the code uses the requests library to get the HTML content of a Yelp page and BeautifulSoup to parse the HTML. It then extracts reviews based on a specific class pattern.

# Create a DataFrame from the extracted reviews
df = pd.DataFrame(np.array(reviews), columns=['review'])
You're creating a DataFrame (df) from the extracted reviews using pandas, which can be useful for further analysis.

# Define a function to get sentiment scores for each review
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1
Here, you define a function (sentiment_score) that takes a review, tokenizes it, and uses the BERT model to predict its sentiment score.

# Apply the sentiment_score function to each review in the DataFrame
df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
This code applies the sentiment_score function to each review in the DataFrame, storing the sentiment scores in a new column called 'sentiment'.

<img width="1440" alt="Screenshot 2024-03-11 at 11 25 08 AM" src="https://github.com/aniketkumar-1907/Sentiment-Analysis/assets/97777060/861c0f3c-020e-4d1f-a9b8-0dc469b59f5b">
