PROBLEM STATEMENT

Elon musk took the world by storm when he bought Twitter. Ever since, his decision has been met with mixed reactions across the globe.
Certain decisions such as handing out verifications for a mere $8 has made finding authentic tweets get lost in parody.
Nevertheless, you as a data scientist, must persevere into this storm, to understand what sentiment his tweets wish to convey in the wake of all of this. 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

My approach

Seeing as we need tweets, the approach I chose was to perform webscraping using python to obtain a csv file of tweets.
After that, I shall perform preprocessing to remove redundant values, garbage values and club similar words together using Porter stemming. 
Finally, I shall train three distinct models and compare their accuracy by testing it out on a test dataset that is also acquired through webscraping


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Core Algorithm/Logic

The Twitter Scraper

Here, I wished to make the scraping as dynamic as the user might want it to be by giving users the option to: 

	- Choose a start and end date for tweets
	- Choose a topic to scrape tweets for from the feed bar
	- Choose a particular user for whom we shall scrape tweets from
	- Number of Tweets to scrape

For the purposes of this demo, I shall take about 1000 tweets of Elon musk from recent months for the training set and 100 tweets of Elon musk for the testing set. 
Sentiment labeling can be done either manually or by relying on the positive, negative keywords in the datasets. 

Splitting the dataset can be achieved by going through the assigned labels and creating a .pos and .neg file based for the positive and negative tweets respectively. 

(I know, I know, twitter can be positive too)

There are several custom functions I defined that I used throughout the project. 

Now, perhaps one of the most easiest analysis to perform on the dataset is to extract the vocabulary and rank them in terms of most used to least used. 
This is exactly what I did in my project by parsing the file, taking each line sepearately and isolating the words. 
After that two csv files were created namely, vocab and vocab_inv which serve as a map of the words in the dataset ranked from most used to least. 

Now, any model requires clean data to perform properly. (even if we don't have washing machines, cleaning data becomes essential for succcess)
This was a rather straightforward process of stripping the excess characters, special characters, emojis, mentions, hashtags. 
Further preprocessing involved maintaining a uniform case and removing redundnat characters/spaces. 
This was followed by making use of the porter stemmer algorithm to stem words that are similar by correlating their suffixes together. 


This preproccessing was performed for both the training and testing dataset to acquire a clean set of tweets. (Finally, we become men of substance)

My gut instinct was to use the ngram model, which I had previously used in my AIML course project. 
However, I wished to go down a different route this time around. 
My learnings had helped me understand the importance of unigrams (fancy way to say unique words) and bigrams. 
I made an algorithm that would provide users with a comprehensive statistical analysis of the dataset. 

By now, we would not only have preprocessed datasets, but also pickled files of the frequency distribution for unigrams and bigrams.

I tested three different models for the purposes of this project. 

 	- Logistical Regression in the form of Maximum Entropy Model
	- SVM
	- Decision Tree

(Because of course, what better way to study tweets of Elon Musk than having LSD in your system)

Each of these models can be trained by keeping the TRAIN flag at True and then use the model generated for the testing data by turning it to False. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Conclusion

All of this was built upon Tensorflow, numpy, and scipy.
The accuracies for the models were 63.5% (Maximum Entropy), 66.6%(Decision Tree), and 72.5% (SVM). 
The accuracies can be improved by tweaking the bigrams and also making it a larger dataset. 
Future development can focus on building a word cloud for the vocabulary.
