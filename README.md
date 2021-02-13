Text classification on movie reviews

I have used NLTK Naïve Bays and Random Forest model in this task and found that random forest has performed better. It has given me 85% accuracy, Precision, Recall and F1-Score. I have also used confusion matrix to see where this model went wrong.
Now let see what all I have done-

Dataset and its Collection:

We will be using movie reviews corpus, which contains 2000 movie reviews. 1000 among them are classified as positive and another 1000 are classified as negative review. 
This corpus is already available in NLTK library and can easily imported by using below command-

from nltk.corpus import movie_reviews

Below are few commands to work with this movie corpus:

movie_reviews.categories() --> Tells you sentiment category ( Positive, Negative )

movie_reviews.fileids('pos') -->  Give you all the fileid containing positive reviews

movie_reviews.words()-->  Tells you all the words used in corpus

movie_reviews.words(fileid) --> Tell you words used in a review under specified fileid

Data Preprocessing:

Reviews in this corpus are not cleaned. It contains many stop words and symbols. We will have to remove them for better model performance. Also words like boy and boys should be treated in the same way. To cater this case we will apply stemming on each word of corpus. Eventually we will shuffle whole dataset of reviews randomly so that when we are dividing them into train and test we have more chances of having pos and neg reviews in equal proportion in both train and test dataset.
Below is the picture of an un-cleaned review.

Data Modelling for NLTK Naïve base classifier

Obviously before feeding data into classifier we will have to format it in a way which classifier can understand. To do that, I have created many variables, but following two are the most important and worth understanding.

Word_Feature:  We will create a list of 8000 most frequent words in the corpus. Obviously, we will not be including stop words and special symbol in it and we will only be considering those words which are greater than 2 in length. This will give us a list of relevant words in the corpus.
Featureset: We will create a feature set for each review using Word_Feature. We will check which all words of Word_Feature exist in review as well and assign True to it if it exist otherwise it will be assigned False. Same process we will repeat for all reviews in our dataset. So eventually the size of Feature set obtained by this process will be N x M (N is total no of reviews and M is length of Word_Feature).

Classification Models:
we will be applying NLTK naïve bays classifier and Scikit learn Random Forest classifier both one by one and check which one performs better




Random Forest Classifier with BI-Gram Countvectorizer / TF-IDF


I have used Scikit learns Random Forest classifier, which expect data to be in numeric format. To do that, I have first combined the tokenized words of reviews and then fed them in Count Vectorizer and TF-IDF both one by one to check which produces better result. 

First, I have trained random forest with vectors that I received with count vectorizer and this model produced the better results than others. It got 85% accuracy, precision, recall and F1-score.

Second, I have again trained random forest with TF-IDF vectors and it was better than Naïve Bays but poor than Random Forest with Contvectorizer model. I got 82% accuracy, precision, recall and F1-score with this model. 

Feature Importance Using Random Forest

Random forest offers importance score against each attribute. We can take advantage of it and know which features were most important

