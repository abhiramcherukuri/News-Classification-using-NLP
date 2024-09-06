# News-Classification-using-NLP
The purpose of this project is to design and implement a machine learning implementation that correctly predicts if a given article would be considered fake news.

ABSTRACT
The concept of Real and Fake news Classification and Detection is a domain which
is still in the initial-development stage as compared to other projects of similar kind in
this domain.ML or Machine Learning is a useful part of this project. The purpose of
using these algorithms is to help the users to understand the various difficult and
unyielding problems and to build Smart Artificial Intelligence and Machine Learning
Systems to tackle problems for this concept. For the purpose of this
research, we have used the concept of NLP along with two popular Machine
Learning Algorithms for the purpose of the classification of real and fake
news. They are Logistic Regression and Decision Tree Classifier. Other Algorithms
like Random Forest, Support Vector Machine can also be used for this. The purpose
of the project that has been built here is not to simply perform classification of
the news articles as one cannot simply implement the ML algorithms and then
predict whether the news is real or not. No, what has been done here is a clear-cut
implementation and a mix of Data Science Tools as well as ML concepts for the
classification as well as prediction of fake news. Various ML models will be
implemented here for the prediction of news. The process of the classification will
focus on using data science tools for pre-processing of the text and then
using the results of the pre-processed dataset to build a improved model for the
project. The major obstacle which was tackled whilst project was the lack of a
properly processed dataset as well as a pre-defined model to differentiate between
the two categories of news as mentioned in the title of the paper. For simplicity’s
sake, some of the more commonly known ML algorithms and classifiers have been
implemented on some datasets that are available on the internet. The results, when
the ML models were implemented on the dataset, have been very encouraging and
can prove to be very useful if any future work is done on this project or in this
particular domain.


OBJECTIVE
This project proposes the question of whether it is possible to detect fake news
through machine learning models. Specifically, the aim of this project is to determine
the ideal model that is efficient in predicting fake news while also limiting the cost of
memory and storage for computation. ”Fake news” has been a very recent and
prevalent problem within recent years.
The purpose of this project is to design and implement a machine learning
implementation that correctly predicts if a given article would be considered fake
news. The contributions of this project are as follows:
• Introduces the topic of fake news and the various machine learning algorithms to
build a model to accurately classify a piece of news as REAL or FAKE.
• Provides an overview of the history and implications of fake news.
• This advanced python project of detecting fake news deals with fake and real
news. Using SK-Learn tools, we will a Tfidf Vectorizer on our dataset.
• Then, we initialize a Passive-Aggressive Classifier (PAC) to fit the model, which
will result in an accuracy score and a confusion matrix that tell us how
well, our model fares.
• Presents a possible solution and lays some groundwork in further study in this
area.


INTRODUCTION
Let us take a look at what Fake news might be. FAKE news can be the spread of
any sort of misinformation, whether it be through online sources, newspapers,
magazines, websites etc. The most prevalent form of spreading fake news is
through news channels and through news reporters as it is very easy to twist
facts in such a way as to spread an entirely different point as compared to what
we want to say. It is not only a problem of recent times, as Fake news have long
been in existence. In this modern era, it is possible for anyone to spread any
sort of news through electronic means which might not be credible as in today’s
era, it is highly possible that fake news will spread much faster than real news.
Especially on the internet, the amount of fake news is very high which is easily
accessed and is deemed believably most of the people today. This problem of
detecting the fake news is resolved through implementing the mentioned ML
models by using the concept of NLP [7]. The major purpose of the projects that
were built previously was to classify and detect the online news and social media
posts in this era where there has been a massive increase in the amount of
fake news prevailing on the Web. To provide an overview, what has been done in
this project is that a dataset is taken, and then after pre-processing, NLP and
ML algorithms have been implemented to develop a model which is then used on
several publicly available datasets to show the accuracy of the models which
have been built in this project. Another important aspect that has to be taken into
account is the number of times that a given word occurs in the given dataset
which is being for the real and fake news classification. The instance of
Cloud Visualization is used in the implementation of this project. This
representation of words in the form of a cloud represent various words. Let us
consider some words such as Russia, Ukraine, War, Political etc. These words will
occur a maximum number of times in the dataset as the probability of news
generation is very high, whether it be real or fake. As already mentioned, various
detests have been used which can contain several types of stories, whether
they be real or fake. The Classification Model is generated using the ML
models and the word cloud is used for the classification and prediction of the
category of news articles i.e., to classify whether they are real or fake. All of this will
be explained properly int the implementation aspect of the project.


METHODOLOGY
•Detailed Description of given Dataset
The dataset which hashas been used forfor this project has been taken from
various sources, all of which are available on the internet, free of cost. The sources
of the dataset for the news articles can be KaggleKaggle, GitHub or somesome
other portal [3]. These articles were collected from various news sources and that
they were labelled as Real and faux. The news articles which have collected
into the dataset are categorized into two categories –Real and Fake. The preprocessing
of the dataset has to take place for the process of classification to
take place. This division are often seen within the project where the news
articles have been sorted into a separate category and others inin a separate
category. Some of the articles have not been classified as Real or Fake in the preprocessing
module as some of their information such as ID, Label etc. is missing
[15]. The pre-processing of this dataset itself is quite a tedious task and if proper
steps are not taken, can result in quitequite an imbalanced datasetThis
problemblem is not only in this project butbut in tthe othersas wellas well FFor
therectification of this error, we skip the records withwith missingdata.


Exploration of different ML Models
There are several types of MachineMachine Learning Algorithms wwhich can bbe
used for the purpose of this project. After rigorous experimentation and testing of
different algorithms, different ML models were deployed for the classification
process. Some models which had already proven to be effective were taken
into consideration whereas other models which did not give correct, or
desired results, were removed from the proposed work. The models which
have been retained for the project work are Logistic Regression and Decisionand
Decision Tree Classifier as they have given a higher aaccuracy as compared to
other models like SVM, Naïve Bayes etc.Bayes etc. AAfter theidentification
ofidentification of the models to be deployed, the NLP concepts were implemented
on them for the prediction of Fake news. For safety, nltk library of NLP module was
also implemented on some of the higher performing rejected models. But
still, the accuracy of the two mentioned models has proven to be greater as
compared to other Machine Learning Models.
•Naïve Bayes
The Naïve Bayes Algorithm was implemented for the classificationclassification
of Real andand Fake News because, as per the previous works, works, ts
performance was quite good. But while implementing this model alongalong with
NLP concepts, its performance was found to be lacking as compared to others.
This can be explained by the algorithmic formula for thethe NaïveNaïve Bayes
as wellwell as the classification report that was generated.
•Support Vector Machine
Support Vector Machine is another model that was implemented for the
purpose of Real and Fake news classification. This model is very useful and has
several merits. The rate of training of this model is relatively higher as
compared to others. Also, this model waswas used by previous works for the
Real and Fake news classification. However, this model also did not perform
accurately. Simply, the accuracy of this model was lower as compared to
others. This model has a number of advantages, such as the ability of
tolerance to unimportant material in the dataset[13]. But still, for the purpose of
this project work, this model has been discarded.
•Passive Aggressive
The Passive Aggressive model is not as much used for the Real and Fake
News Classification, but still it is an important and upcoming model in
various other domains, due to which this was also taken into consideration.
The implementation of this model is easier as compared to other complex
models and has been demonstrated by several works of different authors [6].
But still, seeing the low accuracy score of this model, it was also discarded
from the given project work.
•Logistic Regression
The Logistic Regression is quite a popular model and has been used in
various domains of project work. The accuracy score of this model was quite
high as compared to other algorithms and the prediction was also accurate.
The Logistic Regression Model is capable of handling large amounts of data
which made it an essential part of the prediction and classification model.
This model has been used in this project for the classification purpose.
Re-Initialization
This process is known by various names such as re-coding, re-initialization or preprocessing.
This is the initial module of the project. The purpose of this module
is to process or re-initialize the given data into a suitable format which can be
used by the Machine Learning Model. There are various methods of proper reinitialization
such as assigning a proper ID, cleaning of repeated data, filtering
unwanted data etc. All of this is managed in this module of the project[14]. The
sklearn and NumPy libraries are used for this. Along with this, various NLP
concepts are also used in the pre-processing module. The major purpose of
this module is to ensure proper categorization and representation of data i.e.,
the labels of real and fake are accurately assigned to each row, the rows are
unique etc. Then the concept of NLP is used to ensure that words are
present int their base format for easier processing. There are various other
concepts of Data Science as well as Machine Learning that have been taken
into account for this project. Tokenization is an important aspect which is
used to split the text in any row in the dataset into singular words. The concept of
stop words is also used to remove the words which do not carry any meaning
in identifying whether the given news is real or not.

Machine Learning
In the 21stcentury, in the modern era, there is a rapid boom in the generation of fake
news, rumours etc. whether it be online or offline. Due to this rapid increase, the
need of a device or a software to detect fake news has also increased. Machine
Learning Models like Decision Tree Classifier and Logistic Regression have
been developed and implemented along with Natural Language Processing for
the classification and prediction of the fake news which has been generated.
The goal of this project is to merge the data science as well the concepts of
Machine Learning along with NLP to sort the category of news based on the
headlines as well as the news article. This means that the classification can take
place either by the title of the news article or the content. This module has been
successfully implemented in the project.
•Natural Language Processing
The vast amount of repetitive and irrelevant features which are present in the
given dataset often result or cause a significant negative affect on the
accuracy of the result as well as the performance of the classifier. This
causes a huge problem in the prediction and the classification module. In
order to tackle these issues, feature extraction is used to reduce the length of
the text as well as to implement stop-words and other modules. The process
of NLP consists of various steps and many functions are implemented in this.
Some of them are conversion of the casing of the alphabets, sum of the
number of words in a particular part of the dataset, removal of punctuations etc. The
Natural Language Toolkit or nltk of the NLP library is used in this project. The
concept of stop-words, stemming, tokenization etc. has been implemented in this
module and has been successfully integrated into the project
•Count Vectorizer
This is an important part of NLP and ML algorithms. The purpose of Count
Vectorizer is to take a particular aspect of the dataset and then removing
punctuation marks, conversion of the casing of the words, pre-processing of the
text in the dataset etc. The first step is to gather the data into a word cloud using
the implemented modules. Then, the generated vocabulary is used for the
classification and prediction of fake news. By using the mentioned module i.e.,
vectorizer, a table is generated which stores the number of times a word occurs in
the dataset and where it occurs.
•Term Frequency-Inverse Doc. Frequency
TF-IDF is another important aspect of NLP which has been used in this
project. It stands for Term Frequency. The IDF stands for Inverse document
Frequency. The number of times a word occurs in the dataset is stored after the reinitialization
of the dataset. The above module is used to calculate the
importance of a given string as per the classification point of view. Its purpose is to
convert sentences into an array of integers. For their usage in further modules.
The formula which can be used for the measurement of this and for the
measurement of corpus is : 𝑇𝐹-IDF= 𝑇(w)d × 𝐼𝐷𝐹(w)D. Let’s try to understand this
concept through a example. Consider a situation in which we have a news
report which consists of over 500 words. To determine the TF and IDF for the given
word “president”. Term “president” is present in the document for a total number
of 15 times. By this logic, the Term Frequency will come out to be 15/500
i.e., 0.03. Now we calculate the IDF for 300 reports for the given word. By
using the formula, the IDF and TF-IDF is calculated.


IMPLEMENTATION
The various modules which have been described in this research paper are
implemented in the project of Real and Fake News Classification. Different
tools, functionalities have been used for the implementation of this project. First
off, the Data Science tools such as sklearn, NumPy have been used for
the re-initialization. But first, the pandas library of python is used. It is a freely
available, open-source library that has significant uses in the field of Data
Science and Machine Learning, one of them being the classification as done in
thus project. The accuracy of different models is also represented through
graphs by using matplotlib to plot graphs. Various graphs can be used
such as bar graph, scatter plot etc. Here, a simple Bar Graph has been used for
the visualization. Prior to the implementation of the various ML algorithms,
the concepts of NLP such as Count Vectorizer and TF are added in the
project. This is done using the nltk library. NLTK stands for Natural Language
Toolkit. It is used in the re-initialization phase and also in the subsequent steps.
The project was created using the software known as Anaconda Jupyter.
So, all the importing of libraries as well as implementation of models
takes place in Jupyter notebook. While ensuring that no alterations were
made in the processing of the training as well as the testing data, we
have further attached the data that has been tested with certain algorithms
of Machine Learning and Natural Language Processing. One of the prerequisites
of this project was to deploy a model that is able to calculate
the Term Frequency as well as Inverse Doc. Frequency, which has been
successfully implemented. The purpose of this project was to classify the given
news articles into two categories-Real and Fake and also to train the
model on the given dataset to predict whether the news which is prevailing
at that moment is authentic or not. Sometimes, the classification models
also give errors in predicting the results of real or fake news. The primary goal
is to create a model that supports count vectorization and TF. -IDF.
Also, before the actual classification or prediction actually takes place, we
have to perform feature extraction as well. After these steps are complete,
we can deploy the NLP and Machine Learning models to detect their accuracy
as well as to predict whether the given news article is real or not. This phase's
goal is to reduce the size of the information by eliminating unnecessary data
that isn't necessary for categorization. The information was then altered so
that it wouldn't produce impartial findings when applying ML techniques if
the first half of the information had a false label set and the second half had
a true label.

CONCLUSION
The project of Real and Fake news Classification has been successfully
deployed. Various types of Machine Learning Algorithms were used to classify
the news into the category of Real or Fake-whether they be supervised
or unsupervised. As seen from the previous works, the most popular topic for
the generation of fake news is politics. So, it might be possible that the models
which have been deployed give accurate results for the news in politics but
there might be a slight margin of error when comparing to news of other
topics. From what has been observed in this project and the previously
developed models, it has been induced that it is a very tedious task to generate a
ML model that is capable of working on all categories of news. Here, the best
work has been done to deploy the models accurately for the classification to take
place. We observed that the Random Forests algorithm with an easy term
frequency-inverse document frequency vector gives the simplest output compares
to others. Our study examines various text properties which will be wont to
distinguish fake and real content, and we trained a mixture of various
machine learning algorithms using these properties. The models which were used
for the classification process have been deployed accurately as per the
accuracy score. The results show that that the project has been implemented
successfully and the prediction module also works correctly 97% of the time. There
is a slight margin of error which an be rectified in the future works. If this project is
taken up in the future, then those errors can be taken into account and can be
rectified by connecting the developed models to the cloud for the classification
and prediction of real and fake news. Other factors like the writers of the
particular news articles, the score of how much fake news or rumours are
given by a particular writer, popular topics etc. can be taken into account as well.
