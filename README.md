<h1 style="text-align:center;font-size:30px;" > Quora Question Pairs Similarity Detection </h1>
<img src='images/quora.jpg'/>
<h1> 1. Business Problem </h1>
<h2> 1.1 Description </h2>
<p>Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.</p>
<p>
Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.
</p>
<br>
> Credits: Kaggle 

__ Problem Statement __
- Identify which questions asked on Quora are duplicates of questions that have already been asked. 
- This could be useful to instantly provide answers to questions that have already been answered. 
- We are tasked with predicting whether a pair of questions are duplicates or not. 

<h2> 1.2 Sources/Useful Links</h2>

- Source : https://www.kaggle.com/c/quora-question-pairs
<br><br>____ Useful Links ____
- Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments
- Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
- Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
- Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30

<h2>1.3 Real world/Business Objectives and Constraints </h2>

1. The cost of a mis-classification can be very high.
2. You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
3. No strict latency concerns.
4. Interpretability is partially important.

<h1>2. Machine Learning Probelm </h1>
<h2> 2.1 Data </h2>
<h3> 2.1.1 Data Overview </h3>
<p> 
- Data will be in a file Train.csv <br>
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate <br>
- Size of Train.csv - 60MB <br>
- Number of rows in Train.csv = 404,290
</p>

<h3> 2.1.2 Example Data point </h3>
<pre>
"id","qid1","qid2","question1","question2","is_duplicate"
"0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
"1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
"7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"
"11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"
</pre>

<h2> 2.2 Mapping the real world problem to an ML problem </h2>
<h3> 2.2.1 Type of Machine Leaning Problem </h3>

<p> It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not. </p>
<h3> 2.2.2 Performance Metric </h3>

Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s): 
* log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
* Binary Confusion Matrix

<h2> 2.3 Train and Test Construction </h2>

<p>  </p>
<p> We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with. </p>

## Conclusion: 

The main objective of this Case Study is to classify a new question on Quora as either a duplicate of an existing question or not. Each day thousands of questions are asked on Quora. So there's bound to be questions that are duplicates. Imagine two people asking two differently worded questions which basically have the same meaning and hence the same answer. In this case if two questions are classified as duplicates of one another, using this flag, both the questions can be linked to the same set of answers. If they were not classified as duplicates of each other, then there would be seperate answers for each of these questions. This is not the ideal scenario. Because, ideally we would want the system to pick up duplicate questions and assign answers to a new duplicate question which has been answered for an earlier question. So having said that, the main objective of this experiment was to build an intelligent system which can classify new questions to be duplicate or not of all the existing questions.

For the given task, we are provided with a dataset which contains almost 400K sample pairs of questions. Each of these data points has a label 'is_duplicate'. 'is_duplicate' = 0, if the pairs of questions are not duplicate of one another. 'is_duplicate' = 1, if the pair of questions are duplicates! The problem that we have is a binary classification problem, since, for a given pair of question have to predict either 'is_duplicate' equal to 0 or 1. The metric that we will chose for this problem is 'log loss'. We will also use a binary confusion, precision and recall matrices to get more crisp ideas about individual classes. 

Few things to keep in mind:

1. The cost of misclassification is very very high: Imagine if we wrongly tag a question to be duplicate of another question, the user might be redirected to the answer page that is not meant for the new question that he has asked. This wil absolutely hinder user experience!

2. Idealy, we would want a probability value (P) of Q1 being similar to Q2. We can now set a threshold like if P > 0.95, then we will classify the questions to be duplicate of one another. After getting the probability scores, we can experiment with different values of P. For example we can try with P=0.95. If we are satisfied with the result then it's fine. If not, we can always go ahead and increase the threshold to 0.99 in order to avoid misclassification. 


Why have we chosen our metric to be 'log loss' ?

As we have discussed above, we will use our Key Performance Indicator to be 'log' loss. Minimising the Log Loss is basically equivalent to maximising the accuracy of the classifier. In order to calculate Log loss, the classifier must actually assign a probability value for each of the class labels. Ideally as the predicted class probabilities improve, the Log loss keeps on reducing. Log loss penalises the classifier very heavily if it classifies a Class 0 to be Class 1 and vice versa.  For example, if for a particular observation, the classifier assigns a very small probability to the correct class then the corresponding contribution to the Log Loss will be very large. Naturally this is going to have a significant impact on the overall Log Loss for the classifier,which will then become higher. But, in other scenario if the classifier assigns a high probability value to the correct class, then the Log loss will reduce significantly. Now, imagine a scenario where the model doesn't predict probability scores. It only predicts class labels. In this case, class labels can be either 0 or 1. So we won't get a deep understanding or any interpretability about why a particular pair of question has been labeled as either 0 or 1. Chosing the KPI as Log loss gives us this freedom. We can interpret the models. We can say this two questions are 95% similar or 80% similar, instead of just bluntly classifying them as duplicates.

For deep understanding of log loss please visit: https://datawookie.netlify.com/blog/2015/12/making-sense-of-logarithmic-loss/

Anyway, lets move on to EDA. A high level overview reveals that there are almost 250K question pairs that aren't duplicates and almost 150K question pairs are duplicates. Also, there are a total of 537933 unique questions out of which 111780 of questions occurs more than once. The maximum number of times any question is repeated across the entire data is 157. Thankfully, there are no duplicate pairs of questions. 

Before the data cleaning stage, we have done some basic feature engineering. For more information please refer to section 3.3. On analysing our 'word_share' feature we have seen that the two classes are partially seperable with some overlap. In general, as the 'word_share' count increases, the probability of any pair of questions to be duplicates also increases. This suggest that 'word_share' might be an important feature as far as this problem is concerned. The feature 'word_Common' might not be hugely important, as there is a lot of overlap between the classes. 

Also, I have used Decision Trees to determine the most important features out of all the 26 hand crafter features we have extracted from the text data.

We have removed HTML tags from each of the questions, removed all the punctuations, stopwords. We have expanded themost commonly used English contractions for each questions. We have also performed Stemming using SnowBall stemmer. 

After performing data cleaning, we have extracted some advance features. The four most important amongst them are - fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio and longest_substring_ratio. Go to section 3.5 for further details. On plotting the word cloud we have observed that 16110763 words are there in all the questions belonging to duplicate questions and almost 33201102 are there in the non duplicate pairs.

We have used these advance features to plot a TSNE plot. By looking at the plot, we can clearly see that the two classes can be seperated nicely, with some overlapping. This suggests that the advance features are useful in some ways.  

Now that we have extracted our basic and advance features, it's time for us to featurize the text in questions. We will use two approach here - using simple TFIDF Features and using TFIDF Weighted Word2Vec representations. We will combine all the features we have obtained into one single data frame and apply Machine Learning models on top of it.

Before building our models, the first thing we need to do is to build a baseline (random) model. Why this is done? Well, we have selected our KPI to be log loss. We all know that the minimum value of log loss is 0, but the maximum value goes till infinity! So we want to determine using a random model what our worst case log loss will be. In a random model - for every data point we generate the predicted class labels randomly. This is the most dumb model we can ever build. The log loss for this model is 0.88. So using this value we will clearly get an idea what out worst case log loss should be! Now using ML models, we will try and reduce this log loss and bring it as close to 0 as possible. 


The models we have used are Logistic Regression with SGDClassifier, Linear SVM with SGD Classifier, a simple Logistic Regression model and XGBoost model. Amongst all the featurizations and all the models that we have used, the XGBoost model applied on the simple TFIDF Vectorizors gave us the least value of test log loss - 0.299. The train log loss obtained using this model is 0.266. The small difference in the train and test log loss suggests that we might have a bit of overfitting (very small)







