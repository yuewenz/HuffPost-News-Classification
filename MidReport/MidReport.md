<br>

**CS7641 Group12**

*Team Members: Jiabin Fang, Wang Xie, Winnie Zheng, Xinyu Chen, Ziyi Dai*

<br>


## Introduction and Problem Definition

Today, with the rise of the Internet, more and more news is generated and spread out. Therefore, a good news classification system would allow people to access information of interest more effectively and enable new agencies to manage news more efficiently. To reach this goal, text classification, a process of assigning tags and categories to texts according to their descriptions or contents, should be applied<sup>[1]</sup>. As a foundational task in one of the Natural language processing (NLP) applications<sup>[2]</sup>, test classification would first transform unstructured text data into structured text, and then insert the text into machine learning (ML) models for classification purpose. Specific goals of this project is to predict categories for the given news using available news features such as headlines and short descriptions so as to rapidly figure out essential tags of the news. Over 200k Huffpost news was selected for use, with multiple machine learning and deep learning algorithms trained and validated for classification purposes. The model trained on our dataset with the best performance can be used to identify categories of tags for large quantities of untracked news articles with headlines or short descriptions available. 

## Data Collection 

The dataset was directly downloaded from Kaggle in JSON format (https://www.kaggle.com/rmisra/news-category-dataset), the raw news data contains 202,372 records with 6 attributes (category, headline, authors, link, short description, and date). Some data pre-processing was conducted before inserting the data into machine learning classification models.

## Methods

### Data Pre-processing

Attributes of short description and headline serve as the target features for predicting news categories. First, we removed records with empty headlines and empty short descriptions. 181,140 records were retained for further use. The original news data has 41 unique categories (labels), ranging from the least category of 1,004 rows in “EDUCATION” to the most category of 32,739 rows in “POLITICS”. Figure 1 below shows that this is quite imbalanced data, so we merged categories of similar meaning into one large category, such as “GREEN” and “ENVIRONMENT” as “ENVIRONMENT”, “SCIENCE” and “TECH” as “SCIENCE & TECH”, the combination turns 41 categories into 28 categories. Figure 2 and 3 show the average number of words within the headline and short descriptions in each category. 

<br>

<p align="center"> 
  <img src="image1.png" 
       alt="Figure 1. Number of News in Each Category"
       style="zoom:100%;" align="center" />    
</p>
<p align="center">Figure 1. Number of News in Each Category</p>

<br>

<p align="center"> 
  <img src="image2.png" 
       alt="Figure 2. Average Number of Words in Headline in Each Category"
       style="zoom:100%;" align="center" />    
</p>
<p align="center">Figure 2. Average Number of Words in Headline in Each Category</p>

<br>

<p align="center"> 
  <img src="image3.png" 
       alt="Figure 3. Average Number of Words in Short Description in Each Category"
       style="zoom:100%;" align="center" />     
</p>
<p align="center">Figure 3. Average Number of Words in Short Description in Each Category</p>

<br>

Some pre-processing steps were taken before splitting the training and testing dataset. Firstly, non-alphabetic words were removed from headlines and short descriptions, including numbers and punctuations. Secondly, stop words such as “the”, “is” were removed since they did not add to additional meanings. Thirdly, lemmatization was performed to reduce words into dictionary root form. Last, a new single attribute combining the headline and short description attribute was created for more powerful text representation.

Finally, numeric labels were generated for each news category and the dataset was divided into 8:2 training and testing subsets, with the training dataset holding 144,912 records and the testing dataset holding 36,228 records.


### Tokenizing and Text Representation

Before inserting data into classification models for training and predicting, tokenizing and text representation are necessary steps to transform a sentence string into a list of tokens and transform the unstructured text into mathematically computable forms. In this study, we adopted two commonly-used text representative methods to extract text features, the bag-of-words model and the TF-IDF model. 

The bag-of-words model is a simplifying model where the text is represented as the bag of its words with word count specified. The biggest disadvantage of this model is that it disregards the co-occurrence statistics between words, assuming the independence of all words, and due to the large size of vocabulary, the model is highly-dimensional and highly-sparse. Here we used the 10,000 most common words with unigrams and bigrams. 

The TF-IDF model, on the other hand, is intended to reflect how important a word is to a document in a corpus. The TF-IDF value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word. 

Besides these two traditional text representative methods, distributed text representation using word embedding was also considered for deep learning classification models. The underlying idea of word embedding is that a word is characterized by the company it keeps, therefore, word embedding forms a real-valued vector that encodes the meaning of the word such that words closer in the vector space are expected to be similar in meaning. Here we used the GloVe embedding, with 25,000 most common words and a maximum length of 50 words considered. 

### Machine Learning Classification Algorithms

#### Naive Bayes Classifier

Naive Bayes Classifier (NBC) is a set of supervised learning algorithms based on Bayes’ theorem which could be performed with naive independence assumptions between each feature. Therefore, a NBC could indicate the presence or absence of one feature of a class is not related to the other features’ presence or absence. This precise nature of the probability model could help NBC to be very efficient when we used to train the datasets<sup>[3]</sup>.

<p align="center"><img src="https://latex.codecogs.com/svg.image?P(x_i&space;\mid&space;y)&space;=&space;\frac{1}{\sqrt{2\pi\sigma^2_y}}&space;\exp\left(-\frac{(x_i&space;\mu_y)^2}{2\sigma^2_y}\right)" title="P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i \mu_y)^2}{2\sigma^2_y}\right)" /></p>

where <img src="https://latex.codecogs.com/svg.image?x_i" title="x_i" /> represents the feature vector and <img src="https://latex.codecogs.com/svg.image?y" title="y" /> represents the corresponding label. 

Among the different types of naive bayes classifier, we choose the Gaussian Naive Bayes because it could perform better than the other two based on it formula which depends on the mean (μ) and Bessel corrected variance (σ) of the frequency of each word in the class of messages.

#### Support Vector Machine 

Support Vector Machine (SVM) is a set of supervised learning methods which are useful for pattern recognition and regression analysis<sup>[4]</sup>. When labeled training datasets for each category are given to SVM, they could categorize new text. Therefore, the SVM model could be applied well in the field of text classification. 

<p align="center"><img src="https://latex.codecogs.com/svg.image?\min_&space;{w,&space;b}&space;\frac{1}{2}&space;w^T&space;w&space;&plus;&space;C&space;\sum_{i=1}\max(0,&space;y_i&space;(w^T&space;\phi(x_i)&space;&plus;&space;b))" title="\min_ {w, b} \frac{1}{2} w^T w + C \sum_{i=1}\max(0, y_i (w^T \phi(x_i) + b))" /></p>

where <img src="https://latex.codecogs.com/svg.image?x_i" title="x_i" /> represents the feature vector and <img src="https://latex.codecogs.com/svg.image?y_i" title="y_i" /> represents the corresponding label.

In order to be more flexible in the choice of penalties and loss function and scale better to large numbers of samples, for this case, we applied linear SVC which could support both dense and sparse input and handle the multiclass support based on a one-vs-the-rest scheme<sup>[5]</sup>.

#### Multinomial Logistic Regression

Multinomial Logistic Regression (MLR) generalized logistic regression to multi-classification problems. In the multi-class case of logistic regression, the training algorithm uses the one-vs-all(one-vs-rest) scheme. In particular, logistic regression is used to predict the probabilities of all labels separately, and the label with the highest probability wins.

After tokenizing and vectorizing, the scaled counts of words(for bags-of-words model) or the TF-IDF values(for TF-IDF model) are used as features of data points. In our approach, we used gradient ascent(stochastic gradient ascent) with regularization to train our model, and weights are trained separately for different labels. In each step for each label, we assign the current label to 1 and the others to 0, which reduces the problem to binary classification. In each iteration, the loss is the difference between the real label and the predicted label passing a sigmoid function, and we use the loss to update weights in each iteration. In stochastic gradient ascent, the weights are updated using a mini-batch of the data points. 

After training, the dot product of the data point and weights for a given label passing the sigmoid function indicates the probability of the data point being(belonging to) that label. For simplicity, since the sigmoid function is an increasing function, the highest dot product of data point and weights indicate the highest probability, we calculated dot products for all pairs of data points and labels and used an argument max to get the final predicted labels.

#### Convolutional Neural Network

The convolutional neural network (CNN) is a class of deep neural networks in deep learning, which has achieved remarkable results in natural language processing in recent years.<sup>[6]</sup> After using the unsupervised learning algorithm GloVe to obtain embedded vectors as input, in this text classfication task, we passed this feature matrix through a series of convolution layers with filters, pooling layers, and fully connected layers. Finally, it classified outputs as specific values for each class with an activation function. 

#### Recurrent Neural Network

Recurrent Neural Network (RNN) is a generalization of neural networks which allow previous outputs to be used as input while still having hidden layers. After the output is generated, it is kept in the memory while sending back to the recurrent network. Instead of forward passing the calculation in neural networks, RNN is able to use the memory to process a sequence of input data, which makes all the inputs dependent on each other. However, RNN suffers from the gradient vanishing issue which prevents models from capturing long term dependencies, therefore, Long Short Term Memory (LSTM) is introduced to solve the problem. 

We used pre-trained GloVe word embeddings as the input, along with the word vectors to generate the output to pass through each convolutional layer in a many-to-one mechanism. Here GloVe uses a co-occurrence matrix where each element represents the number of times that a target occurred with a specific context. We used the LSTM library to add the layer before training, with the argument of “return sequences” set to true, the output of the hidden state of each neuron is used as an input to the next LSTM layer. We used mean square error as the loss function, we used “Adam” as the optimizer, to build the LSTM model. The final output of the RNN model is a list of indices corresponding to each category.

## Results and Discussion

Table 1.1 and 1.2 presented the results of validation set under traditional machine learning classification algorithms using either the bag-of-word method or the TF-IDF method using different input features. For model performance evaluation, accuracy and micro F1-score were selected as the metrics for assessment. For now, all the model hyperparameters were set to default and have not been tuned and the hyperparameter tuning and model optimization will be conducted after midpoint check, it is assumed that the accuracy and F1-score would both increase after some tuning effort. 

From the two tables below, from the input feature perspective, it is not hard to see that using the combination of both headline and short description will present us with the highest classification accuracy, this result is consistent with our assumption since the combination of these two attributes adds to more text representation power. Moreover, if only a single attribute is available, the results showed that the headline attribute actually does have more explanatory power of the news category than the short description attribute. On the hand, from the classification algorithm and perspective, Naive Bayes algorithm performs the best using the bag-of-words method, while Multinomial Logistic Regression performs the best using the TF-IDF model. Support Vector Machine seems to fail in achieving satisfactory performance using both text representative methods. More solid conclusions could be drawn after more model tuning effort will be conducted.

<p align="center">Table 1.1: Model Performance Statistics using Bag-of-Words Method</p>
<table>
  <tr>
   <th rowspan="2" align="center">Method</th>
   <th colspan="5" align="center">Accuracy</th>
   <th colspan="3" align="center">F1-Score</th>
  </tr>
  <tr>
   <th colspan="3" valign="middle">Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
   <th>Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
  </tr>
  <tr>
   <th>NBC</th>
   <td colspan="3" >13.21%
   </td>
   <td >10.50%
   </td>
   <td>9.10%
   </td>
   <td>13.54%
   </td>
   <td>8.36%
   </td>
   <td>9.03%
   </td>
  </tr>
  <tr>
   <th>SVM</th>
   <td colspan="3" >5.35%
   </td>
   <td>5.76%
   </td>
   <td>5.25%
   </td>
   <td>5.83%
   </td>
   <td>5.12%
   </td>
   <td>4.88%
   </td>
  </tr>
  <tr>
   <th>MLR</th>
   <td colspan="3" >10.70%
   </td>
   <td>10.45%
   </td>
   <td>9.78%
   </td>
   <td>10.23%
   </td>
   <td>9.05%
   </td>
   <td>9.28%
   </td>
  </tr>
</table>

<br>

<p align="center">Table 1.2 : Model Performance Statistics using TF-IDF Method</p>
<table>
  <tr>
   <th rowspan="2" align="center">Method</th>
   <th colspan="5" align="center">Accuracy</th>
   <th colspan="3" align="center">F1-Score</th>
  </tr>
  <tr>
   <th colspan="3" >Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
   <th>Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
  </tr>
  <tr>
   <th>NBC</th>
   <td colspan="3" >13.01%
   </td>
   <td >10.77%
   </td>
   <td>9.22%
   </td>
   <td>13.16%
   </td>
   <td>9.92%
   </td>
   <td>9.03%
   </td>
  </tr>
  <tr>
   <th>SVM</th>
   <td colspan="3" >5.94%
   </td>
   <td>6.23%
   </td>
   <td>5.85%
   </td>
   <td>6.31%
   </td>
   <td>6.06%
   </td>
   <td>5.62%
   </td>
  </tr>
  <tr>
   <th>MLR</th>
   <td colspan="3" >15.110%
   </td>
   <td>13.24%
   </td>
   <td>11.60%
   </td>
   <td>10.70%
   </td>
   <td>8.40%
   </td>
   <td>8.16%
   </td>
  </tr>
</table>
<br>

Apart from the above three traditional machine learning methods, we also trained two deep learning models, one using the Concurrent Neural Network and one using the Recurrent Neural Network. It can be observed from table 2 that the architecture of additional RNN layers outperforms the one with just convolutional layers, achieving higher accuracy and F1-score. And for now, the deep learning models greatly outperform the traditional machine learning classification models. 

<br>

<p align="center">Table 2: Model Performance Statistics using GloVe Embedding</p>
<table>
  <tr>
   <th rowspan="2" align="center">Method</th>
   <th colspan="5" align="center">Accuracy</th>
   <th colspan="3" align="center">F1-Score</th>
  </tr>
  <tr>
   <th colspan="3" >Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
   <th>Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
  </tr>
  <tr>
   <th>CNN</th>
   <td colspan="3" >60.45%
   </td>
   <td >39.06%
   </td>
   <td>27.59%
   </td>
   <td>61.01%
   </td>
   <td>38.70%
   </td>
   <td>19.41%
   </td>
  </tr>
  <tr>
   <th>RNN</th>
   <td colspan="3" >67.35%
   </td>
   <td>38.35%
   </td>
   <td>30.65%
   </td>
   <td>66.55%
   </td>
   <td>39.55%
   </td>
   <td>33.39%
   </td>
  </tr>
</table> 

<br>

To achieve better performance of models, hyperparameter tuning as well as some regularization methods were conducted towards each model for performance enhancing. For the Naive Bayes Classifier method, Gaussian Naive Bayes, Bernoulli Naive Bayes as well as Multinomial Naive Bayes were tested to adjust the prior distribution of the data and finally Gaussian Naives Bayes was chosen. For the Support Vector Machine method, MinMaxScaler and StandardScaler were used for data normalization. The hyperparameter *C* and *gamma* were optimized using grid search with cross validation and finally *C* = 1, *gamma* = 0.0001 and StandardScaler normalization were chosen. For the Multinomial Logistic Regression method, stochastic gradient ascent (SGA) and regularization were conducted. SGA batches the input data in each iteration and increases the convergence speed, and regularization avoids overfitting for our model. Cross-validation scheme was applied to set the optimal value of the regularization parameter as 1.0. Moreover, we stored the splitting indices after train test split instead of directly storing the training and test data, which decreases the variance of nlp processing steps.

For two deep learning algorithms, for the Convolutional Neural Network method, instead of using simple two convolutional layers, we tried TextCNN in classification. Pre-trained GloVe word embeddings were used as the initial embedding layer. All words, including the unknown ones utilizing GloVe, were kept initialized, with other parameters in the model being tuned and learned. The normal initializer was used to initialize the convolutional layers with Dropout layers added as regularizers to prevent overfitting and increase model accuracy. For Recurrent Neural Network method, pre-trained GloVe word embeddings was applied as the input, along with the word vectors to generate the output to pass through each convolutional layer in a many-to-one mechanism. Multiple LSTM library layers were added into the training model, along with the loss function being set as mean squared error and the optimizer set as Adam. We attempted additional LSTM layers each with a dropout rate of 0.2 as the regularizer to prevent overfitting. The output layer is a fully connected layer with units of 28, which is the number of categories in the dataset. We merely achieved a significant accuracy improvement with additional LSTM layers, and therefore, only one additional layer was included in the model.

Table 3.1 and 3.2 presented us the model performance using the same statistics after the above stated tuning effort. Comparing these statistics we can tell that MLR method performed the best using the Bag-of-Word vectorization method with news headline and short descriptions inserted as the input. 

<p align="center">Table 3.1: Model Performance Statistics using Bag-of-Word Method after Tuning</p>
<table>
  <tr>
   <th rowspan="2" align="center">Method</th>
   <th colspan="5" align="center">Accuracy</th>
   <th colspan="3" align="center">F1-Score</th>
  </tr>
  <tr>
   <th colspan="3" valign="middle">Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
   <th>Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
  </tr>
  <tr>
   <th>NBC</th>
   <td colspan="3" >21.32%
   </td>
   <td >22.08%
   </td>
   <td>13.30%
   </td>
   <td>19.62%
   </td>
   <td>20.01%
   </td>
   <td>13.01%
   </td>
  </tr>
  <tr>
   <th>SVM</th>
   <td colspan="3" >47.00%
   </td>
   <td>50.35%
   </td>
   <td>33.82%
   </td>
   <td>38.00%
   </td>
   <td>40.18%
   </td>
   <td>25.48%
   </td>
  </tr>
  <tr>
   <th>MLR</th>
   <td colspan="3" >53.85%
   </td>
   <td>43.10%
   </td>
   <td>37.08%
   </td>
   <td>37.19%
   </td>
   <td>24.06%
   </td>
   <td>22.23%
   </td>
  </tr>
</table>

<br>

<p align="center">Table 3.2 : Model Performance Statistics using TF-IDF Method after Tuning
</p>
<table>
  <tr>
   <th rowspan="2" align="center">Method</th>
   <th colspan="5" align="center">Accuracy</th>
   <th colspan="3" align="center">F1-Score</th>
  </tr>
  <tr>
   <th colspan="3" valign="middle">Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
   <th>Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
  </tr>
  <tr>
   <th>NBC</th>
   <td colspan="3" >21.06%
   </td>
   <td >22.05%
   </td>
   <td>13.40%
   </td>
   <td>19.38%
   </td>
   <td>20.43%
   </td>
   <td>13.14%
   </td>
  </tr>
  <tr>
   <th>SVM</th>
   <td colspan="3" >47.24%
   </td>
   <td>50.00%
   </td>
   <td>34.35%
   </td>
   <td>37.92%
   </td>
   <td>40.00%
   </td>
   <td>25.61%
   </td>
  </tr>
  <tr>
   <th>MLR</th>
   <td colspan="3" >31.56%
   </td>
   <td>30.32%
   </td>
   <td>17.22%
   </td>
   <td>10.23%
   </td>
   <td>9.04%
   </td>
   <td>18.73%
   </td>
  </tr>
</table>

<br>

Table 4 below displays the model performance for the two deep learning frameworks after some more model modification effort. Consistent with what was observed before, the RNN method still outperformed the CNN method (this also matches our assumption based on the model characteristics of different neural networks), but as stated in the model tuning description, very little improvement in model accuracy was achieved after adding additional LSTM layers.

<p align="center">Table 4: Model Performance Statistics using GloVe Embedding after Tuning</p>
<table>
  <tr>
   <th rowspan="2" align="center">Method</th>
   <th colspan="5" align="center">Accuracy</th>
   <th colspan="3" align="center">F1-Score</th>
  </tr>
  <tr>
   <th colspan="3" >Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
   <th>Headlines and Short descriptions</th>
   <th>Headlines</th>
   <th>Short descriptions</th>
  </tr>
  <tr>
   <th>CNN</th>
   <td colspan="3" >63.00%
   </td>
   <td >41.65%
   </td>
   <td>34.33%
   </td>
   <td>63.43%
   </td>
   <td>38.80%
   </td>
   <td>21.21%
   </td>
  </tr>
  <tr>
   <th>RNN</th>
   <td colspan="3" >68.01%
   </td>
   <td>39.02%
   </td>
   <td>31.32%
   </td>
   <td>66.88%
   </td>
   <td>40.22%
   </td>
   <td>35.16%
   </td>
  </tr>
</table> 

 <br>

Figure 4 and Figure 5 presented us with the confusion matrices generated by the two models with relatively the best performance. It is clear from the confusion matrix that the model performs the best towards categories with the largest sample sizes, i.e., POLITICS, WELLNESS and ENTERTAINMENT. Moreover, it is obvious that model is biased towardspredicting these more common classes (which is the most evident in the POLITICS category). It can also be observed from the visualization that categories sharing similar topics, which could lead to similar descriptions and contents do confuse with each other, for example, categories of WELLNESS and HEALTHY LIVING, as well as categories having overlapping topics with each other, for example, categories of BUSINESS and POLITICS. Some news could involve topics of multiple categories, especially containing sentences or words which are characteristic of other news categories, but the label just stands for the main category it lies in. 

<br>

<p align="center"> 
  <img src="image4.png" 
       alt="Figure 4. Confusion Matrix for MLR with using Bag-of-Word Method"
       style="zoom:100%;" align="center" />    
</p>
<p align="center">Figure 4. Confusion Matrix for MLR with using Bag-of-Word Method</p>

<br>

<p align="center"> 
  <img src="image5.png" 
       alt="Figure 5. Confusion Matrix for RNN using GloVe Embedding"
       style="zoom:100%;" align="center" />    
</p>
<p align="center">Figure 5. Confusion Matrix for RNN using GloVe Embedding</p>

<br>

## Conclusions

### Conslusions

In this study, a number of classification models were built using both machine learning algorithms and deep learning frameworks under different text representation methods to classify the news categories using data from Huffpost. A full process of text classification has been initiated and finished, from data collection and cleaning, to text vectorization and representation, to model training and tuning. In the meantime, how the input of different features impact the text classification performance was also tested (i.e., the news headlines and short descriptions in our case). Results showed that combination of headline and short description brings about relatively the best performance, which makes sense as this feature has more explanatory power. Moreover, model performance after tuning effort showed that the Multinomial Logistic Regression method performs the best using the Bag-of-Word method. But if compared with deep learning models, RNN outperforms all the other models (achieving an accuracy of around 68%) as RNN takes the interrelation between the input text into consideration as well as integrates more complex distributed text representative methods.

### Limitations and Possible Further Steps

As we can observe from the data, this is a quite imbalanced dataset, therefore, possible further improvements could be targeted towards dealing with the imbalance issue among different categories. In this study, we did not perform example weighting to deal with the possible model bias towards more common classes, but this step should definitely be taken into consideration for possible further steps. 

## References

[1] *Guide to Text Classification with Machine Learning*. (2020). MonkeyLearn. https://monkeylearn.com/text-classification<br>
[2] Lee, J. Y., & Dernoncourt, F. (2016). Sequential short-text classification with recurrent and convolutional neural networks. *arXiv preprint arXiv*:1603.03827.<br>
[3] Mangal, S. B., & Goyal, V. (2014). Text news classification system using Naïve Bayes classifier. *Intenational Journal of Engineering Sciences, 3.*<br>
[4] Saigal, P., & Khanna, V. (2020). Multi-category news classification using Support Vector Machine based classifiers. *SN Applied Sciences*, 2(3), 1-12.<br>
[5] *sklearn.svm.LinearSVC — scikit-learn 0.24.1 documentation*. (2007b). Scikit-Learn. https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC<br>
[6] Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *EMNLP*.