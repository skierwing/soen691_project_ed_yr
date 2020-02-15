Abstract

Using big data techniques with predictive analytics has become a key to fully understand how things work in different fields, and further a key to sustainable developments and improvements. This project presents a comparative study between two data classifiers in big data, in an attempt to make predictions based on basketball statistics. Data obtained for analysis contains many irrelevant classes and therefore was pruned from unnecessary attributes. There will be a quantitative analysis of classifiers studied, as well as an understanding of basketball statistics and their application in predicting seasonal player scores and positions,as well as the chance of a team making it to playoffs.

I. Introduction:

   The overall performance of an NBA team considers the players who conforms to the team's lineup and the coach who trains them. Predictions of sports events has always been intriguing while challenging as there are many factors that have to be considered. The objective of this project is to make predictions on the positions of a NBA player is playing and on the outcome of the NBA playoffs. 
   
   We aim to solve the problem that:
   1. based on two datasets on the players and the teams, use SVM and Naïve Bayes classifiers to predict the outcome of the team who makes it to the playoffs, as well as the positions of players
   2. based on our research results, do an analystic study on the performance of different classifiers applied. 
   
   Related work:
   1. Data preparation: raw datasets[1] will be cleaned first and then converted into two separate formatted csv files, of which one is for the players and the other is for the team. The player dataset will have keywords such as name, id, height, blocks, assists, and points, etc. The team dataset will cover keywords like name, season, minutes played, and points, etc.
   2. Data mining techniques and sports data analysis: There will be a study on sports data mining[2] prior to the design of our prediction strategy, during the implementation, SVM and Naïve Bayes classifiers will be used. training data and test data will be 80% and 20% respectively out of all data when building models. 
   3. Calculation and evaluation on predictions.
   4. Analysis on the performance of classifiers studied.
   
II. Materials and Methods: 

  The dataset:
  Using NBA/ABA statistics data gathered from 1946 till 2004, the team will create the needed datasets to try to predict players scores and position.  The datasets will be cleaned, filtered and then labeled to suit the calculations that are going to be done. 
   
   technologies and algorithms that will be used:
   Spark's SVM and scikit-learn Naive Bayes libraries will be used to conduct the predictions.
   
References:

[1] NBA statistics data http://www.cs.cmu.edu/~awm/10701/project/data.html

[2] Cao,Chenjie "Sports Data Mining Technology Used in Basketball Outcome Prediction" https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1040&context=scschcomdis 


