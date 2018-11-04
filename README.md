# Yelp Recommender System INF-553
## Yelp Dataset Description
yelp academic dataset business.json : 188,593 records <br/> 
yelp academic dataset review.json : 5,996,996 records <br/> 
yelp academic dataset user.json : 1,518,169 records<br/> 
yelp academic dataset checkin.json : 157,075 records<br/> 
yelp academic dataset tip.json : 1,185,348 records<br/> 
### Preprocessing <br/>
We extract the subset of the whole dataset contains 452353
reviews between 30,000 user and 30,000 business and split them to train data
(90%) and test data (10%). you can get two files in the Data/: train review.csv
and test review.csv, each file contain three conlumns: user id, business id, and
stars. And we will use these two files to finish and test our recommendation
system. <br/>

Task1: Model-based CF Algorithms (Spark MLlib) <br/>
Task2: User-based or Item-based CF Algorithm <br/>

### Commands:<br/>
Spark-submit --class ModelBasedCF Recommender_System.jar <rating file path><testing file path><br/>
Spark-submit --class ItemBasedCF Recommender_System.jar <rating file path><testing file path><br/>
