
# Recommendation System - Last.fm

## Introduction
[GitHub](https://github.com/nachiketmparanjape/Music-Recommender-last.fm)

### Background of the Problem:
For companies which have business in the domains such as media streaming, e-commerce, it is important to know what content its customers might like. Having a recommendation system can help the provide a personalized experience to a customer by making personalized recommendations.
    
In this project, we can leverage customer data from last.fm to build recommendation systems that can provide personalized recommendations using listening history and / or user information such as user age, location, gender, etc.
    
Apart from building a recommendation algorithm, I have also used this project to introduce myself to Apache Spark. I have used ALS tool from pyspark to build a matrix factorization based recommendation algorithm. Apart from this algorithm, I also use k-means clustering to aggregate data based on users’ age, gender and country to make crude recommendation for users without any listening history.

### Who is it useful for?
#### Music Streaming Company
* Better user engagement
* Better user retention
#### Customers
* Personalized Experience
* Ease of use - Auto-curated Playlists / Personalized Radio
* More likely to discover likeable new music (that’s what a good recommendation means!)
#### Artists
* Easier for new artists to be discovered
* Better discovery of music in general
* Components of the Problem

## Data
The data I’m using for this project is collected from a website last.fm. This link contains anonymized user data from the website. Here are some statistics about the dataset

* Total Data Points:  17,559,530
* Unique Users:       359,347
* Artists:            295,015    

The data is in the form of two text (tsv) files.
**Files -**

usersha1-artmbid-artname-plays.tsv

This file contains information about number of times an artist is played by a user

| Variable Name | Datatype | Example                                  |
|--------------:|----------|------------------------------------------|
| userId        | string   | 00000c289a1829a808ac09c00daf10bc3c4e223b |
| artistId      | string   | 3bd73256-3905-4f3a-97e2-8b341527f805     |
| artistName    | string   | betty blowtorch                          |
| plays         | integer  | 2137                                     |

usersha1-profile.tsv

| VariableName | Datatype | Example                                  |
|-------------:|----------|------------------------------------------|
| userId       | string   | 00000c289a1829a808ac09c00daf10bc3c4e223b |
| gender       | string   | f                                        |
| age          | integer  | 22                                       |
| country      | string   | Germany                                  |
| signupDate   | datetime | Feb1,2007                                |


### Uses and Limitations of the data
#### Uses
One datafile contains information of the number of times a user has listened to a song aggregated to the artist-level. So, it is number of times a user who has listened to a particular artist. The other one has information about the customer - gender, age, country and signup date.

Apart from building a recommendation system that I am building, such data can be used to gamify music listening experience to improve engagement by rewarding listeners in some way (badges, points etc.). Also, it can be used to create a social network like platform where you can follow and connect with people with similar musical tastes.
Limitations

This data is aggregated to an artist-level, hence lacks full granularity. Also, it does not contains information on the timeline for every user, so we wouldn’t know if someone has stopped listening to someone and we will not be able to recommend based on recent listening history.

User information can be augmented by getting user’s full digital footprint by linking his account to his accounts on other major platforms which can have a useful information available.

#### Data Cleaning
##### Users’ Data

The data is relatively clean already. The age column has unreasonable values (less than 0 and more than 122) which are replaced by null values and all other columns are kept intact. After that, we delete all rows containing null values in any of their columns (age, gender, country). As we do not have sufficient information to predict the unknown values, we do not have any other option. This only done for a clustering-based recommendation, as ALS does not require this information for making recommendations.

##### Listening Data

This data needs two main cleaning tasks -
1.  Missing artist Ids. Not all artists have artistIds (Approx 2%). For the sake of simplicity, corresponding data is not included in the project.
2.  Duplicate artist names. There are more than one ways in which a single artist is named in the artist column. We need some normalization, regularization and text processing to deal with this issue.

## Preliminary Data Exploration
Here are some interesting visualizations from the data -

**Activity of 20 most active listeners**

![10-most-active-listeners](https://user-images.githubusercontent.com/11637437/31066718-c312f8b6-a703-11e7-958b-1dd55bd0b463.png)

**20 Most Popular Artists**

![10-most-popular-artists](https://user-images.githubusercontent.com/11637437/31066721-c31a4a4e-a703-11e7-8319-95ed0924477e.png)

**Top 20 countries**

![top-countries](https://user-images.githubusercontent.com/11637437/31066714-c306d93c-a703-11e7-811f-14b850e2cca0.png)

**Gender Proportions**

![gender-ratio](https://user-images.githubusercontent.com/11637437/31066715-c306eba2-a703-11e7-9a30-93e0cdc7bceb.png)

**Age Distribution**

![age-distribution](https://user-images.githubusercontent.com/11637437/31066719-c3189960-a703-11e7-9b96-a494432a362f.png)

**Sign-ups over the years**

![signups-over-months-and-years](https://user-images.githubusercontent.com/11637437/31066716-c3075056 a703-11e7-8f98-5ade47ee873a.png)

**Sign-ups per Month-of-the-year**

![signups-per-month-of-the-year](https://user-images.githubusercontent.com/11637437/31066717-c30cfe52-a703-11e7-9f38-979f8d99a3d9.png)

**Sign-ups per Day-of-the-week**

![signups-per-day-of-the-week](https://user-images.githubusercontent.com/11637437/31066713-c300fb02-a703-11e7-877d-ea77e28945ab.png)

## Apache Spark and Collaborative Filtering (ALS)

ALS model in pyspark uses 3 columns rom the data - userId, artistId and plays. In simple terms, we need to construct a matrix with plays as values, userId as rows and artistId as columns, and use already present values to find missing ones. It assumes that there are ‘k’ attributes or features and each artist can be represented by the same. We try to construct the said matrix and minimize a loss function to solve the problem. The loss function used in this case, is alternating least squares.

### Defining a ‘normalized’ plays variable

Plays variable is defined as number of times an artist has been played for a particular user. By its definition, it implies that an older user is more likely to have more number of plays in general. So, to deal with this problem, I have divided the variable by number of days each corresponding user has been active, turning it into a more ‘normalized’ plays/day term. Here is a histogram of the same.

![plays-per-day-hist](https://user-images.githubusercontent.com/11637437/31066720-c31a316c-a703-11e7-83d5-cc9cd108f234.png)

#### Here is a step-by-step information about my implementation of ALS
1.  Used the following data - userId, artistId and plays (plays-per-day)
2.  Substituted userId and artistId by integer values using zipWithIndex() function
3.  Fed the data to the ALS function in Apache Spark which performs matrix factorization and uses ALS to minimize the loss function
4.  Tuned following parameters - rank, regularization parameter, alpha, iterations
5.  Analyzed the performance using 5-fold cross validation and root-mean-squared value
6.  Used the best parameters to train the full dataset
7.  Used the trained model to predict top 20 artists for every user

## K-means Clustering

The user-user based collaborative filtering is not useful for a user who does not have any listening history. It is thus important to be able to make some predictions about their musical preferences, which could be done using information about the user such as their age, gender and country. This could be done by separating the rating information based on these criteria separately and then using rating-mean to find out the top artists for users who fit in the same criteria. Instead of doing that, I have used k-means clustering to cluster the data into an optimum number of clusters. I have then aggregated the rating information among the clusters (mean) to generate a list of top 20 artists for every user.

### Number of Clusters (k)

While using Apache Spark, Sillhoutte Score is not a good metric as it requires calculation of distance between all the datapoint that are present in the dataset which is impractical with millions of datapoints, which is usually the case with Spark. So, to draw the elbow curve, I used Within Set Sum of Squared Distances. It is sum of squares of the distances of every point from its cluster center. Using this information, I created an Elbow Curve and decided the optimum number of clusters to be 10 by qualitative observation. Here is what the chart looks like -

![elbow_curve](https://user-images.githubusercontent.com/11637437/31146772-84d88e00-a83c-11e7-9842-d3763285a1f5.png)

*Thus, ALS could be used to recommend new artists to an existing user, whereas we can use the clustering algorithm to give recommendations to new users with zero or little listening experience.*

*Also, as a future scope for this project, we can also create multiple composite models using different linear combinations of these two models for different use-cases.*
