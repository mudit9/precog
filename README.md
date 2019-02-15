# Movie Recommender System

Precog task by - Mudit Saxena (Shiv Nadar University)

The Webapp is deployed on heroku: https://precog-movierecommender.herokuapp.com/

Technologies used: Flask, Python, Pandas, Numpy, Scipy, MongoDB, Docker

![heroku](/screenshots/heroku.png?raw=true)


## Start App

1. Install all dependencies using requirements.txt

```
$  pip install -r requirements.txt
```

2. Start app

```
$  python app.py
```

3. Open localhost:5000

## Methodology

### 1. Data Extraction
To extract the IMDB data of the movies, a web scraper was used. For each genre, (action, comedy, drama, sci-fi, animation, thriller) 50 movies were scraped. Movie details (Name, Year of release, Rating, Genre, IMDB_Url, votes) were scraped off the IMDB sort by genre page. Code can be found in [DataExtraction.ipynb](../Python Notebooks/DataExtraction.ipynb).
Note - TV series is being treated as movies only.

```
df = pd.read_csv('imdb.csv')
df
```

![IMDB](/screenshots/imdb.png?raw=true)

To generate the dummy data for the user ratings, 999 random users were created and each user gave 20 random movies a random rating out from 1 to 5.

![UserData](/screenshots/user_rating.png?raw=true)


### 2. Data Pre-processing

#### IMDB data
After scraping data for 300 movies in IMDB, duplicate movies were removed. The movies dataset now had 193 unique movies.

![duplicate](/screenshots/shape.png?raw=true)


#### MongoDB
To store the movie instances into a MongoDB collection named 'moviedata', python's pyMongo library was used.

![MongoDB-Import](/screenshots/mongodb_import.png?raw=true)

And exporting the mondoDB database to a csv file:

![MongoDB-Export](/screenshots/mongoDB_export.png?raw=true)

#### User Data

Tuples where a user has rated the same movie twice or more than twice were removed from the dataset.

#### Movie-User Utility Matrix

A dataframe where all the movies are on the x axis and all the users on the y axis and each column represents the movies vector i.e the movie-user[i][j] is the value which the jth user has given to ith movie. NaN implies that jth user has not rated ith movie.

![User-Movie Data](/screenshots/userUser_matrixUtlity.png?raw=true)

#### User-Movie Utility Matrix

A dataframe where all the users are on the x axis and all the movies on the y axis and each column represents the user vector
i.e the user-movie[i][j] is the value which the Ith user has given to jth movie. NaN implies that ith user has not rated  jth movie.

![Movie-User Data](/screenshots/movie_user.png?raw=true)

### 3. Collaborative Filtering

```
Idea: If a person A likes item 1, 2, 3 and B like 2,3,4 then they have similar interests and A should like item 4 and B should like item 1.
```

The co-efficient I have used to calculate similarity between 2 vectors is Pearson similarity Co-efficient. Pearson correlation coefficient is applicable when the data is not normalised which is good in our case. Pearson correlation co-efficient is calculated from the formula:

![pearson](/screenshots/pearson.png?raw=true)

Pandas library in python has a builtin function ``corrwith()`` which calculates the pearson correlation between 2 vectors.

Also, to avoid cold-start problem, each new user has to rate at least 5 movies to get a correlation through which another user can be called a similar user.

### Correlation

The Pandas function ``corrwith()`` returns a dataframe of other all other columns with the respective correlation. Had the user-rating dataset been real and with actual ratings of real people, the correlation would be much higher.
For example:

![Correlation](/screenshots/correlation.png?raw=true)

### Item-Item

In item-item collaborative filtering, we recommend movies which have been rated similarly to the movies which have been rated high by the new user.
In my approach, if the new user has rated all movies poorly, (s)he'd be recommended movies which have the highest IMDB ratings.
For step-by-step explanation, follow comments.

![Item-Item Collaborative filtering](/screenshots/itemitem.png?raw=true)

### User-User

Here we find users which are alike(based on similarity) and recommend movies which the most similar user to the new user has rated high before.
First, the most similar user is found out using pearson similarity coefficient and then the top most rated movies of the most similar movies are retrieved. If there is any movie in the recommended movies list which has already been rated by the new user, it is removed from the recommended movies list.
For step-by-step explanation, follow comments.


![User-User Collaborative Filtering](/screenshots/useruser.png?raw=true)

### Matrix Decomposition
In Matrix Decomposition using singular value decomposition the formula used is: R=UΣV <sup>T</sup> where R is the user_rating matrix, U is the user matrix, Σ is the diagonal matrix with singular values and V<sup>T</sup> is the movie matrix.
For step-by-step explanation, follow comments.

![Matrix Decomposition](/screenshots/MatrixDecomposition.png?raw=true)


### 4. Docker

The DockerFile with the OS Ubuntu 16.04 has been added to the repository.

#### External Resources
1. https://www.analyticsvidhya.com
2. https://stackabuse.com
3. https://stackoverflow.com
4. https://beckernick.github.io
