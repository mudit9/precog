# Movie Recommendation System
Webapp: https://precog-movierecommender.herokuapp.com/

Technologies used: Flask 0.12.2, Python, pandas, scipy

## How to start up app


2. Install all dependencies using requirements.txt

```
$  pip install -r requirements.txt
```

3. Start app

```
$  python app.py
```

## Methodology

### 1. Data Extraction
To extract the IMDB data of the movies, a web scraper was used. For each genre, (action, comedy, drama, sci-fi, animation, thriller) 50 movies were scraped. Movie details (Name, Year of release, Rating, Genre, IMDB_Url, votes) were scraped off the IMDB sort by genre page. Code can be found in : [DataExtraction.ipynb](DataExtraction.ipynb).
```
df = pd.read_csv('imdb.csv')
df
```

![IMDB](/screenshots/imbd.png?raw=true)


### 2. Data Processing

#### IMDB data
After scraping data for 300 movies in IMDB, duplicate movies were removed. The movies dataset now had 193 unique movies.

#### MongoDB
To store the movie instances into a MongoDB collection named 'moviedata', python's pyMongo library was used.

![MongoDB-Import](/screenshots/mongodb_import.png?raw=true)

And exporting the mondoDB database to a csv file:

![MongoDB-Export](/screenshots/mongoDB_export.png?raw=true)

#### User Data
Further to generate the dummy data for the user ratings, 999 random users were created and each user gave a random movie a random rating out 1 to 5.

![UserData](/screenshots/user_rating.png?raw=true)
