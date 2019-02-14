from flask import Flask, render_template, jsonify, redirect, request
from random import randint
from sklearn.externals import joblib
import pandas as pd
from scipy.sparse.linalg import svds
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

app = Flask(__name__)

#9066ab23d19e

a=[]
movieids = []
movie = pd.read_csv('moviedata.csv')
movies_data = pd.read_csv('Movies.csv')
user_rating = pd.read_csv('User_rating.csv')

@app.route("/",methods=['GET'])
def index():
    a[:] = []
    movieids[:] = []
    for i in range(0,5):
        movieid = randint(1,193)
        movieids.append(movieid)
        a.append(movies_data.loc[movies_data['MovieID'] == movieid,['Name']].iloc[0]['Name'])
    return render_template("index.html",pred1 = a[0],pred2 = a[1],pred3 = a[2],pred4= a[3],pred5= a[4],name1 = movieids[0],name2 = movieids[1],name3 = movieids[2],name4 = movieids[3],name5 = movieids[4])

@app.route('/get-user-data-item', methods=['POST'])
def predict_stuff_user():
    movie = pd.read_csv('moviedata.csv')
    movies_data = pd.read_csv('Movies.csv')
    user_rating = pd.read_csv('User_rating.csv')

    if request.method == 'POST':
        n1 = request.form.get('id1')
        n2 = request.form.get('id2')
        n3 = request.form.get('id3')
        n4 = request.form.get('id4')
        n5 = request.form.get('id5')
        print n1
        print n2
        print n3
        rating1 = request.form.get('rating1')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': int(n1), 'Rated': rating1}, ignore_index=True)
        rating2 = request.form.get('rating2')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': int(n2), 'Rated': rating2}, ignore_index=True)
        rating3 = request.form.get('rating3')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': int(n3), 'Rated': rating3}, ignore_index=True)
        rating4 = request.form.get('rating4')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': int(n4), 'Rated': rating4}, ignore_index=True)
        rating5 = request.form.get('rating5')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': int(n5), 'Rated': rating5}, ignore_index=True)
        #ITEM ITEM
        max = user_rating.loc[user_rating['UserID'] == 1000, ['Rated']]
        max2 = max['Rated'].astype(float).idxmax()
        max_movie_id = user_rating.iloc[max2]['MovieID']
        max_movie_name1 = movies_data.loc[movies_data['MovieID'] == max_movie_id, ['Name']]
        print ("afa")
        print max_movie_name1
        max_movie_name = max_movie_name1.iloc[0]['Name']
        movie_data = pd.merge(movies_data,user_rating, on='MovieID')
        user_movie_rating = movie_data.pivot_table(index='UserID', columns='Name', values='Rated',aggfunc='first').astype(float)
        ratings = user_movie_rating[str(max_movie_name)].astype(float)
        movies_like = user_movie_rating.corrwith(ratings,method='pearson')
        corr= pd.DataFrame(movies_like, columns=['Correlation'])
        corr.dropna(inplace=True)
        corr = corr.sort_values('Correlation', ascending=False).head(6)
        recommends = list(corr.index)
        recommends.pop(0)
        return render_template("recommend.html",title = "Item-Item",movie_name = max_movie_name,rec1 = recommends[0],rec2 = recommends[1],rec3 = recommends[2],rec4 = recommends[3],rec5 = recommends[4])

@app.route('/get-user-data-user', methods=['POST'])
def predict_stuff_item():
    movie = pd.read_csv('moviedata.csv')
    movies_data = pd.read_csv('Movies.csv')
    user_rating = pd.read_csv('User_rating.csv')
    if request.method == 'POST':
        rating1 = request.form.get('rating1')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': movieids[0], 'Rated': rating1}, ignore_index=True)
        rating2 = request.form.get('rating2')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': movieids[1], 'Rated': rating2}, ignore_index=True)
        rating3 = request.form.get('rating3')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': movieids[2], 'Rated': rating3}, ignore_index=True)
        rating4 = request.form.get('rating4')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': movieids[3], 'Rated': rating4}, ignore_index=True)
        rating5 = request.form.get('rating5')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': movieids[4], 'Rated': rating5}, ignore_index=True)
                #USER USER1
        movie_data = pd.merge(movies_data,user_rating, on='MovieID')
        movie_user_rating = movie_data.pivot_table(index= 'Name',columns = 'UserID',values='Rated',aggfunc= 'first').astype(float)
        user_rating = movie_user_rating[1000].astype(float)
        user_like_user = movie_user_rating.corrwith(user_rating,method='pearson')
        corr_user = pd.DataFrame(user_like_user, columns=['Correlation'])
        corr_user.dropna(inplace=True)
        corr_user = corr_user.sort_values('Correlation', ascending=False).head(10)
        similar_user = list(corr_user.index)[1]
        similar_user_rating = movie_user_rating[similar_user]
        user_rating.dropna(inplace=True)
        similar_user_rating.dropna(inplace=True)
        similar_user_rating = similar_user_rating.sort_values(ascending=False).head(6)
        recommended_movies = list(similar_user_rating.index)
        recommended_movies.pop(0)
        return render_template("recommend.html",similar_user = similar_user,rating_1=similar_user_rating[0],rating_2=similar_user_rating[1],rating_3=similar_user_rating[2],rating_4=similar_user_rating[3],rating_5=similar_user_rating[5],title = "User-User",rec1 = recommended_movies[0],rec2 =recommended_movies[1],rec3 = recommended_movies[2],rec4 = recommended_movies[3],rec5 = recommended_movies[4])

@app.route('/get-user-data-matrix',methods=['Post'])
def predit_stuff_matrix():
    movie = pd.read_csv('moviedata.csv')
    movies_data = pd.read_csv('Movies.csv')
    user_rating = pd.read_csv('User_rating.csv')
    if request.method == 'POST':
        rating1 = request.form.get('rating1')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': movieids[0], 'Rated': rating1}, ignore_index=True)
        rating2 = request.form.get('rating2')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': movieids[1], 'Rated': rating2}, ignore_index=True)
        rating3 = request.form.get('rating3')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': movieids[2], 'Rated': rating3}, ignore_index=True)
        rating4 = request.form.get('rating4')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': movieids[3], 'Rated': rating4}, ignore_index=True)
        rating5 = request.form.get('rating5')
        user_rating = user_rating.append({'UserID': 1000, 'MovieID': movieids[4], 'Rated': rating5}, ignore_index=True)
        movie_data = pd.merge(movies_data,user_rating, on='MovieID')
        user_movie_rating = movie_data.pivot_table(index='UserID', columns='Name', values='Rated',aggfunc='first')
        # MATRIX decomposition
        matrix_df = user_rating.pivot_table(index = 'UserID',columns ='MovieID', values = 'Rated',aggfunc='first').astype(float).fillna(0)
        R = matrix_df.as_matrix()
        user_ratings_mean = np.mean(R, axis = 1)
        matrix_de_mean = R - user_ratings_mean.reshape(-1, 1)
        U, sigma, Vt = svds(matrix_de_mean, k = 50)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        prediction_df = pd.DataFrame(all_user_predicted_ratings, columns = user_movie_rating.columns)
        def recommend_movies(predictions_df, userID, movies_df, users_ratings_df, num_recommendations=5):

            user_number = userID - 1
            sorted_predictions = predictions_df.iloc[user_number].sort_values(ascending=False)

            user_data = users_ratings_df[users_ratings_df.UserID == (userID)]
            user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                             sort_values(['Rated'], ascending=False))

            recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
                 merge(pd.DataFrame(sorted_predictions).reset_index(), how = 'left').
                 rename(columns = {user_number: 'Predictions'}).
                 sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1])
            return user_full, recommendations

        already_rated, predictions = recommend_movies(prediction_df, 1000, movies_data,user_rating, 20)
        recommended_movies = list(predictions['Name'].head(5))
        return render_template("recommend.html",title = "Matrix decomposition",rec1 = recommended_movies[0],rec2 =recommended_movies[1],rec3 = recommended_movies[2],rec4 = recommended_movies[3],rec5 = recommended_movies[4])

if __name__ == "__main__":
    movie = pd.read_csv('moviedata.csv')
    movies_data = pd.read_csv('Movies.csv')
    user_rating = pd.read_csv('User_rating.csv')
    app.run(host='0.0.0.0')
