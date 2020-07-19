import os
import time
import gc
import argparse

# data science imports
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
# utils import
from fuzzywuzzy import fuzz


#Building and training the model
from sklearn.neighbors import KNeighborsClassifier

#调入 sklearn训练数据和测试数据的分割方法 Splitthe datasetinto train and test data
#下面没有使用，直接使用的random() method
from sklearn.model_selection import train_test_split
#k-Fold Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


class KnnRecommender:
    """
    This is an item-based collaborative filtering recommender with
    KNN implmented by sklearn
    """
    #定义初始方法 path_movies, path_ratings等参数传入
    def __init__(self, path_movies, path_ratings):

        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.movie_rating_thres = 0
        self.user_rating_thres = 0
        #这一步的操作就直接把model定义成了 NearestNeighbors类型，下面的model已经转换
        self.model = NearestNeighbors()

        self.train_data, self.test_data = self._prep_data()


    def set_filter_params(self, movie_rating_thres, user_rating_thres):
        """
        movie_rating_thres: int, minimum number of ratings received by users
        user_rating_thres: int, minimum number of ratings a user gives
        """
        #设置额定频率阈值以过滤不知名的电影（考虑前25%的电影，并且作为阈值）和不太活跃的用户（用户限制在前40％），
        self.movie_rating_thres = movie_rating_thres
        self.user_rating_thres = user_rating_thres

    def set_model_params(self, n_neighbors, algorithm, metric, n_jobs=None):
        """
        这一步作用：
        将NearestNeighbors类初始化为model_knn并将稀疏矩阵适合该实例。
        通过指定metric = cosine，模型将通过使用余弦相似度来测量艺术家矢量之间的相似度。
        设置sklearn.neighbors.NearestNeighbors的模型参数，参量
        ----------
        n_neighbors: int, optional (default = 5)
        algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        metric: string or callable, default 'minkowski', or one of
            ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        n_jobs: int or None, optional (default=None)
        """
        if n_jobs and (n_jobs > 1 or n_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algorithm,
            'metric': metric,
            'n_jobs': n_jobs})

    #装载数据
    def _prep_data(self):
        """
        prepare data for recommender
        1. movie-user scipy sparse matrix
        2. hashmap of movie to row index in movie-user scipy sparse matrix
        """
        # 读入数据，表之间的连接和处理
        df_movies = pd.read_csv(
            os.path.join(self.path_movies),
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})
        df_ratings = pd.read_csv(
            os.path.join(self.path_ratings),
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        # filter data
        df_movies_cnt = pd.DataFrame(
            df_ratings.groupby('movieId').size(),
            columns=['count'])
        popular_movies = list(set(df_movies_cnt.query('count >= @self.movie_rating_thres').index))  # noqa
        movies_filter = df_ratings.movieId.isin(popular_movies).values

        df_users_cnt = pd.DataFrame(
            df_ratings.groupby('userId').size(),
            columns=['count'])
        active_users = list(set(df_users_cnt.query('count >= @self.user_rating_thres').index))  # noqa
        users_filter = df_ratings.userId.isin(active_users).values

        df_ratings_filtered = df_ratings[movies_filter & users_filter]

        # pivot and create movie-user matrix
        movie_user_mat = df_ratings_filtered.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)
        # create mapper from movie title to index
        '''
        hashmap = {
            movie: i for i, movie in
            enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title)) # noqa
        }
        # transform matrix to scipy sparse matrix 将矩阵转换为稀疏矩阵
        '''
        # 这里开始把上面的数据转化为稀疏矩阵，由于将要执行线性代数运算
        movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

        # clean up
        del df_movies, df_movies_cnt, df_users_cnt
        del df_ratings, df_ratings_filtered, movie_user_mat
        gc.collect()

        #train和test数据集的建立 数据的分成 train70% 和 test30%  random隨機選一下
        train_data,test_data = movie_user_mat_sparse.randomSplit([0.7, 0.3])
        # 注意：上面movie_user_mat_sparse已经转化成了矩阵，这里train和test依然是矩阵
        return train_data, test_data

    #添加代码
    #定义交叉验证
    def Cross_validation(self):
        #数据使用train70%，执行交叉验证
        #为了直接找到最优的 n_neighbors 也就是 K值，直接使用 GridSearchCV 这个超调参数
        #GridSearchCV的工作原理是在我们指定的参数范围内多次训练我们的模型。这样，我们可以用每个参数来测试我们的模型，
        # 并找出最优值，以获得最佳的精度结果。超调参数找到您模型的最佳参数以提高准确性

        #create new a knn model
        knn2 = KNeighborsClassifier()
        #create a dictionary of all values we want to test for n_neighbors
        param_grid = {'n_neighbors': np.arange(1, 25)}
        #use gridsearch to test all values for n_neighbors
        #我们使用网格搜索的新模型将采用新的k-NN分类器，即param_grid和交叉验证值5，
        # 以便找到“ n_neighbors”的最佳值
        knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

        # fit model to data
        # 這裡的X,Y 从上面的train_data，使用scikit_learn的方法得到
        X, Y = train_test_split(self.train_data, test_size=0.2, random_state=1, stratify=y)
        # fit model to data
        knn_gscv.fit(X, Y)

        #check which of our values for ‘n_neighbors’ that we tested performed the best.
        # knn_gscv.best_params_会返回一个字典格式的数据{n_nerghbors: 14}
        # 最合适的K,就是個字典的value值
        k = knn_gscv.best_params_.value
        return k

    #再添加两个方法
    def accuracy(self):

        # 下面这一步：
        # check mean score for the top performing value of n_neighbors
        # best_score_’输出通过交叉验证获得的分数的平均准确性
        knn2 = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(1, 25)}
        knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
        knn_gscv.fit(self.train_data, self.test_data)
        # 下面这一步：
        # check mean score for the top performing value of n_neighbors
        # best_score_’输出通过交叉验证获得的分数的平均准确性
        accuracy = knn_gscv.best_score_
        return accuracy


    #下面开始做电影推荐，同等结构直接转化为目前项目，家具的推荐，有待修改相应的条件和参数
    def _fuzzy_matching(self, hashmap, fav_movie):
        """
        return the closest match via fuzzy ratio.
        If no match found, return None
        Parameters
        ----------
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        Return
        ------
        index of the closest match
        """
        match_tuple = []
        # get match
        for title, idx in hashmap.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    def _inference(self, model, data, hashmap,
                   fav_movie, n_recommendations):
        """
        return top n similar movie recommendations based on user's input movie
        Parameters 根据用户输入的电影返回前n个相似的电影推荐参量
        ----------
        model: sklearn model, knn model
        data: movie-user matrix
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar movie recommendations
        """
        # fit
        model.fit(data)
        # get input movie index
        print('You have input movie:', fav_movie)
        idx = self._fuzzy_matching(hashmap, fav_movie)
        # inference
        print('Recommendation system start to make inference')
        print('......\n')
        t0 = time.time()
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)
        # get list of raw idx of recommendations
        raw_recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        # return recommendation (movieId, distance)
        return raw_recommends

    def make_recommendations(self, fav_movie, n_recommendations):
        """
        make top n movie recommendations
        Parameters
        ----------
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations
        """
        # get data
        movie_user_mat_sparse, hashmap = self._prep_data()
        # get recommendations
        raw_recommends = self._inference(
            self.model, movie_user_mat_sparse, hashmap,
            fav_movie, n_recommendations)
        # print results
        reverse_hashmap = {v: k for k, v in hashmap.items()}
        print('Recommendations for {}:'.format(fav_movie))
        for i, (idx, dist) in enumerate(raw_recommends):
            print('{0}: {1}, with distance '
                  'of {2}'.format(i+1, reverse_hashmap[idx], dist))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Movie Recommender",
        description="Run KNN Movie Recommender")
    parser.add_argument('--path', nargs='?', default='../KNN/Data',
                        help='input data path')
    parser.add_argument('--movies_filename', nargs='?', default='movies.csv',
                        help='provide movies filename')
    parser.add_argument('--ratings_filename', nargs='?', default='ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--movie_name', nargs='?', default='',
                        help='provide your favoriate movie name')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n movie recommendations')
    return parser.parse_args()

def test(best_K, train_set, test_set):

    '''# In order to train and test our model using cross-validation,
        # we will use the ‘cross_val_score’ function with a cross-validation
        # value of 5. ‘cross_val_score’ takes in our k-NN model and our data as
        # parameters. Then it splits our data into 5 groups and fits and
        # scores our data 5 seperate times, recording the accuracy score
        # in an array each time. We will save the accuracy scores in the ‘cv_scores’ variable.
    '''
    knn_cv = KNeighborsClassifier(best_K)
    cv_scores = cross_val_score(knn_cv, train_set, test_set, cv=5)
    return cv_scores


def predict(best_K):
    arrary = KNeighborsClassifier(best_K).predict(test_set)[0:10]
    return arrary



if __name__ == '__main__':
    # get args
    args = parse_args()
    data_path = args.path
    movies_filename = args.movies_filename
    ratings_filename = args.ratings_filename
    movie_name = args.movie_name
    top_n = args.top_n
    # initial Recommence system
    recommender = KnnRecommender(
        os.path.join(data_path, movies_filename),
        os.path.join(data_path, ratings_filename))
    # set params 42-52 ，把（50,50）賦值到裡面
    recommender.set_filter_params(50, 50)

    #需要交叉驗證需要找到的最合適的K,所以改成变量best_K
    #再实例化,为了使用最合适的K值，调用类里面的Cross_validation method
    Myrecommender = KnnRecommender
    best_K = Myrecommender.Cross_validation
    #这一步操作就是使用最合适的K,生成矩阵
    train_set, test_set = Myrecommender._prep_data

    #开始测试经过训练得到的准确性的列表
    accuracy_scores = test(best_K, train_set, test_set)

    # 调用recommender中accuracy的方法，得到最终的准确度
    accuracy = recommender.accuracy

    #前10次推荐是否可以推荐成功，1表示成功，0表示失败
    Target_arrary = predict(best_K)



    recommender.set_model_params(best_K, 'brute', 'cosine', -1)
    # 调用calss 里面make_recommendations的方法完成推荐
    recommender.make_recommendations(movie_name, top_n)



#####
#1. predict和score的定义在哪？
#2. CV使用了哪种evaluation metric(评估指标)来评判最好的参数？ 解决，直接使用 GridSearchCV 这个超调参数
#3. test的准确率怎么体现的？使用了那种evaluation metric(评估指标)? 和CV一致吗

# 1. https://github.com/benfred/implicit
# 2. https://implicit.readthedocs.io/en/latest/index.html
# 3. Rich的代码
# 4. KNN 对隐性数据的处理
# 5. test 数据的评价 Evaluation  还要添加跟与原来的差距， evaliztion的包
# 6. https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/
# 7. https://jessesw.com/Rec-System/

# David7:
# reference md当中加入pyspark tutorial
#
# David7:
# how-to中加入pyspark sql和RDD的cheat sheet
#
# David7:
# 请大家 学习pyspark的基础知识
