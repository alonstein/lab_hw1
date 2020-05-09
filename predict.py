import argparse
import pandas as pd
import ast
import statistics
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
import pickle


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
df = pd.read_csv(args.tsv_path, sep="\t")

#####
with open('model', 'rb') as f:
    model_dict = pickle.load(f)
m = 10
m_belongs_to_collection = m
m_spoken_languages = m
m_original_language = m
m_production_comapnies = m
m_director = m
m_producer = m
m_cast = m
m_keywords = m
m_title = m
m_overview = m
m_tagline = m
m_country_count = m


def get_category_score(category,score_dict,count_dict,smoothing_factor):
    if not isinstance(category, str):
        return model_dict['average_revenue']
    if category in count_dict and count_dict[category]>=5:
        score = score_dict[category]
        smoothed_score = (score*count_dict[category]+smoothing_factor*model_dict['average_revenue'])/(smoothing_factor+count_dict[category])
    else:
        smoothed_score = model_dict['average_revenue']
    return smoothed_score


def get_multiple_category_score(category_list,score_dict,count_dict,smoothing_factor):
    scores = []
    for category in category_list:
        if category in count_dict and count_dict[category]>=5:
            score = score_dict[category]
            smoothed_score = (score*count_dict[category]+smoothing_factor*model_dict['average_revenue'])/(smoothing_factor+count_dict[category])
        else:
            smoothed_score = model_dict['average_revenue']
        scores.append(smoothed_score)
    return scores


def extract_target_encoding_features(df, column,category_count,category_score,m):
    scores_series = df[column].apply(lambda x:get_multiple_category_score(x,category_score,category_count,m))
    df['min_'+column+'_score'] = scores_series.apply(lambda x:min(x) if len(x)>0 else model_dict['average_revenue'])
    df['max_'+column+'_score'] = scores_series.apply(lambda x:max(x) if len(x)>0 else model_dict['average_revenue'])
    df['mean_'+column+'_score'] = scores_series.apply(lambda x:sum(x)/len(x) if len(x)>0 else model_dict['average_revenue'])
    df['median_'+column+'_score'] = scores_series.apply(lambda x:statistics.median(x) if len(x)>0 else model_dict['average_revenue'])
    return df


# backdrop_path
df = df.drop(['backdrop_path'], axis=1)


# release date:
def get_release_year(date):
    return int(date[:4])


def get_release_month(date):
    return int(date[5:7])


df['release_year'] = df['release_date'].apply(lambda x:get_release_year(x))
df['release_month'] = df['release_date'].apply(lambda x:get_release_month(x))
df = df.drop(['release_date'], axis=1)

# belongs_to_collection
df['no_collection'] = df['belongs_to_collection'].isna().astype(int)
df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x:str(ast.literal_eval(x)['id']) if pd.notnull(x) else x)
df['is_james_bond'] = df['belongs_to_collection'].apply(lambda x:1 if x=='645' else 0)
df['collection_target_encoding'] = df['belongs_to_collection'].apply(lambda x:get_category_score(x,model_dict['collection_score'],model_dict['collection_count'],m_belongs_to_collection))
df = df.drop(['belongs_to_collection'], axis=1)


# spoken languages

def get_languages_list(item):
    item = ast.literal_eval(item)
    languages = [language_dict['iso_639_1'] for language_dict in item]
    return languages


df['spoken_languages'] = df['spoken_languages'].apply(lambda x:get_languages_list(x))
df['spoken_languages_amount'] = df['spoken_languages'].apply(lambda x: len(x))
df['english_speaking'] = df['spoken_languages'].apply(lambda x: 1 if 'en' in x else 0)
df = extract_target_encoding_features(df, 'spoken_languages', model_dict['spoken_languages_counts'], model_dict['spoken_languages_score'], m_spoken_languages)
df = df.drop(['spoken_languages'], axis=1)


# genres
def get_genres_list(item):
    item = ast.literal_eval(item)
    genres = [genre_dict['name'] for genre_dict in item]
    return genres


df['genres'] = df['genres'].apply(lambda x:get_genres_list(x))
genres_by_movie = list(df['genres'])
for genre in model_dict['genres_list']:
    genre_binary = []
    for movie in genres_by_movie:
        if genre in movie:
            genre_binary.append(1)
        else:
            genre_binary.append(0)
    df[genre+'_movie'] = genre_binary
df = df.drop(['genres'], axis=1)


# original language
for language in model_dict['top_languages']:
    df[language+'_original'] = df['original_language'].apply(lambda x: 1 if x == language else 0 )
df['original_language_target_encoding'] = df['original_language'].apply(lambda x:get_category_score(x,model_dict['languages_score'],model_dict['languages_count'],m_original_language))
df = df.drop(['original_language'], axis=1)


# production comapnies
def get_companies_list(item):
    item = ast.literal_eval(item)
    companies = [companies_dict['name'] for companies_dict in item]
    return companies


df['production_companies'] = df['production_companies'].apply(lambda x:get_companies_list(x))
df['production_companies_amount'] = df['production_companies'].apply(lambda x: len(x))
top_copanies = model_dict['top_companies']
top_company_per_movie = []
companies_by_movie = list(df['production_companies'])
for movie in companies_by_movie:
    flag = False
    for company in movie:
        if company in top_copanies:
            flag = True
    top_company_per_movie.append(int(flag))
df['top_production_company'] = top_company_per_movie
df = extract_target_encoding_features(df,'production_companies',model_dict['production_comapnies_count'],model_dict['production_comapnies_score'],m_production_comapnies)
df = df.drop(['production_companies'], axis=1)


# production countries
def get_countries_list(item):
    item = ast.literal_eval(item)
    countries = [countries_dict['name'] for countries_dict in item]
    return countries


df['production_countries'] = df['production_countries'].apply(lambda x:get_countries_list(x))
country_by_movie = list(df['production_countries'])
top_countries = model_dict['top_countries']
top_country_per_movie = []
american_production = []
for movie in country_by_movie:
    flag = False
    american = False
    for country in movie:
        if country in top_countries:
            flag = True
        if country == 'United States of America':
            american = True
    top_country_per_movie.append(int(flag))
    american_production.append(int(american))
df['top_country'] = top_country_per_movie
df['american_production'] = american_production
df = extract_target_encoding_features(df, 'production_countries', model_dict['countries_count'], model_dict['countries_score'], m_country_count)
df = df.drop(['production_countries'], axis=1)


# cast

def get_actors_list(item):
    item = ast.literal_eval(item)
    actors = [actor_dict['id'] for actor_dict in item]
    return actors

df['cast_list'] = df['cast'].apply(lambda x: get_actors_list(x))
df['cast_size'] = df['cast'].apply(lambda x: len(x))
df = extract_target_encoding_features(df,'cast_list',model_dict['cast_counts'],model_dict['cast_score'],m_cast)
df = df.drop(['cast_list','cast'],axis=1)


# crew

def get_crew_member(item, job):
    director = []
    item = ast.literal_eval(item)
    for crew_member in item:
        if crew_member['job'] == job:
            director.append(crew_member['id'])
    return director

df['director'] = df['crew'].apply(lambda x: get_crew_member(x, 'Director'))
df['producer'] = df['crew'].apply(lambda x: get_crew_member(x, 'Producer'))
df['crew_size'] = df['crew'].apply(lambda x: len(ast.literal_eval(x)))

df = extract_target_encoding_features(df,'director',model_dict['director_count'],model_dict['director_score'],m_director)
df = extract_target_encoding_features(df,'producer',model_dict['producer_count'],model_dict['producer_score'],m_producer)
df = df.drop(['crew','director','producer'], axis=1)


stemmer = PorterStemmer()
def text_preprocessing(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = [stemmer.stem(word) for word in sentence.split()]
    return words


# title
df['title'] = df['title'].apply(lambda x:text_preprocessing(x))
df = extract_target_encoding_features(df,'title',model_dict['title_count'],model_dict['title_score'],m_title)
df = df.drop(['title','original_title'],axis=1)

# overview

df['overview'] = df['overview'].replace(np.nan, '', regex=True)
df['overview'] = df['overview'].apply(lambda x:text_preprocessing(x))
df = extract_target_encoding_features(df,'overview',model_dict['overview_count'],model_dict['overview_score'],m_overview)
df = df.drop(['overview'],axis=1)

#tagline
df['no_tagline'] = df['tagline'].isna().astype(int) # Add feature for missing taglines
df['tagline'] = df['tagline'].replace(np.nan, '', regex=True)
df['tagline'] = df['tagline'].apply(lambda x:text_preprocessing(x))
df = extract_target_encoding_features(df,'tagline',model_dict['tagline_count'],model_dict['tagline_score'],m_tagline)
df = df.drop(['tagline'],axis=1)

# homepage
df['no_homepage'] = df['homepage'].isna().astype(int)
df = df.drop('homepage', axis=1)

# runtime
df['runtime'] = df['runtime'].fillna(df['runtime'].mean())

#status
df = df.drop('status',axis=1)

#video
df['video'] = df['video'].astype(int)

# Keywords

def get_Keywords_list(item):
    item = ast.literal_eval(item)
    Keywords = [Keyword_dict['name'] for Keyword_dict in item]
    return Keywords


df['Keywords'] = df['Keywords'].apply(lambda x: get_Keywords_list(x))
df = extract_target_encoding_features(df,'Keywords',model_dict['key_words_count'],model_dict['Keywords_score'],m_keywords)
df = df.drop(['Keywords'],axis=1)

X = df[['budget','popularity','runtime','video','vote_average','vote_count','release_year','release_month',
       'no_collection','is_james_bond','spoken_languages_amount', 'english_speaking',
       'Adventure_movie', 'TV Movie_movie', 'Mystery_movie', 'Fantasy_movie',
       'Science Fiction_movie', 'Family_movie', 'Western_movie', 'War_movie',
       'Animation_movie', 'Documentary_movie', 'Crime_movie', 'Thriller_movie',
       'Music_movie', 'Action_movie', 'Romance_movie', 'History_movie',
       'Horror_movie', 'Comedy_movie', 'Drama_movie', 'en_original',
       'cn_original', 'de_original', 'es_original', 'fr_original',
       'hi_original', 'it_original', 'ja_original', 'ko_original',
       'ru_original', 'zh_original','production_companies_amount','top_production_company','top_country',
       'american_production','cast_size','crew_size','no_tagline','no_homepage','max_Keywords_score','min_spoken_languages_score',
        'collection_target_encoding','min_production_companies_score','mean_title_score','min_tagline_score']]

df = df.fillna(df.mean())
xgboost = model_dict['model']
df['ln_prediction'] = xgboost.predict(X)
df['prediction'] = np.exp(df['ln_prediction'])-1
####

# export prediction results
df[['id', 'prediction']].to_csv("prediction.csv", index=False, header=False)




