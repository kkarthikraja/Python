%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
import numpy as np
from numpy import random

#load data (make sure you have downloaded database.sqlite)
with sqlite3.connect('../input/database.sqlite') as con:
    countries = pd.read_sql_query("SELECT * from Country", con)
    matches = pd.read_sql_query("SELECT * from Match", con)
    leagues = pd.read_sql_query("SELECT * from League", con)
    teams = pd.read_sql_query("SELECT * from Team", con)
    Player_detail = pd.read_sql_query("SELECT * from Player_Attributes", con)
    Player = pd.read_sql_query("SELECT * from Player", con)
    

sns.set()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
Player_detail.shape

Player_detail.head()

fig ,axes = plt.subplots(nrows=2,ncols=2,figsize=(20,14))

attack_work_rate = Player_detail[Player_detail['attacking_work_rate'].isin(['medium','high','low'])]
defence_work_rate = Player_detail[Player_detail['defensive_work_rate'].isin(['medium','high','low'])]


sns.histplot(attack_work_rate['attacking_work_rate'],ax=axes[0,0],color='green')
sns.histplot(defence_work_rate['defensive_work_rate'],ax=axes[0,1])
sns.histplot(Player_detail['overall_rating'],ax=axes[1,0],color='red')
sns.histplot(Player_detail['potential'],ax=axes[1,1],color='orange')

Players = pd.merge(Player_detail,Player,on='player_api_id',how='left')

print(Player.shape)
print(Player_detail.shape)
print(Players.shape)

ballondr = Players[Players['player_name'].isin(['Lionel Messi','Cristiano Ronaldo','Luka Modric','Shinji Kagawa'])]
ballondr.head()

overall = ballondr[['overall_rating','player_name']].groupby('player_name').mean()
overall

Players['overall_rating'].describe()

from datetime import date

japan = Players[Players['player_name'].isin(['Shinji Kagawa','Atsuto Uchida','Maya Yoshida','Keisuke Honda','Makoto Hasebe','Yuto Nagatomo'])]
japan[['date','birthday']] = japan[['date','birthday']].apply(pd.to_datetime)
japan['age'] = (japan['date'] - japan['birthday']).dt.days // 365
japan = japan[japan['age']>20]
japan.head()

japan_rate = japan[['player_name','age','overall_rating']].groupby(['age','player_name']).mean().unstack()
japan_rate.columns = ['Atsuto Uchida','Keisuke Honda','Makoto Hasebe','Maya Yoshida','Shinji Kagawa','Yuto Nagatomo']
japan_rate


ax = japan_rate.plot(figsize=(16,8),marker='o')
plt.title("japanese ratting in 2008-2016")

plt.xlabel("age")


japan = Players[Players['player_name'].isin(['Shinji Kagawa','Atsuto Uchida','Maya Yoshida','Keisuke Honda','Makoto Hasebe','Yuto Nagatomo','Gotoku Sakai','Hiroki Sakai','Shinji Okazaki','Hiroshi Kiyotake','Yuya Osako'])]
japan_max_ratio = japan[['player_name','overall_rating']].groupby('player_name').max()
japan_max_ratio['max'] = 1
japan = pd.merge(japan,japan_max_ratio,on=['player_name','overall_rating'],how='inner')
japan = japan.drop_duplicates(subset=['player_name']).reset_index(drop=True)
japan['country'] = 'japan'



spain = Players[Players['player_name'].isin(['Gerard Pique','Sergio Ramos','Sergio Busquets','Xavi Hernandez','Andres Iniesta','David Silva','Fernando Torres','David Villa','Juan Mata','Jordi Alba','Cesc Fabregas'])]
spain_max_ratio = spain[['player_name','overall_rating']].groupby('player_name').max()
spain_max_ratio['max'] = 1
spain = pd.merge(spain,spain_max_ratio,on=['player_name','overall_rating'],how='inner')
spain = spain.drop_duplicates(subset=['player_name']).reset_index(drop=True)
spain['country'] = 'spain'


germany = Players[Players['player_name'].isin(['Mats Hummels','Jerome Boateng','Shkodran Mustafi','Philipp Lahm','Bastian Schweinsteiger','Toni Kroos','Mesut Oezil','Lukas Podolski','Miroslav Klose','Sami Khedira','Julian Draxler'])]
germany_max_ratio = germany[['player_name','overall_rating']].groupby('player_name').max()
germany_max_ratio['max'] = 1
germany = pd.merge(germany,germany_max_ratio,on=['player_name','overall_rating'],how='inner')
germany = germany.drop_duplicates(subset=['player_name']).reset_index(drop=True)
germany['country'] = 'germany'


countries_player = pd.concat([japan,spain,germany],axis=0)
countries_player = countries_player[['player_name','overall_rating', 'potential','crossing', 'finishing','heading_accuracy',
                                    'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
                                    'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
                                    'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
                                    'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
                                    'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle','height', 'weight']]
countries_player = countries_player.reset_index(drop=True)
countries_player = countries_player.set_index('player_name')
countries_player.shape


from sklearn import preprocessing

countries_player_scale = preprocessing.scale(countries_player)

countries_player_scale = pd.DataFrame(countries_player_scale)
countries_player_scale.columns = countries_player.columns
countries_player_scale.index = countries_player.index
countries_player_scale.head()


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

Vec = KMeans(n_clusters=3)
group_num = Vec.fit_predict(countries_player_scale)


sse = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(countries_player_scale)
    
    sse.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(sse)


range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(countries_player_scale)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(countries_player_scale, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


player_clus = countries_player_scale.copy()
player_clus['group'] = group_num

player_clus = player_clus[['overall_rating','finishing','short_passing','free_kick_accuracy',
                           'height','weight','positioning','agility','dribbling','heading_accuracy','group']].groupby('group').mean()
player_clus = player_clus.T
player_clus


player_clus[0:].plot(figsize=(12,10),kind='barh',subplots=True,layout=(1,4),sharey=True)

from sklearn.decomposition import TruncatedSVD

model_svd = TruncatedSVD(n_components=2)
vecs_list = model_svd.fit_transform(countries_player_scale)

X = vecs_list[:,0]
Y = vecs_list[:,1]

plt.figure(figsize=(16,16))

color_codes = {0:'blue',1:'orange',2:'green'}
colors = [color_codes[x] for x in group_num]
plt.scatter(X,Y,color=colors)

for i,(x_name,y_name) in enumerate(zip(X,Y)):
    plt.annotate(countries_player_scale.index[i],(x_name,y_name))
plt.show()

from scipy.cluster.hierarchy import linkage,dendrogram,fcluster

linkage_result = linkage(countries_player_scale,method='ward',metric='euclidean')

sns.set()
dendrogram(linkage_result,labels=countries_player_scale.index)
plt.rcParams['figure.figsize'] = (20 ,10)
fig = plt.Figure(figsize=(20,10))
plt.show()


cluster_labels = cut_tree(linkage_result, n_clusters=3).reshape(-1, )
cluster_labels


countries_player_scale['Cluster_Labels'] = cluster_labels
countries_player_scale.head()


countries_player_scale = countries_player_scale[['overall_rating','finishing','short_passing','free_kick_accuracy',
                           'height','weight','positioning','agility','dribbling','heading_accuracy','Cluster_Labels']].groupby('Cluster_Labels').mean()
countries_player_scale = countries_player_scale.T
countries_player_scale



countries_player_scale[0:].plot(figsize=(12,10),kind='barh',subplots=True,layout=(1,4),sharey=True)

countries

matches.head()


leagues


teams.head()


print(countries.shape)
print(matches.shape)
print(leagues.shape)
print(teams.shape)


#select relevant countries and merge with leagues

selected_countries = ['England','France','Germany','Italy','Spain','Belgium']

countries = countries[countries.name.isin(selected_countries)]

leagues = countries.merge(leagues,on='id',suffixes=('', '_y'))


#select relevant fields

matches = matches[matches.league_id.isin(leagues.id)]

matches = matches[['id', 'country_id' ,'league_id', 'season', 'stage', 'date','match_api_id', 'home_team_api_id', 'away_team_api_id','B365H', 'B365D' ,'B365A']]

matches.dropna(inplace=True)

matches.head()

from scipy.stats import entropy



def match_entropy(row):

    odds = [row['B365H'],row['B365D'],row['B365A']]

    #change odds to probability

    probs = [1/o for o in odds]

    #normalize to sum to 1

    norm = sum(probs)

    probs = [p/norm for p in probs]

    return entropy(probs)



#compute match entropy

matches['entropy'] = matches.apply(match_entropy,axis=1)
matches.head()


#compute mean entropy for every league in every season

entropy_means = matches.groupby(['season','league_id']).entropy.mean()

entropy_means = entropy_means.reset_index().pivot(index='season', columns='league_id', values='entropy')

entropy_means.columns = [leagues[leagues.id==x].name.values[0] for x in entropy_means.columns]

entropy_means.head(10)


#plot graph
ax = entropy_means.plot(figsize=(12,8),marker='o')

#set title
plt.title('Leagues Predictability', fontsize=16)

#set ticks roatation
plt.xticks(rotation=50)

#keep colors for next graph
colors = [x.get_color() for x in ax.get_lines()]
colors_mapping = dict(zip(leagues.id,colors))

#remove x label
ax.set_xlabel('')

#locate legend 
plt.legend(loc='lower left')

#add arrows
ax.annotate('', xytext=(7.2, 1),xy=(7.2, 1.039),
            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)

ax.annotate('', xytext=(7.2, 0.96),xy=(7.2, 0.921),
            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)

ax.annotate('less predictable', xy=(7.3, 1.028), annotation_clip=False,fontsize=14,rotation='vertical')
ax.annotate('more predictable', xy=(7.3, 0.952), annotation_clip=False,fontsize=14,rotation='vertical')


