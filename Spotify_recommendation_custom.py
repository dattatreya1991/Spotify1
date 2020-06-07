import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


import numpy as np

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

from sklearn.manifold import TSNE

from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

cid ='8d3e3934439b444d9baf6fdf25af1302' # Client ID; copy this from your app
secret = '37fb9d4bbbcb40358d082b7231464f6e' # Client Secret; copy this from your app
username = 'dattatreya1991' # Your Spotify username

scope = 'user-library-read playlist-modify-public playlist-read-private'
#user-library-read playlist-modify-public playlist-read-private
#user-library-read playlist-read-private user-top-read user-follow-read

# https://beta.developer.spotify.com/dashboard/applications/ef60e9f2d37b4913963c7e7e9c572c96
# https://developer.spotify.com/dashboard/applications/8d3e3934439b444d9baf6fdf25af1302
redirect_uri='https://beta.developer.spotify.com/dashboard/applications/ef60e9f2d37b4913963c7e7e9c572c96' # Paste your Redirect URI here

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
token = util.prompt_for_user_token(username, scope, cid, secret, redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)


#1fCOovfyVaAbUiPlqtRF09
#37i9dQZEVXcLXLKYSmx8AY
sourcePlaylistID = '1fCOovfyVaAbUiPlqtRF09'
sourcePlaylist = sp.user_playlist(username, sourcePlaylistID)
tracks = sourcePlaylist["tracks"]
songs = tracks["items"]

track_ids = []
track_names = []

for i in range(0, len(songs)):
    if songs[i]['track']['id'] != None: # Removes the local tracks in your playlist if there is any
        track_ids.append(songs[i]['track']['id'])
        track_names.append(songs[i]['track']['name'])

features = []
for i in range(0,len(track_ids)):
    audio_features = sp.audio_features(track_ids[i])
    for track in audio_features:
        features.append(track)

playlist_df = pd.DataFrame(features, index = track_names)
print(playlist_df.head())

df = playlist_df.copy()
print(df.shape)

print(df.valence)

column_names = df.columns.values.tolist()
print(column_names)
print(type(column_names))


playlist_df=playlist_df[["id", "acousticness", "danceability", "duration_ms",
                         "energy", "instrumentalness",  "key", "liveness",
                         "loudness", "mode", "speechiness", "tempo", "valence"]]



v=TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 6), max_features=10000)
X_names_sparse = v.fit_transform(track_names)
X_names_sparse.shape



#playlist_df['ratings']=[10, 9, 9, 10, 8, 6, 8, 4, 3, 5, 7, 5, 5, 8, 8, 7, 8, 8, 10, 8, 10, 8, 4, 4, 4, 10, 10, 9, 8, 8]
#playlist_df['ratings']=[10, 9, 9, 10, 8, 6, 8, 4, 3, 5, 7, 5, 5, 8, 8, 7, 8, 8, 10, 8, 10, 8, 4, 4, 4, 10, 10, 9, 8, 8, 4]
playlist_df['ratings']=[10, 9, 9, 10, 8, 6, 8, 4, 3, 5, 7, 5, 5, 8, 8, 7, 8, 8, 10, 8, 10, 8, 4, 4, 4, 10, 10, 9, 8, 8, 4,6,7,8,8,9,9,10,4,8,6]
playlist_df.head()




X_train = playlist_df.drop(['id', 'ratings'], axis=1)
y_train = playlist_df['ratings']
forest = RandomForestClassifier(random_state=42, max_depth=5, max_features=12) # Set by GridSearchCV below
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature rankings
print("Feature ranking:")

for f in range(len(importances)):
    print("%d. %s %f " % (f + 1,
            X_train.columns[f],
            importances[indices[f]]))




X_scaled = StandardScaler().fit_transform(X_train)

pca = decomposition.PCA().fit(X_scaled)

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 12)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(8, c='b') # Tune this so that you obtain at least a 95% total variance explained
plt.axhline(0.95, c='r')
plt.show()



pca1 = decomposition.PCA(n_components=8)
X_pca = pca1.fit_transform(X_scaled)



tsne = TSNE(random_state=17)
X_tsne = tsne.fit_transform(X_scaled)


X_train_last = csr_matrix(hstack([X_pca, X_names_sparse])) # Check with X_tsne + X_names_sparse also



# Initialize a stratified split for the validation process
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Decision Trees First
tree = DecisionTreeClassifier()

tree_params = {'max_depth': range(1,11), 'max_features': range(4,19)}

tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)

tree_grid.fit(X_train_last, y_train)
tree_grid.best_estimator_, tree_grid.best_score_


parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42,
                             n_jobs=-1, oob_score=True)
gcv1 = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv1.fit(X_train_last, y_train)
gcv1.best_estimator_, gcv1.best_score_


## KNN

knn_params = {'n_neighbors': range(1, 10)}
knn = KNeighborsClassifier(n_jobs=-1)

knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
knn_grid.fit(X_train_last, y_train)
knn_grid.best_params_, knn_grid.best_score_


# Now build your test set;
# Generate a new dataframe for recommended tracks
# Set recommendation limit as half the Playlist Length per track, you may change this as you like
# Check documentation for  recommendations; https://beta.developer.spotify.com/documentation/web-api/reference/browse/get-recommendations/
rec_tracks = []
for i in playlist_df['id'].values.tolist():
    rec_tracks += sp.recommendations(seed_tracks=[i], limit=int(len(playlist_df)/2))['tracks'];

rec_track_ids = []
rec_track_names = []
for i in rec_tracks:
    rec_track_ids.append(i['id'])
    rec_track_names.append(i['name'])

rec_features = []
for i in range(0,len(rec_track_ids)):
    rec_audio_features = sp.audio_features(rec_track_ids[i])
    for track in rec_audio_features:
        rec_features.append(track)

rec_playlist_df = pd.DataFrame(rec_features, index = rec_track_ids)
rec_playlist_df.head()
rec_playlist_df.shape

X_test_names = v.transform(rec_track_names)
rec_playlist_df=rec_playlist_df[["acousticness", "danceability", "duration_ms",
                         "energy", "instrumentalness",  "key", "liveness",
                         "loudness", "mode", "speechiness", "tempo", "valence"]]


# Make predictions
tree_grid.best_estimator_.fit(X_train_last, y_train)
rec_playlist_df_scaled = StandardScaler().fit_transform(rec_playlist_df)
rec_playlist_df_pca = pca1.transform(rec_playlist_df_scaled)
X_test_last = csr_matrix(hstack([rec_playlist_df_pca, X_test_names]))
y_pred_class = tree_grid.best_estimator_.predict(X_test_last)

rec_playlist_df['ratings']=y_pred_class
rec_playlist_df = rec_playlist_df.sort_values('ratings', ascending = False)
rec_playlist_df = rec_playlist_df.reset_index()

# Pick the top ranking tracks to add your new playlist 9, 10 will work
recs_to_add = rec_playlist_df[rec_playlist_df['ratings']>=9]['index'].values.tolist()

# # No ratings of 9 or 10 this case try adding 8's only
# recs_to_add = rec_playlist_df[rec_playlist_df['ratings']==8]['index'].values.tolist()

# Check what is about to happen :)
print(len(rec_tracks))
print(rec_playlist_df.shape)
print(len(recs_to_add))
print(type(recs_to_add))
#print(recs_to_add[0])

rec_array = np.reshape(recs_to_add, (8, 53))

# Create a new playlist for tracks to add - you may also add these tracks to your source playlist and proceed
playlist_recs = sp.user_playlist_create(username,
                                        name='PCA + tf-idf + DT - Recommended Songs for Playlist - {}'.format(sourcePlaylist['name']))

print(playlist_recs.head())

# Add tracks to the new playlist
for i in rec_array:
#for i in recs_to_add:
    sp.user_playlist_add_tracks(username, playlist_recs['id'], i);
