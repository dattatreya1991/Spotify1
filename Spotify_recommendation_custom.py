import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd

cid ='8d3e3934439b444d9baf6fdf25af1302' # Client ID; copy this from your app
secret = '37fb9d4bbbcb40358d082b7231464f6e' # Client Secret; copy this from your app
username = 'dattatreya1991' # Your Spotify username

scope = 'user-library-read playlist-modify-public playlist-read-private'
#user-library-read playlist-modify-public playlist-read-private
#user-library-read playlist-read-private user-top-read user-follow-read

redirect_uri='https://developer.spotify.com/dashboard/applications/8d3e3934439b444d9baf6fdf25af1302' # Paste your Redirect URI here

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
token = util.prompt_for_user_token(username, scope, cid, secret, redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)



sourcePlaylistID = '37i9dQZEVXcLXLKYSmx8AY'
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

column_names = df.columns.values.tolist()
print(column_names)
print(type(column_names))
