# Number of records retrieved using the API
num = 10

# Required packages
import base64
from  requests import get, post
import json
import pandas as pd

print("Packages imported")

# Access token
client_id = 'e21ae8429f404455a547adf8acd0f6d3'
client_secret = 'ae5e5fcfd32448c6acc2055316eeb6d2'

print("access")

# Function to get token and authorization
def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8") # Access credentials

    url = "https://accounts.spotify.com/api/token" # API URL to request access
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content) # Permissions to access API data
    token = json_result["access_token"]
    return token # Token is returned

# Function to get authorization
def get_auth_header(token):
    return {"Authorization": "Bearer " + token} # Authorization of token

# Function to search for track id
def search_for_track_id(token, track_name):
    url = "https://api.spotify.com/v1/search" # API URL for searching the track id
    header = get_auth_header(token)
    query = f"?q={track_name}&type=track" # Query to get track id

    query_url = url + query
    result = get(query_url, headers=header) # Requesting track id
    json_result = json.loads(result.content) # Track id result
    if len(json_result) == 0:
        return None # Result is returned
    return json_result['tracks']['items'][0]['id'] # Result is returned

# Function to get audio features by track id
def get_audio_features_by_id(token, track_id):
    url = "https://api.spotify.com/v1/audio-features" # API URL for audio features

    header = get_auth_header(token)
    query = f"?ids={track_id}" # Query to get audio features

    query_url = url + query
    result = get(query_url, headers=header) # Requesting audio features
    json_result = json.loads(result.content) # Audio features result
    if len(json_result) == 0:
        return None # Result is returned
    elif 'audio_features' not in json_result.keys():
        return None # Result is returned
    elif json_result['audio_features'][0] == None:
        return None # Result is returned
    return json_result['audio_features'][0] # Result is returned

def main():
  # Dataset that has uri for spotify songs
  df = pd.read_csv('../csv_files/spotify_youtube.csv')

  # Slicing dataset
  df = df[:num]

  print(df) # Prints the sliced the spotify_youtube dataset

  # Length of the spotify_youtube dataset
  print(len(df)) # Prints the length of the sliced dataset

  # Length of the spotify id in the spotify_youtube dataset
  print(len(df['Uri'])) # Prints spotify uri

  # Seperating track id from spotify uri
  tracks_id = [] # Initializing an array
  for uri in df['Uri']:
    temp = uri[14:] # Getting spotify id
    tracks_id.append(temp) # Appending id into a new array

  print(tracks_id) # Prints all the track ids

  # Length of track ids
  print(len(tracks_id)) # Prints length of tracks_id list

  # Initialising array for spotify song ids
  tracks_ids = []

  # Initialising arrays for Audio features of the spotify songs
  danceability = []
  energy = []
  key = []
  loudness = []
  mode = []  # major = 1; minor = 0
  speechiness = []
  acousticness = []
  instrumentalness = []
  liveness = []  # The closer the instrumentalness value is to 1.0
  valence = []
  tempo = []
  duration_ms = []
  time_signature = []
  analysis_url = []
  track_href = []

  print("Initialised arrays")

  # Getting tokens
  token = get_token()

  print("Tokens")

  # Appending song id and features
  for ids in tracks_id[:num]:
    tracks_ids.append(ids) # Appending all track id

  print(tracks_ids) # Prints the track id
  print("Appending audio features")

  # Appending audio features from spotify
  count = 0
  for ids in tracks_id[:num]:
    print(str(count)+": Appending id "+str(ids))
    audio_features_result = get_audio_features_by_id(token, ids) # Gets audio features from API request
    if audio_features_result == None:
      danceability.append(None)
      energy.append(None)
      key.append(None)
      loudness.append(None)
      mode.append(None)
      speechiness.append(None)
      acousticness.append(None)
      instrumentalness.append(None)
      liveness.append(None)
      valence.append(None)
      tempo.append(None)
      duration_ms.append(None)
      time_signature.append(None)
      analysis_url.append(None)
      track_href.append(None)
    else:
      if audio_features_result['danceability'] == None:
          danceability.append(None)
      else:
          danceability.append(audio_features_result['danceability'])

      if audio_features_result['energy'] == None:
          energy.append(None)
      else:
          energy.append(audio_features_result['energy'])

      if audio_features_result['key'] == None:
          key.append(None)
      else:
          key.append(audio_features_result['key'])

      if audio_features_result['loudness'] == None:
          loudness.append(None)
      else:
          loudness.append(audio_features_result['loudness'])

      if audio_features_result['mode'] == None:
          mode.append(None)
      else:
          mode.append(audio_features_result['mode'])

      if audio_features_result['speechiness'] == None:
          speechiness.append(None)
      else:
          speechiness.append(audio_features_result['speechiness'])

      if audio_features_result['acousticness'] == None:
          acousticness.append(None)
      else:
          acousticness.append(audio_features_result['acousticness'])

      if audio_features_result['instrumentalness'] == None:
          instrumentalness.append(None)
      else:
          instrumentalness.append(audio_features_result['instrumentalness'])

      if audio_features_result['liveness'] == None:
          liveness.append(None)
      else:
          liveness.append(audio_features_result['liveness'])

      if audio_features_result['valence'] == None:
          valence.append(None)
      else:
          valence.append(audio_features_result['valence'])

      if audio_features_result['tempo'] == None:
          tempo.append(None)
      else:
          tempo.append(audio_features_result['tempo'])

      if audio_features_result['duration_ms'] == None:
          duration_ms.append(None)
      else:
          duration_ms.append(audio_features_result['duration_ms'])

      if audio_features_result['time_signature'] == None:
          time_signature.append(None)
      else:
          time_signature.append(audio_features_result['time_signature'])

      if audio_features_result['analysis_url'] == None:
          analysis_url.append(None)
      else:
          analysis_url.append(audio_features_result['analysis_url'])

      if audio_features_result['track_href'] == None:
          track_href.append(None)
      else:
          track_href.append(audio_features_result['track_href'])
    print(audio_features_result)
    count += 1

  print("Data collected")

  # Link between mood dataset and spotify data
  uri = df['Uri'] # Prints all the spotify uri of songs

  # Merging all data columns into one dataframe
  data = pd.DataFrame(
      {
        'Uri': uri, # spotify uri of the song
        'tracks_id': tracks_id, # track id of the song
        'danceability': danceability, # danceability of the song
        'energy': energy, # energy of the song
        'key': key, # key of the song
        'loudness': loudness, # loudness of the song
        'mode': mode, # mode of the song
        'speechiness': speechiness, # speechiness of the song
        'acousticness': acousticness, # acousticness of the song
        'intrumentalness': instrumentalness, # instrumentalness of the song
        'liveness': liveness, # liveness of the song
        'valence': valence, # valence of the song
        'tempo': tempo, # tempo of the song
        'duration_ms': duration_ms, # duration of the song in ms
        'time_signature': time_signature, # time signature of the song
        'analysis_url': analysis_url, # song analysis url
        'track_href': track_href # track href url
      }
  )

  # Converting the dataframe to .csv format
  data.to_csv('../csv_files/spotify_api_songs.csv') # Saves the data in .csv format

  print("Data collected has been successfully converted to CSV")

  print("CSV is successfully generated")
  return

if __name__ == '__main__':
  main()