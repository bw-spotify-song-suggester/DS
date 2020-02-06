from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import sqlite3
import numpy as np
import json
import pickle
from math import pi
import matplotlib.pyplot as plt
import io, base64


DB = SQLAlchemy()

songs = pd.read_csv('SpotifyAudioFeaturesApril2019.csv')
features = ['valence','speechiness','liveness','instrumentalness',
            'energy','danceability','acousticness']

# pickle_path = 'nn_base.pkl'
# with open(pickle_path,'rb') as f:
#     model = pickle.load(f)

features = ['valence','speechiness','liveness','instrumentalness','energy','danceability','acousticness']
song_features = songs[features]
nn = NearestNeighbors(n_neighbors=30)
nn.fit(song_features)

conn = sqlite3.connect('songs_df.sqlite3')
songs.drop_duplicates(['track_id'],inplace=True)
songs.to_sql('songs', conn, if_exists='replace')
# from model.py import track_ids

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI']= 'sqlite:///songs_df.sqlite3'
db = SQLAlchemy(app)

def encoded_img(df,track_id,features):
    """
    Function that generates image data in binary64 format to return to 
    the backend when they request a list of suggested songs
    
    The image is a radarchart that displays the features of the 
    requested song.
    
    df = song dataframe with all the information in it 
    It must at the very least include track_id and all the features
    
    features = the features that you would like plotted in the radar chart
    
    track_id = track_id of the song you wish to plot
    """
    categories=features
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values= df[df['track_id']==track_id][features].values.flatten().tolist()
    values += values[:1]
    values
    print(values)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25,0.5,0.75,1], ["0.25","0.5","0.75","1.0"], color="grey", size=7)
    plt.ylim(0,1)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)

    pic_bytes = io.BytesIO()
    plt.savefig(pic_bytes, format="png")
    pic_bytes.seek(0)
    plt.clf()
    data = base64.b64encode(pic_bytes.read()).decode("ascii")
    img = "data:image/png;base64,{}".format(data)
    
    return img

@app.route('/processjson', methods=['POST'])
def processjson():
    # Json objects will be converted into python dictionaries
    # arrays will be converted input_data = to a list
    track_id = request.get_json(force=True)
    print(track_id)
    # recieved_id = data[0]
    song = songs[songs["track_id"] == track_id["1"]].iloc[0] # Get Song
    songs_selected = songs.copy()
    
    #nn = model # Nearest Neighbor Model
    #np array
    
    song = np.array(song[features]).reshape(1, -1)
    results = nn.kneighbors(song)

    track_ids = []
    for i in results[1][0]:
        track_ids.append(songs['track_id'].iloc[i])

    #print(track_ids)

    img = encoded_img(songs, track_id, features)


    return jsonify({'track_ids': track_ids, 'img': img})
    # return jsonify({'track_ids': track_ids})

