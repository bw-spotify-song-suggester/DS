from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import json
import pickle
from math import pi
import matplotlib.pyplot as plt
import io, base64



songs = pd.read_csv('SpotifyAudioFeaturesApril2019.csv')
songs.drop_duplicates(['track_id'],inplace=True)

features = ['valence','speechiness','liveness','instrumentalness',
            'energy','danceability','acousticness']
song_features = songs[features]

nn = NearestNeighbors(n_neighbors=30)
nn.fit(song_features)


app = Flask(__name__)
CORS(app)

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

    # The request from BE is going to contain a Json
    # object with the format {"track_id":"0M5gFT4YDJv5uiqRB2CEvF"}
    # The key to check from the dictionary is "track_id"

    # Get Song
    song = songs[songs["track_id"] == track_id["track_id"]].iloc[0]

    # Convert the song data into a format that the nn model understands
    song = np.array(song[features]).reshape(1, -1)
    results = nn.kneighbors(song)

    # if you want to return track id uncomment the following code
    # track_ids = []
    # for i in results[1][0]:
    #     track_ids.append(songs['track_id'].iloc[i])


    return_dict = {}
    for index,i in enumerate(results[1][0]):
        return_dict[str(index)] = {"track_name" : songs['track_name'].iloc[i],
            `                      "artist" : songs['artist_name'].iloc[i]}
    # Print(track_ids)

    # Get the string that encodes the image from the encode_img function
    img = encoded_img(songs, track_id["track_id"], features)

    return jsonify({"results": return_dict, "img": img})
    # return JSON object
    # {'track_ids': [List of track ids],"img":[String with encoded image data]]}
