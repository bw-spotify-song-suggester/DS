# Data-science

- [Spotify Song Suggester API](#spotify-song-suggester-api)
  - [Use_Case](#Use_Case)
    - [Suggestion of 30 songs based on one track_id given](#retrieve-30-songs-suggested-based-on-one-track_id-given)
    - [Suggestion of 30 songs based on a list of favorited song track_ids](#retrieve-30-songs-suggested-based-on-a-list-of-favorited-song-track_ids)
    - [Radar chart image based on one track_id given](#radar-chart-image-based-on-one-track_id-given)
  - [Testing](#testing)
  - [Deployment](#deployment)

  ## Usage

#### Retrieve 30 songs suggested based on one track_id given

### `[] /song/<track_id>`


**Parameters:** None

- request:track_id

**Returns:** JSON array containing the full details of the top 30 suggestions for a track_id given and descriptive image. 

### Testing

Flask API was tested in Postman.

---

## Deployment

Find the last version of the Flask API on:

[Home](https://spotify_suggest_josh.herokuapp.com)