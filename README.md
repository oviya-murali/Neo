 Neo — Music Discovery & Recommendation System

Neo is a music recommendation system designed to help users discover **new songs and artists** based on their existing musical taste.

Instead of simply recommending more songs from artists a user already listens to, Neo analyzes the characteristics of their playlist and ranks previously unseen music based on similarity, genre compatibility, popularity, and diversity.

---

## 🚀 What Neo Does

Given a playlist of songs and artists, Neo:

- Builds a representation of the user's musical taste
- Matches songs against multiple music datasets
- Uses audio features such as:
  - Energy
  - Danceability
  - Acousticness
- Measures similarity between candidate songs and the user's overall taste
- Compares candidates with individual songs from the playlist
- Incorporates genre compatibility
- Uses popularity as a small ranking signal
- Filters out artists and songs already present in the playlist
- Applies diversity-aware selection to avoid repetitive recommendations

The result is a ranked list of **new music recommendations**.
