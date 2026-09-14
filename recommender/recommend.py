import pandas as pd
import re
import numpy as np
# Load our two datasets
playlist = pd.read_csv("playlist.csv")
music = pd.read_csv("spotify-tracks-dataset-detailed.csv")
billboard = pd.read_csv("Hot 100 Audio Features.csv")
tracks = pd.read_csv("tracks.csv")

print("\n600K Spotify tracks dataset:")
print("Number of tracks:", len(tracks))
print("Columns:")
print(tracks.columns.tolist())

print("\nFirst 5 tracks:")
print(tracks.head())
print("\nBillboard dataset:")
print("Number of songs:", len(billboard))
print("Columns:")
print(billboard.columns.tolist())

print("\nFirst 5 Billboard songs:")
print(billboard.head())


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()

print("Our playlist:")
print(playlist)

print("\nNumber of songs in our playlist:", len(playlist))

print("\nNumber of songs in the music database:", len(music))


# Remove duplicate song/artist combinations
music_clean = music.drop_duplicates(
    subset=["track_name", "artists"]
)

print("\nSongs after removing duplicates:", len(music_clean))
# Create normalized names for matching
playlist["song_normalized"] = playlist["song"].apply(normalize_text)
playlist["artist_normalized"] = playlist["artist"].apply(normalize_text)

music_clean["song_normalized"] = music_clean["track_name"].apply(normalize_text)
music_clean["artist_normalized"] = music_clean["artists"].apply(normalize_text)

# Normalize Billboard song and artist names -- 
billboard["song_normalized"] = billboard["Song"].apply(normalize_text) 
billboard["artist_normalized"] = billboard["Performer"].apply(normalize_text) 

# -----------------------------------------
# CHECK MISSING SONGS IN 600K DATASET
# -----------------------------------------

# Normalize track and artist names 
tracks["song_normalized"] = tracks["name"].apply(normalize_text)  
tracks["artist_normalized"] = tracks["artists"].apply(normalize_text)  

# BUILD OUR FINAL SEED FEATURE TABLE
# -----------------------------------------

# Start with our new 15 songs
seed_features = playlist[
    ["song", "artist"]
].copy()

# Create empty feature columns
seed_features["energy"] = np.nan
seed_features["danceability"] = np.nan
seed_features["acousticness"] = np.nan

# Keep track of where each song's features came from
seed_features["source"] = "not_found"


# -----------------------------------------
# 1. SEARCH THE 114K DATASET
# -----------------------------------------

for i, row in seed_features.iterrows():

    song_match = normalize_text(row["song"])
    artist_match = normalize_text(row["artist"])

    possible = music_clean[
        (music_clean["song_normalized"] == song_match) &
        (music_clean["artist_normalized"] == artist_match)
    ]

    if len(possible) > 0:

        track = possible.iloc[0]

        seed_features.loc[i, "energy"] = track["energy"]
        seed_features.loc[i, "danceability"] = track["danceability"]
        seed_features.loc[i, "acousticness"] = track["acousticness"]

        seed_features.loc[i, "source"] = "114K_dataset"


# -----------------------------------------
# 2. SEARCH THE 600K DATASET
# -----------------------------------------

for i, row in seed_features.iterrows():

    if row["source"] != "not_found":
        continue

    song_match = normalize_text(row["song"])
    artist_match = normalize_text(row["artist"])

    possible = tracks[
        (tracks["song_normalized"] == song_match) &
        (tracks["artist_normalized"] == artist_match)
    ]

    if len(possible) > 0:

        track = possible.iloc[0]

        seed_features.loc[i, "energy"] = track["energy"]
        seed_features.loc[i, "danceability"] = track["danceability"]
        seed_features.loc[i, "acousticness"] = track["acousticness"]

        seed_features.loc[i, "source"] = "600K_dataset"


# -----------------------------------------
# 3. SEARCH BILLBOARD DATASET
# -----------------------------------------

for i, row in seed_features.iterrows():

    if row["source"] != "not_found":
        continue

    song_match = normalize_text(row["song"])
    artist_match = normalize_text(row["artist"])

    possible = billboard[
        (billboard["song_normalized"] == song_match) &
        (billboard["artist_normalized"] == artist_match)
    ]

    if len(possible) > 0:

        track = possible.iloc[0]

        seed_features.loc[i, "energy"] = track["energy"]
        seed_features.loc[i, "danceability"] = track["danceability"]
        seed_features.loc[i, "acousticness"] = track["acousticness"]

        seed_features.loc[i, "source"] = "Billboard_dataset"


# -----------------------------------------
# SHOW FINAL SEED FEATURES
# -----------------------------------------

print("\nFinal seed features:")
print(seed_features.to_string(index=False))


# -----------------------------------------
# CHECK FOR MISSING SONGS
# -----------------------------------------

missing_seeds = seed_features[
    seed_features["source"] == "not_found"
]

if len(missing_seeds) > 0:

    print("\nSongs not found in our datasets:")

    print(
        missing_seeds[
            ["song", "artist"]
        ].to_string(index=False)
    )


# -----------------------------------------
# CALCULATE USER TASTE PROFILE
# -----------------------------------------

features = [
    "energy",
    "danceability",
    "acousticness"
]

# Only use songs for which we have features
profile_data = seed_features.dropna(
    subset=features
)

taste_profile = profile_data[
    features
].mean()

print("\nYour new taste profile:")
print(taste_profile)

print(
    "\nSongs used for taste profile:",
    len(profile_data),
    "/",
    len(seed_features)
)

# Convert taste profile to NumPy array
taste_vector = taste_profile[
    features
].values

print("\nTaste vector:")
print(taste_vector)


# -----------------------------------------
# BUILD RECOMMENDATION SYSTEM
# -----------------------------------------

# Find genres related to your artists

seed_artists = set(
    playlist["artist_normalized"]
)

seed_artist_tracks = music_clean[
    music_clean["artist_normalized"].isin(seed_artists)
]

genre_counts = (
    seed_artist_tracks
    .groupby("track_genre")["artist_normalized"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nYour strongest genres:")
print(genre_counts.head(10))


# -----------------------------------------
# CREATE GENRE WEIGHTS
# -----------------------------------------

preferred_genres = genre_counts.head(10)

if len(preferred_genres) > 0:

    genre_weights = (
        preferred_genres / preferred_genres.max()
    )

else:

    genre_weights = pd.Series(dtype=float)

print("\nGenre weights:")
print(genre_weights)


# -----------------------------------------
# CREATE CANDIDATE POOL
# -----------------------------------------

candidates = music_clean.copy()

print(
    "\nInitial candidate songs:",
    len(candidates)
)


# Remove artists already in playlist

known_artists = set(
    playlist["artist_normalized"]
)

candidates = candidates[
    ~candidates["artist_normalized"].isin(
        known_artists
    )
].copy()


# Remove songs already in playlist

known_songs = set(
    playlist["song_normalized"]
)

candidates = candidates[
    ~candidates["song_normalized"].isin(
        known_songs
    )
].copy()

print(
    "Candidates after filtering:",
    len(candidates)
)


# -----------------------------------------
# CALCULATE WEIGHTED AUDIO SIMILARITY
# -----------------------------------------

candidate_features = candidates[
    features
].values

feature_weights = np.array([
    0.35,   # energy
    0.35,   # danceability
    0.30    # acousticness
])

weighted_difference = (
    candidate_features - taste_vector
) ** 2

weighted_difference = (
    weighted_difference * feature_weights
)

candidates["taste_distance"] = np.sqrt(
    weighted_difference.sum(axis=1)
)


# -----------------------------------------
# COMPARE WITH INDIVIDUAL SEED SONGS
# -----------------------------------------

seed_matrix = profile_data[
    features
].values

candidate_matrix = candidates[
    features
].values

individual_distances = np.sqrt(
    (
        (
            candidate_matrix[:, None, :]
            - seed_matrix[None, :, :]
        ) ** 2
        * feature_weights
    ).sum(axis=2)
)

candidates["seed_distance"] = (
    individual_distances.min(axis=1)
)


# -----------------------------------------
# GENRE SCORE
# -----------------------------------------

candidates["genre_score"] = (
    candidates["track_genre"]
    .map(genre_weights)
    .fillna(0)
)

# -----------------------------------------
# ENGLISH-LANGUAGE PREFERENCE
# -----------------------------------------

# Songs appearing in the Billboard Hot 100
# are used as a rough English-music signal.

billboard_song_keys = set(
    zip(
        billboard["song_normalized"],
        billboard["artist_normalized"]
    )
)

candidates["english_score"] = candidates.apply(
    lambda row: 1
    if (
        row["song_normalized"],
        row["artist_normalized"]
    ) in billboard_song_keys
    else 0,
    axis=1
)

print(
    "\nEnglish-likelihood signal added."
)
# -----------------------------------------
# POPULARITY SCORE
# -----------------------------------------

candidates["popularity_score"] = (
    candidates["popularity"] / 100
)


# -----------------------------------------
# NORMALIZE DISTANCES
# -----------------------------------------

candidates["taste_distance_norm"] = (
    candidates["taste_distance"]
    / candidates["taste_distance"].max()
)

candidates["seed_distance_norm"] = (
    candidates["seed_distance"]
    / candidates["seed_distance"].max()
)


# -----------------------------------------
# FINAL V1 SCORE
# -----------------------------------------

candidates["final_score"] = (
    0.50 * candidates["taste_distance_norm"]
    + 0.20 * candidates["seed_distance_norm"]
    - 0.15 * candidates["genre_score"]
    - 0.05 * candidates["popularity_score"]
    - 0.10 * candidates["english_score"]
)


# -----------------------------------------
# DIVERSITY-AWARE SELECTION
# -----------------------------------------

remaining = candidates.copy()

selected = []

while len(selected) < 10 and len(remaining) > 0:

    best_score = float("inf")
    best_index = None

    for index, row in remaining.iterrows():

        score = row["final_score"]

        # Penalize repeated genres
        if row["track_genre"] in [
            song["track_genre"]
            for song in selected
        ]:
            score += 0.04

        # Penalize repeated artists
        if row["artist_normalized"] in [
            song["artist_normalized"]
            for song in selected
        ]:
            score += 0.15

        if score < best_score:

            best_score = score
            best_index = index

    selected.append(
        remaining.loc[best_index]
    )

    remaining = remaining.drop(
        best_index
    )


# -----------------------------------------
# FINAL V1 RECOMMENDATIONS
# -----------------------------------------

recommendations = pd.DataFrame(
    selected
)

print("\n")
print("=========================================")
print("🎧 FINAL V1 RECOMMENDATIONS")
print("=========================================")

print(
    recommendations[[
        "track_name",
        "artists",
        "track_genre",
        "energy",
        "danceability",
        "acousticness",
        "genre_score",
        "final_score"
    ]].to_string(index=False)
)