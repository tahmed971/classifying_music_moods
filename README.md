# Classifying the moods of songs
**General Assembly Data Science Immersive - final Capstone Project** 

The purpose of the project is to create a machine learning classifier which predicts the mood of a song - specifically, using Audiofiles library of mood grouped sounds.

## Background and problem statement
We often see individuals run projects on classifying genres of music, and whilst I personally find this interesting, I feel that we don't always choose to listen to music based on genres.
Often individuals and groups listen to music based on their mood, so why not attempt to classify off this instead?
This project aims to do this.

A challenge many find regarding this is that it's harder to find data available for music grouped by mood compared to genres. Many are able to turn to Spotify, YouTube and other platforms to pull songs based on genres though this is harder to find for moods.
Fortunately Audioset has a large library of YouTube id's grouped by seven mooods (happy, funny, sad, tender, exciting, angry, scary).

In addition, I was keen to explore weather using features from spectrograms would provide an accurate model for analysis.
A spectrogram is a visual representation of an audio frequency over a period of time. Many use these for assessing things like heart-beat abnormalities though this technique is increasingly being used within the music sphere.

## Objectives
1. Find a list of song names tagged to a mood(s).
2. Identify music from the data.
3. Extract new features from music to create new data.
4. Perform necessary EDA and pre-processing.
5. Run classification models to predict the moods of songs.

## Key tools for this project
- NumPy, Pandas, Matplotlib, scikit-learn, Seaborn 
- Moviepy, scipy.io, Librosa
