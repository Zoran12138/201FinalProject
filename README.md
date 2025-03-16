# 201FinalProject

TEAM NAME: OhCaptainMyCaptain

Members: Zherong Yu + Haoyang Li

# **Overview**

Welcome to this Speaker Recognition project repository😊 

This project builds a Speaker Recognition system which aims to determine which speaker said a particular piece of audio. 
The system was built using:

1.**Feature Extraction** with Mel-Frequency Cepstral Coefficients (MFCC)

2.**Vector Quantization** (VQ) via the Linde-Buzo-Gray (LBG) clustering algorithm, to create speaker-specific codebooks

3.A standard matching procedure that compares test-speech MFCC vectors against each speaker’s VQ codebook

The project also explores robustness tests by applying notch filters (or other distortions) to the test data, as well as additional tasks such as identifying not only which speaker but also which word (e.g., “zero” or “twelve”).

**This system achieved 100% accuracy on training data. For testing data, when it only needs to recognize the speaker without noise, accuracy ranged from 83.33% to 97.83%, depending on the speaker set.**

# Motivation 
Speaker recognition is a broad field in speech processing that typically has two main goals:

Speaker Identification: Given an input speech sample, determine which speaker (from a known set) produced it. 

Word (or utterance) Recognition: Distinguish what word or phrase was spoken. 

We focus on a simpler scenario: classify which speaker out of a small group, and/or also identify which single word was spoken (among a small set, like “zero” vs. “twelve,” or “five” vs. “eleven”). This naturally leads to interesting sub-tasks like:
(A) Speaker classification (is the speaker ID=1, 2, …?)Word classification (is the speech “zero” or “twelve”?). 
(B) Word classification (is the speech “zero” or “twelve”?).

# Challenges
Audio Variability: Different speakers, different recording conditions, amplitude differences, leading to wide dynamic range. 

Noise and Distortion: We tested the system’s robustness by adding notch filters or other degradations. 

Limited Data: For each speaker or word, we only have a few recordings. This can reduce accuracy. 

MFCC Configuration: Deciding on window length, overlap, number of FFT points, number of mel-filter banks, and number of DCT coefficients can significantly influence performance. 

# Data Description
We used three sets of data to demonstrate how the system performs:

Default Speech Data: “zero” utterances from multiple speakers.

2024 Student Data: Utterances of “zero” and “twelve.”

2025 Student Data: Utterances of “five” and “eleven.”

Each set has train and test subsets:

Train: Audio files used to build the codebooks (one codebook per speaker).

Test: Audio files used to evaluate how well the trained system identifies unseen utterances.


# Approach
The speaker recognition pipeline comprises the following steps:

## Feature Extraction (MFCC)
1.Frame Blocking: Segment the raw audio into frames of 𝑁=256 samples, with an overlap of N−M (e.g., M=100).

2.Windowing: Apply a Hamming window to each frame to minimize spectral leakage.

3.FFT: Compute the short-time Fourier transform to get each frame’s frequency spectrum.

4.Mel-Frequency Wrapping: Multiply the spectrum by a set of Mel-spaced triangular filters to replicate the human ear’s critical bandwidth perception.

5.Log & DCT: Take the log of the Mel-filter outputs and then apply a Discrete Cosine Transform to produce MFCC features. Typically the 0th coefficient is dropped, resulting in (K-1) dimensions per frame.

## Vector Quantization and Codebooks
1.Training:

Aggregate all MFCC feature vectors from a given speaker. Use LBG clustering to split them into Q clusters, producing Q centroids. These centroids (codewords) form the speaker’s VQ codebook.

Testing:

For a new utterance, compute its MFCC features. Compare each feature vector to the codewords in each speaker’s codebook using Euclidean distance. Compute the average distortion to each speaker’s codebook and pick the speaker with the smallest distortion as the recognized identity.





