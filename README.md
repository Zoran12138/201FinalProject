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

2.Testing:
For a new utterance, compute its MFCC features. Compare each feature vector to the codewords in each speaker’s codebook using Euclidean distance. Compute the average distortion to each speaker’s codebook and pick the speaker with the smallest distortion as the recognized identity.

# Implementation & Results
We performed a series of tests (labeled “Test 1” through “Test 10,” etc.):

Test 1: Our human performance recognition rate is 72.7%.

Test 2: After the normalization of the 11 raw data from "train" folder, we plot the signal in time domain. 

![image](https://github.com/user-attachments/assets/ec035457-5611-482c-ac0f-6dbe0161a993)

Use STFT to generate periodogram.
![image](https://github.com/user-attachments/assets/9be3497c-099e-4b29-964f-b2d3d74c0278)


Test 3: Plot the mel-spaced filter bank responses.
![image](https://github.com/user-attachments/assets/a3d77328-4543-41af-80fe-238a88e464bf)

The explanation of the impact of the melfb.m is described in https://github.com/Zoran12138/201FinalProject/blob/ef75104a93adafd676167c7d09a5243cbf555e4b/src/melfb.m

Test 4: Combining all steps into an MFCC extraction function. 
https://github.com/Zoran12138/201FinalProject/blob/9d14b0f85faca1eabc60f4d0791c73ff63b92f6b/src/computeMFCC_all.m

Test 5: Visualizing MFCC 2D scatter for different speakers. 
![image](https://github.com/user-attachments/assets/4d6a16d2-8bfb-4050-99c6-0fda243f13a0)

Test 6: Generating LBG codewords and plotting them over the MFCC scatter. 
![image](https://github.com/user-attachments/assets/c4431e23-38e1-4fb9-b516-6327ac0fd363)

Test 7: Full speaker recognition with multiple training/testing sets, measuring final accuracy.
Results show that our system yield a training accuracy of 100% and a testing accuracy of 87.5%.

<img width="366" alt="60ee8ecc7e49f1f3c10c39a3bec2a29" src="https://github.com/user-attachments/assets/5f9a502c-7d4a-4e76-8778-f055e272850f" />

<img width="402" alt="7d1babf60c732093ad7b4287eb5f0a3" src="https://github.com/user-attachments/assets/8b72a61e-a20e-4e76-af73-e71f655e29f0" />



Test 8: Applying notch filters to the test signals, verifying how the system performance is affected. 
Test 9: Extending the system to more speakers, additional data, or different words (“zero” -> “twelve,” plus new students’ voices). 
Test 10: Attempting multiword or multi-speaker tasks, evaluating how the system recognizes both the word and the speaker.




