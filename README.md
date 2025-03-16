# 201FinalProject

TEAM NAME: OhCaptainMyCaptain

Members: Zherong Yu + Haoyang Li

# **Overview**

This project builds a Speaker Recognition system which aims to determine which speaker said a particular piece of audio. 
The system was built using:

1.**Feature Extraction** with Mel-Frequency Cepstral Coefficients (MFCC)

2.**Vector Quantization** (VQ) via the Linde-Buzo-Gray (LBG) clustering algorithm, to create speaker-specific codebooks

3.A standard matching procedure that compares test-speech MFCC vectors against each speaker’s VQ codebook

The project also explores robustness tests by applying notch filters (or other distortions) to the test data, as well as additional tasks such as identifying not only which speaker but also which word (e.g., “zero” or “twelve”).



