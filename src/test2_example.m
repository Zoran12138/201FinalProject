% test2_example.m
%
% Demonstration for Task 2 (Test 2):
%   1) Read and play an audio file using "getFile" and "sound"
%   2) Display sampling rate
%   3) Compute how many milliseconds are in 256 samples
%   4) Plot the signal in the time domain (check if normalization is needed)
%   5) Use STFT (spectrogram) to generate the periodogram
%   6) Identify the time (ms) and frequency (Hz) where the energy is concentrated
%   7) Repeat for different frame sizes: N = 128, 256, 512 (frame increment ~ N/3)

clear; clc; close all;

%% 1) Read an audio file and play it
[s, fs, t] = getFile(1, "train");  % Example: reading s1.wav from the 'train' folder
sound(s, fs);

disp(['Sampling rate = ', num2str(fs), ' Hz']);

%% 2) Compute the duration (in ms) of 256 samples
numSamples = 256;
msFor256 = (numSamples / fs) * 1000;
disp(['256 samples ~= ', num2str(msFor256), ' ms']);

%% 3) Plot the time-domain waveform and consider normalization
figure;
plot(t, s);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time-Domain Waveform (Original)');

% Optional normalization if amplitude is very large
maxVal = max(abs(s));
if maxVal > 1
    s = s / maxVal;
    disp('Signal was normalized (amplitude > 1).');
end

% Plot again after normalization
figure;
plot((0 : length(s) - 1) / fs, s);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time-Domain Waveform (After Normalization)');

%% 4) Use STFT to generate the spectrogram for different frame sizes
plotSTFT(s, fs, 128);
plotSTFT(s, fs, 256);
plotSTFT(s, fs, 512);

% Observing these spectrograms can help locate where the energy is strongest 
% (time in seconds and frequency in Hz).

function plotSTFT(signal, fs, N)
% PLOTSTFT  Wrapper function to visualize the spectrogram of 'signal'
%           using a frame size of N.
%
%           The frame increment M is set to roughly N/3, and the overlap
%           is then N - M. We use a Hamming window of length N.

    M = round(N / 3);
    noverlap = N - M;

    figure;
    spectrogram(signal, hamming(N), noverlap, N, fs, 'yaxis');
    title(['Spectrogram with Frame Size N = ', num2str(N)]);
    colormap jet; 
    colorbar;
end
