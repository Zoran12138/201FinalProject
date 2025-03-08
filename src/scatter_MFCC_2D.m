% scatter_MFCC_2D.m
%
% This script plots two selected MFCC dimensions (2nd and 8th) for two speakers.
% Each speaker has an MFCC matrix of size [numCoeffs x numFrames].
% Here, "Speaker 2" and "Speaker 8" are shown as examples.

clear; clc; close all;

% Step 1: Choose two MFCC dimensions.
dimX = 2;  % X-axis dimension
dimY = 8;  % Y-axis dimension

% Step 2: Provide the MFCC matrices for two speakers.
% Replace the following lines with actual MFCC data.
% For example, MFCC_Speaker2 should be size [>=8 x #frames].
MFCC_Speaker2 = randn(13, 100);  % Placeholder: random data
MFCC_Speaker8 = randn(13,  90);  % Placeholder: random data

% Step 3: Extract the chosen dimensions from each speaker's MFCC.
X_Speaker2 = MFCC_Speaker2(dimX, :);
Y_Speaker2 = MFCC_Speaker2(dimY, :);

X_Speaker8 = MFCC_Speaker8(dimX, :);
Y_Speaker8 = MFCC_Speaker8(dimY, :);

% Step 4: Plot the data in a 2D plane.
figure;
plot(X_Speaker2, Y_Speaker2, 'bo'); % Blue circles for Speaker 2
hold on;
plot(X_Speaker8, Y_Speaker8, 'rx'); % Red crosses for Speaker 8
hold off;

legend('Speaker 2','Speaker 8','Location','best');
xlabel(['MFCC dimension ', num2str(dimX)]);
ylabel(['MFCC dimension ', num2str(dimY)]);
title('2D Scatter of MFCC Features');
grid on;
