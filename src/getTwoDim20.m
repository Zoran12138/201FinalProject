function getTwoDim20()
    clear; clc;

    % Step A: Compute MFCC for speaker s2
    % Suppose we want 20 mel-filters, and 12 or 13 coefficients
    numFilters = 20;
    numCoeffs = 12;
    
    mfcc_s2 = mfcc('s2.wav', numFilters, numCoeffs);
    % mfcc_s2 now has size [numCoeffs x numFrames_s2], e.g. [12 x 30]

    % Step B: Compute MFCC for speaker s8
    mfcc_s8 = mfcc('s8.wav', numFilters, numCoeffs);
    % mfcc_s8 might be [12 x 35], for example

    disp(['Size of mfcc_s2 = ', mat2str(size(mfcc_s2))]);
    disp(['Size of mfcc_s8 = ', mat2str(size(mfcc_s8))]);

    % Step C: We want to pick "2nd and 8th" MFCC dimension => rows [2,8].
    % Also, we only take 20 frames => columns 1:20 (assuming we have >=20).
    
    % Check if we have enough frames
    if size(mfcc_s2, 2) < 20 || size(mfcc_s8, 2) < 20
        error('One of the audio files has fewer than 20 frames, cannot get [2 x 20].');
    end

    s2data = mfcc_s2([2, 8], 1:20);  % [2 x 20]
    s8data = mfcc_s8([2, 8], 1:20);  % [2 x 20]

    % Step D: Display final results
    disp('s2data = ');
    disp(s2data);
    disp(['size(s2data) = ', mat2str(size(s2data))]);
    
    disp('s8data = ');
    disp(s8data);
    disp(['size(s8data) = ', mat2str(size(s8data))]);
end
