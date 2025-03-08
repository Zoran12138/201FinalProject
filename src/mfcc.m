function MFCC = mfcc(filename, numFilters, numCoeffs)
% MFCC  Compute Mel-Frequency Cepstral Coefficients from an audio file.
%
%   MFCC = mfcc(filename, numFilters, numCoeffs)
%
%   Inputs:
%       filename   : path to the audio file (e.g., 'speech.wav')
%       numFilters : number of mel filters (e.g., 20~26)
%       numCoeffs  : number of cepstral coefficients (excluding the 0th)
%                    (often 12 or 13)
%
%   Output:
%       MFCC       : matrix of size (numCoeffs x numFrames)
%                    each column is the MFCC vector for one frame
%
%   Example:
%       % Suppose you have "s1.wav" at 16kHz
%       mfccMatrix = mfcc('s1.wav', 20, 12);
%       % mfccMatrix will be 12 x (number_of_frames)
%
%   Steps included:
%       1) Read audio and (optional) pre-emphasis
%       2) Framing + Hamming window
%       3) FFT -> power spectrum
%       4) Apply mel filterbank
%       5) log + DCT => MFCC
%
%   Note: This function depends on melfb.m for the Mel filter bank.
%         Make sure melfb.m is on the MATLAB path or in the same folder.

    %% ========== 1. Read the audio file ==========
    [audioData, fs] = audioread(filename);
    audioData = audioData(:,1);   % if stereo, keep only the first channel

    % (Optional) Pre-emphasis to boost high frequencies:
    preEmph = 0.97;
    for i = length(audioData):-1:2
        audioData(i) = audioData(i) - preEmph * audioData(i-1);
    end

    %% ========== 2. Framing & Windowing ==========
    % Common frame size: 25 ms, frame shift: 10 ms
    frameSize_ms = 25;
    frameShift_ms = 10;

    frameSize = round(frameSize_ms * 1e-3 * fs);   % in samples
    frameShift= round(frameShift_ms * 1e-3 * fs);

    totalSamples = length(audioData);
    % Number of frames
    numFrames = floor((totalSamples - frameSize) / frameShift) + 1;

    % Pre-allocate memory for the MFCC result
    MFCC = zeros(numCoeffs, numFrames);

    % Hamming window
    hammWin = hamming(frameSize);

    % Typically set FFT size to at least frameSize, e.g., 512
    NFFT = 512;
    halfNFFT = floor(NFFT/2)+1;

    % Prepare Mel filterbank
    %  (melfb outputs a numFilters x halfNFFT matrix)
    melFB = melfb(numFilters, NFFT, fs);

    %% ========== 3. For each frame: FFT -> mel spectrum -> MFCC ==========
    for fIndex = 1:numFrames
        % Frame start/end index
        startIdx = (fIndex-1)*frameShift + 1;
        endIdx   = startIdx + frameSize - 1;

        % Extract frame and apply window
        frame = audioData(startIdx:endIdx) .* hammWin;

        % FFT -> power spectrum
        X = fft(frame, NFFT);
        powerSpec = abs(X(1:halfNFFT)).^2;  % only half

        % Apply mel filterbank
        melEnergy = melFB * powerSpec;  % (numFilters x 1)

        % log compression (avoid log(0))
        melEnergy(melEnergy < 1e-12) = 1e-12;
        logMel = log(melEnergy);

        % DCT to get cepstral coefficients
        c = dct(logMel);   % 1D DCT along freq filter axis
        % usually drop the 0th coefficient (related to overall energy):
        MFCC(:, fIndex) = c(2 : (numCoeffs+1));  
    end
end
