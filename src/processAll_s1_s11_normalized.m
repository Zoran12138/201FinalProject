function processAll_s1_s11_normalized()
% processAll_s1_s11_normalized
%
% For each file s1.wav ~ s11.wav in:
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Train
% we do:
%   1) read => single-channel => subtract mean => amplitude normalize
%   2) plot the normalized time-domain waveform
%   3) STFT => show periodogram in dB
%   4) apply melFB => show mel-spectrum
%   5) compute MFCC => show it
%
% This script ensures the entire processing (STFT, melFB, MFCC) is done
% on the normalized waveform. Each file will generate multiple figures.

    clear; clc; close all;

    folderPath = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Train';
    nFiles = 11;

    % STFT parameters:
    winLen   = 256;
    overlap  = 128;
    nfftSize = 512;

    % Mel / MFCC parameters:
    numMelFilters = 26;
    numCoeffs     = 13;  % e.g. c1..c13 (or skip c0)

    for i=1:nFiles
        %% Step 1: read file
        wavName = sprintf('s%d.wav', i);
        fullPath= fullfile(folderPath, wavName);

        if ~exist(fullPath, 'file')
            fprintf('File not found: %s (skip)\n', fullPath);
            continue;
        end

        [y, fs] = audioread(fullPath);
        if size(y,2) > 1
            y = y(:,1); % keep single channel
        end

        %% Step 2: subtract mean & normalize amplitude
        y = y - mean(y);
        peakVal = max(abs(y));
        if peakVal>1e-12
            y = y / peakVal;
        else
            warning('File %s might be silent (peak < 1e-12).', wavName);
        end

        fprintf('Processing %s => length=%d, fs=%d\n', wavName, length(y), fs);

        % Plot time-domain after normalization
        figTime= figure('Name',['Time domain: ' wavName],'NumberTitle','off');
        t = (0:length(y)-1)/fs;
        plot(t, y, 'b-','LineWidth',1);
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(['Normalized wave: ' wavName]);
        grid on;

        %% Step 3: compute STFT => show periodogram (dB)
        [S, fAxis, tAxis] = spectrogram(y, hamming(winLen), overlap, nfftSize, fs);
        powerSpec = abs(S).^2; % freqBins x numFrames

        figSTFT= figure('Name',['STFT (periodogram): ' wavName],'NumberTitle','off');
        surf(tAxis, fAxis, 10*log10(powerSpec), 'EdgeColor','none');
        axis tight; view(0,90); colorbar;
        colormap jet;
        title(['Periodogram (dB): ' wavName]);
        xlabel('Time (s)'); ylabel('Frequency (Hz)');

        %% Step 4: apply mel filter bank => mel spectrum
        melSpec = applyMelFB(S, fs, numMelFilters);  
        % melSpec => (numMelFilters x numFrames)

        figMel= figure('Name',['Mel spectrum: ' wavName],'NumberTitle','off');
        numFrames = size(melSpec,2);
        surf( (1:numFrames), (1:numMelFilters), 10*log10(melSpec), 'EdgeColor','none');
        axis tight; view(0,90); colorbar;
        colormap jet;
        title(['Mel spectrum (dB): ' wavName]);
        xlabel('Frame index'); ylabel('Mel filter index');

        %% Step 5: compute MFCC => show
        MFCC = computeMFCC(melSpec, numCoeffs);
        % MFCC => (numCoeffs x numFrames)

        figMFCC= figure('Name',['MFCC: ' wavName],'NumberTitle','off');
        imagesc(MFCC); axis xy; colorbar;
        colormap jet;
        xlabel('Frame index'); ylabel('MFCC coefficient');
        title(['MFCC (normed wave): ' wavName]);

        drawnow;
    end
end

%% Subfunction: apply melFB
function melSpec = applyMelFB(S, fs, numFilters)
% S => freqBins x numFrames, from spectrogram
% compute power => multiply by melFB => melSpec

    powerSpec = abs(S).^2;
    freqBins  = size(S,1);
    NFFT      = (freqBins-1)*2;  % typical for onesided stft

    melFB_ = melfb(numFilters, NFFT, fs);
    if size(melFB_,2)~= freqBins
        error('Dimension mismatch in applyMelFB: melFB(%dx%d) vs freqBins=%d',...
            size(melFB_,1), size(melFB_,2), freqBins);
    end

    melSpec = melFB_ * powerSpec;
end

%% Subfunction: computeMFCC
function MFCC = computeMFCC(melSpec, numCoeffs)
% melSpec => (numFilters x numFrames)
% do log => dct => keep c1..cN
    melSpec(melSpec<1e-12) = 1e-12;
    logMel = log(melSpec);

    dctAll = dct(logMel); % (numFilters x numFrames)
    % typically skip c0 => keep c1..c(numCoeffs)
    MFCC = dctAll(2:numCoeffs+1, :);
end

%% Subfunction: melfb
function m = melfb(p, n, fs)
% same as your melfb
    f0  = 700/fs;
    fn2 = floor(n/2);
    lr  = log(1+ 0.5/f0)/(p+1);

    bl= n*(f0*(exp([0,1,p,p+1]*lr)-1));
    b1= floor(bl(1))+1;
    b2= ceil(bl(2));
    b3= floor(bl(3));
    b4= min(fn2, ceil(bl(4)))-1;

    pf= log(1 + (b1:b4)/(n*f0))/lr;
    fp= floor(pf);
    pm= pf-fp;

    r= [fp(b2:b4), 1+fp(1:b3)];
    c= [b2:b4,     1:b3] + 1;
    v= 2*[1-pm(b2:b4), pm(1:b3)];
    m= sparse(r,c,v,p, fn2+1);
end
