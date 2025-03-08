function test8_notch_experiment_success()
% test8_notch_experiment_success
%
% Final working example for Test 8:
% 1) Train speaker models from .wav in train folder (onesided STFT => 1/2 spectrum)
% 2) Test baseline (no filter) and apply Notch filters at various center freq
% 3) Avoid dimension mismatch by using 'FrequencyRange','onesided' for stft
%
% Requirements:
%  - Place this script in "D:\Program Files\Polyspace\R2021a\bin\" 
%  - Have training WAV in: "D:\Program Files\Polyspace\R2021a\bin\EEC201\train\"
%  - Have testing WAV in:  "D:\Program Files\Polyspace\R2021a\bin\EEC201\test\"
%  - NFFT=512 => freqRows = 257 => melfb(numFilters,512,fs) => [numFilters x 257]

    clear; clc;

    % ========== 1) Set Paths & Config ==========
    trainDir = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\';
    testDir  = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\';
    modelFile= 'speakerModels.mat';   % store codebooks
    numFilters= 20;
    numCoeffs = 12;
    M         = 8;  % LBG codebook size

    % For STFT:
    N      = 256;         % frame length
    Mstep  = 100;         % frame step => overlap = N - Mstep
    NFFT   = 512;         % must remain consistent => stft -> freqRows=257 => matches melfb

    centerFreqs= [500, 1000, 2000];  % Notch test freq
    notchWidth = 0.1;                % default bandwidth

    %% ========== 2) Load or Train Models ==========
    if exist(modelFile,'file')==2
        fprintf('Loading existing models from %s...\n', modelFile);
        load(modelFile,'speakerModels','N','Mstep','NFFT','numFilters','numCoeffs','M');
        fprintf('Load done. #Codebooks=%d, N=%d, Mstep=%d, NFFT=%d\n', ...
            numel(speakerModels), N, Mstep, NFFT);
    else
        fprintf('%s not found => Start training...\n', modelFile);
        speakerModels = train_speakers_robust(trainDir, numFilters, numCoeffs, M, N, Mstep, NFFT);
        % Save 
        save(modelFile,'speakerModels','N','Mstep','NFFT','numFilters','numCoeffs','M');
        fprintf('Training done. Saved => %s\n', modelFile);
    end

    %% ========== 3) Baseline Test (No Filter) ==========
    [accBaseline, ~] = test_speakers_robust(testDir, speakerModels, ...
                                numFilters, numCoeffs, N, Mstep, NFFT, false, 0);
    fprintf('\nBaseline accuracy (no filter) = %.2f%%\n\n', accBaseline);

    %% ========== 4) Notch tests ==========
    results = zeros(size(centerFreqs));
    for i=1:numel(centerFreqs)
        fo = centerFreqs(i);
        fprintf('--- Test with Notch@%d Hz ---\n', fo);
        [accNotch, ~] = test_speakers_robust(testDir, speakerModels, ...
                                     numFilters, numCoeffs, N, Mstep, NFFT, true, fo);
        results(i)= accNotch;
        fprintf('Notch freq=%d => accuracy=%.2f%%\n\n', fo, accNotch);
    end

    %% ========== 5) Summary ==========
    fprintf('==== Final Summary ====\n');
    fprintf('Baseline : %.2f%%\n', accBaseline);
    for i=1:numel(centerFreqs)
        fprintf('Notch@%4d : %.2f%%\n', centerFreqs(i), results(i));
    end
    fprintf('=======================\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TRAIN SPEAKERS
function speakerModels = train_speakers_robust(folderTrain, numFilters, numCoeffs, M, N, Mstep, NFFT)
    fList= dir(fullfile(folderTrain,'*.wav'));
    nFiles= length(fList);
    if nFiles<1
        error('No .wav in %s => cannot train', folderTrain);
    end
    speakerModels= cell(nFiles,1);

    for i=1:nFiles
        fPath = fullfile(fList(i).folder, fList(i).name);
        fprintf('Training file #%d => %s\n', i, fList(i).name);

        [y, fs] = audioread(fPath);
        mfccMat= audio2mfcc(y, fs, numFilters, numCoeffs, N, Mstep, NFFT);
        codebook= trainVQ_LBG(mfccMat, M);
        speakerModels{i}= codebook;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TEST SPEAKERS
function [accuracy, predictions] = test_speakers_robust(folderTest, speakerModels, ...
                                numFilters, numCoeffs, N, Mstep, NFFT, doNotch, fo)
    if ~exist('doNotch','var'), doNotch=false; end
    if doNotch && (~exist('fo','var')||isempty(fo))
        fo=1000;
    end

    fList= dir(fullfile(folderTest,'*.wav'));
    nFiles= length(fList);
    predictions= zeros(nFiles,1);
    correct=0;

    if nFiles<1
        warning('No .wav in %s => 0 accuracy.', folderTest);
        accuracy=0; return;
    end

    for i=1:nFiles
        fname= fList(i).name;
        fPath= fullfile(fList(i).folder, fname);
        [y, fs]= audioread(fPath);

        if doNotch
            y= applyNotch(y, fs, fo, 0.1);
        end

        mfccTest= audio2mfcc(y, fs, numFilters, numCoeffs, N, Mstep, NFFT);

        [bestID, distVal] = findBestSpeaker(mfccTest, speakerModels);

        predictions(i)= bestID;
        trueID= parseTrueID(fname);   % from filename e.g. s3.wav => 3

        isCorr= (bestID==trueID && trueID>0);
        if isCorr, correct= correct+1; end

        labelStr= ternary(doNotch,sprintf('Notch@%d',fo),'NoFilter');
        okStr   = ternary(isCorr,'[OK]','[ERR]');
        fprintf('[%s] %s => spk#%d (true:%d) Dist=%.3f %s\n', ...
            labelStr, fname, bestID, trueID, distVal, okStr);
    end
    used= sum(predictions>0);
    if used>0
        accuracy= (correct/used)*100;
    else
        accuracy=0;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% AUDIO -> MFCC
function mfccMat = audio2mfcc(y, fs, numFilters, numCoeffs, N, Mstep, NFFT)
    ps = audio2powerspec(y, fs, N, Mstep, NFFT);
    mfccMat= myMfcc(ps, fs, numFilters, numCoeffs, NFFT);
    mfccMat= mfccMat - (mean(mfccMat,2) + 1e-8);  % optional mean normalization
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% AUDIO -> STFT => POWER SPECTRUM
function ps = audio2powerspec(y, fs, N, Mstep, NFFT)
    % PreEmphasis
    alpha=0.95;
    for i=length(y):-1:2
        y(i)= y(i)- alpha*y(i-1);
    end

    overlap= N - Mstep;
    w= hamming(N);
    % ******* KEY FIX: 'FrequencyRange','onesided' => freqRows = NFFT/2+1 *******
    [S, F, T] = stft(y, fs, ...
        'Window', w, ...
        'OverlapLength', overlap, ...
        'FFTLength', NFFT, ...
        'FrequencyRange','onesided');

    [freqRows,timeCols]= size(S);
    expected= (NFFT/2)+1;
    if freqRows~= expected
        error('Dimension mismatch: stft freqRows=%d but expecting %d (NFFT/2+1)', freqRows, expected);
    end

    ps= (abs(S).^2)./NFFT;  % each col => freq bin power
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MFCC from PowerSpectrum
function c = myMfcc(ps, fs, numFilters, numCoeffs, NFFT)
    [freqRows, nFrames]= size(ps);
    halfNFFT= (NFFT/2)+1;
    if freqRows~= halfNFFT
        error('myMfcc: freqRows=%d but need %d => stft config mismatch', freqRows, halfNFFT);
    end

    melFB= melfb(numFilters, NFFT, fs); % => [numFilters x (NFFT/2+1)]
    if size(melFB,2)~= freqRows
        error('myMfcc: melFB has %d cols but ps has %d rows => mismatch', ...
               size(melFB,2), freqRows);
    end

    c= zeros(numCoeffs, nFrames);
    for fIndex=1:nFrames
        psCol= ps(:,fIndex);  % [freqRows x 1]
        melEnergy= melFB * psCol;  % [numFilters x 1]
        melEnergy(melEnergy<1e-12)= 1e-12;
        logMel= log(melEnergy);
        dctCoeffs= dct(logMel);
        c(:,fIndex)= dctCoeffs(2:(numCoeffs+1));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FIND BEST SPEAKER
function [bestID, minDist] = findBestSpeaker(mfccMat, speakerModels)
    bestID= 0;
    minDist= inf;
    for s=1:numel(speakerModels)
        cb= speakerModels{s};
        if isempty(cb), continue; end
        dVal= computeVQDistortion(mfccMat, cb);
        if dVal< minDist
            minDist= dVal;
            bestID= s;
        end
    end
end

function distVal= computeVQDistortion(mfccMat, codebook)
    [~, N]= size(mfccMat);
    total=0;
    for n=1:N
        vec= mfccMat(:,n);
        diff= codebook - vec;
        dists= sum(diff.^2,1);
        total= total+ min(dists);
    end
    distVal= total/N;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TRAIN VQ with LBG
function codebook= trainVQ_LBG(mfccMat, M)
    epsVal=0.01;
    distThresh=1e-3;
    [D,N]= size(mfccMat);

    cbook= mean(mfccMat,2);
    count=1;
    while count<M
        cbook= [cbook.*(1+epsVal), cbook.*(1-epsVal)];
        count= size(cbook,2);

        prevDist= inf;
        while true
            distMat= zeros(count,N);
            for i=1:count
                diffVal= mfccMat- cbook(:,i);
                distMat(i,:)= sum(diffVal.^2,1);
            end
            [~, nearest]= min(distMat,[],1);

            newCB= zeros(D,count);
            for i=1:count
                idx= (nearest==i);
                if any(idx)
                    newCB(:,i)= mean(mfccMat(:,idx),2);
                else
                    newCB(:,i)= cbook(:,i);
                end
            end

            distortion=0;
            for i=1:count
                idx= (nearest==i);
                if any(idx)
                    diffVal= mfccMat(:,idx)- newCB(:,i);
                    distortion= distortion+ sum(diffVal.^2,'all');
                end
            end
            distortion= distortion/N;

            if abs(prevDist-distortion)/distortion< distThresh
                cbook= newCB;
                break;
            else
                cbook= newCB;
                prevDist= distortion;
            end
        end
    end
    codebook= cbook;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% applyNotch
function yNotch= applyNotch(y, fs, centerFreq, width)
    fn= fs/2;
    freqRatio= centerFreq/fn;
    notchZeros= [exp(1i*pi*freqRatio), exp(-1i*pi*freqRatio)];
    notchPoles= (1-width)*notchZeros;
    b= poly(notchZeros);
    a= poly(notchPoles);
    yNotch= filter(b,a,y);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parse filename => speakerID (if your test naming is "s<number>.wav")
function ID= parseTrueID(fname)
    pat= regexp(fname, '^s(\d+)\.wav$','tokens','once');
    if isempty(pat)
        ID=0;
    else
        ID= str2double(pat{1});
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Ternary
function s= ternary(cond, sTrue, sFalse)
    if cond, s=sTrue; else s=sFalse; end
end
