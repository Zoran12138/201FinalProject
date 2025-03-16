function test8_notch_experiment_trainClean_testNotch()
% test8_notch_experiment_trainClean_testNotch
%
% Demonstration of Test 8 approach where:
%   - Train on "clean" data in Train folder (no notch)
%   - Test on "Test" folder, first no filter => baseline, 
%     then apply notch => measure accuracy => see how system handles that distortion.
%
% Folders:
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Train => raw .wav
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Test  => raw .wav (for baseline),
%        but at run-time we apply notch => see if accuracy drops
%
% Requirements:
%   - All sX.wav in Train + Test => naming "s(\d+).wav"
%   - N=256, Mstep=100, NFFT=512 => freqRows=257 => melFB => (numFilters x 257)
%   - We do 12 MFCC (c2..c13)
%
% Usage:
%   >> test8_notch_experiment_trainClean_testNotch

    clear; clc; close all;

    trainDir = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Train';
    testDir  = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Test';

    % STFT + MFCC config
    N=256; Mstep=100; NFFT=512;
    numFilters=20; numCoeffs=12;
    codebookSize=8;  % LBG codebook

    % ========== Step 1: Train (no filter) ==========
    fprintf('=== Training on clean data (no notch) from: %s\n', trainDir);
    speakerModels= train_noFilter(trainDir, N, Mstep, NFFT, numFilters, numCoeffs, codebookSize);

    % ========== Step 2: Baseline Test (no filter) ==========
    fprintf('\n=== Baseline test => no filter on test set ===\n');
    accBase = test_withNotch(testDir, speakerModels, N, Mstep, NFFT, numFilters, numCoeffs, 0, 0);
    fprintf('Baseline accuracy= %.2f%%\n\n', accBase);

    % ========== Step 3: Notch test => multiple freq ==========
    centerFreqs= [500, 1000, 2000];
    notchWidth= 0.05;  % define radius= 1-0.05= 0.95
    results= zeros(size(centerFreqs));

    for i=1:numel(centerFreqs)
        fo= centerFreqs(i);
        fprintf('--- Testing with Notch@%d Hz on test set ---\n', fo);
        accNotch= test_withNotch(testDir, speakerModels, N, Mstep, NFFT, numFilters, numCoeffs, fo, notchWidth);
        results(i)= accNotch;
        fprintf('Notch@%d => accuracy= %.2f%%\n\n', fo, accNotch);
    end

    % ========== Step 4: Summary ==========
    fprintf('==== Final Summary ====\n');
    fprintf('Baseline (no filter) : %.2f%%\n', accBase);
    for i=1:numel(centerFreqs)
        fprintf('Notch@%4d => %.2f%%\n', centerFreqs(i), results(i));
    end
    fprintf('=======================\n');
end


%% =========================================================
function speakerModels= train_noFilter(folderTrain, N, Mstep, NFFT, numFilters, numCoeffs, codebookSize)
% train_noFilter => read "sX.wav" from train folder => no notch => compute MFCC => LBG => speakerModels

    fList= dir(fullfile(folderTrain,'s*.wav'));
    if isempty(fList)
        error('No s*.wav found in %s => cannot train', folderTrain);
    end

    % group wave files by ID
    dataMap= containers.Map('KeyType','double','ValueType','any');
    for i=1:length(fList)
        fname= fList(i).name; 
        fpath= fullfile(fList(i).folder, fname);
        token= regexp(fname,'^s(\d+)\.wav$','tokens','once');
        if ~isempty(token)
            spkID= str2double(token{1});
            if ~isKey(dataMap, spkID)
                dataMap(spkID)= {};
            end
            arr= dataMap(spkID);
            arr{end+1}= fpath;
            dataMap(spkID)= arr;
        else
            fprintf('Skipping non-matching train file: %s\n', fname);
        end
    end

    spkIDs= sort(cell2mat(keys(dataMap)));
    speakerModels= cell(max(spkIDs),1);

    for i=1:length(spkIDs)
        spkID= spkIDs(i);
        waveList= dataMap(spkID);

        allMFCC= [];
        for k=1:length(waveList)
            wFile= waveList{k};
            [y, fs]= audioread(wFile);
            if size(y,2)>1
                y= y(:,1);
            end
            % preproc => no notch
            y= y - mean(y);
            pk= max(abs(y));
            if pk>1e-12
                y= y/pk;
            end

            mfccMat= audio2mfcc(y, fs, N, Mstep, NFFT, numFilters, numCoeffs);
            allMFCC= [allMFCC, mfccMat];
        end
        % train LBG
        X= allMFCC';  % (#frames x numCoeffs)
        codebook= runLBG(X, codebookSize);

        speakerModels{spkID}= codebook;
        fprintf('Train spkID=%d => #files=%d => codebookSize=%d\n', spkID, length(waveList), codebookSize);
    end
end

%% =========================================================
function accuracy= test_withNotch(folderTest, speakerModels, N, Mstep, NFFT, numFilters, numCoeffs, fo, notchWidth)
% test_withNotch => read sX.wav from test folder => apply notch if fo>0 => compute MFCC => match
% Return => accuracy

    fList= dir(fullfile(folderTest,'s*.wav'));
    if isempty(fList)
        warning('No s*.wav in %s => 0%% accuracy', folderTest);
        accuracy=0; return;
    end

    correct=0; total=0;
    for i=1:length(fList)
        fname= fList(i).name;
        fpath= fullfile(fList(i).folder, fname);
        token= regexp(fname,'^s(\d+)\.wav$','tokens','once');
        if isempty(token)
            fprintf('Skipping non-matching test file: %s\n', fname);
            continue;
        end
        trueID= str2double(token{1});

        [y, fs]= audioread(fpath);
        if size(y,2)>1
            y= y(:,1);
        end
        % preproc
        y= y- mean(y);
        pk= max(abs(y));
        if pk>1e-12
            y= y/pk;
        end
        if fo>0
            y= applyNotch(y, fs, fo, notchWidth);
        end

        mfccTest= audio2mfcc(y, fs, N, Mstep, NFFT, numFilters, numCoeffs);
        [bestID, distVal]= findBestSpeaker(mfccTest, speakerModels);

        isOK= (bestID==trueID);
        if isOK, correct= correct+1; end
        total= total+1;

        label= ternary(fo>0, sprintf('Notch@%d', fo), 'NoFilter');
        fprintf('[%s] %s => spk#%d (true:%d), dist=%.2f %s\n',...
            label, fname, bestID, trueID, distVal, ternary(isOK,'[OK]','[ERR]'));
    end

    if total>0
        accuracy= (correct/total)*100;
    else
        accuracy= 0;
    end
    fprintf('Test done => correct=%d/%d => %.2f%%\n', correct, total, accuracy);
end

%% =========================================================
%% AUDIO -> powerspec -> MFCC
function mfccMat= audio2mfcc(y, fs, N, Mstep, NFFT, numFilters, numCoeffs)
    ps= audio2powerspec(y, fs, N, Mstep, NFFT);
    mfccMat= myMfcc(ps, fs, numFilters, numCoeffs, NFFT);
end

function ps= audio2powerspec(y, fs, N, Mstep, NFFT)
    alpha=0.95;
    for i=length(y):-1:2
        y(i)= y(i)- alpha*y(i-1);
    end
    overlap= N- Mstep;
    w= hamming(N);

    [S,F,T]= stft(y, fs,'Window',w,'OverlapLength',overlap,'FFTLength',NFFT,...
                  'FrequencyRange','onesided');
    ps= (abs(S).^2)./NFFT;
end

function c= myMfcc(ps, fs, numFilters, numCoeffs, NFFT)
    [freqRows, nFrames]= size(ps);
    halfN= (NFFT/2)+1;
    if freqRows~=halfN
        error('myMfcc: freqRows=%d, expecting %d', freqRows, halfN);
    end

    melFB= melfb(numFilters, NFFT, fs);
    if size(melFB,2)~= freqRows
        error('Mismatch => melFB(%dx%d), ps(%dx%d)', ...
            size(melFB,1), size(melFB,2), freqRows, nFrames);
    end

    c= zeros(numCoeffs, nFrames);
    for fIndex=1:nFrames
        pcol= ps(:,fIndex);
        melE= melFB* pcol;
        melE(melE<1e-12)=1e-12;
        logMel= log(melE);
        dctCoeffs= dct(logMel);
        c(:,fIndex)= dctCoeffs(2:(numCoeffs+1));
    end
end

%% Melfb
function m= melfb(p,n,fs)
    f0= 700/fs;
    fn2= floor(n/2);
    lr= log(1+0.5/f0)/(p+1);

    bl= n*(f0*(exp([0,1,p,p+1]*lr)-1));
    b1= floor(bl(1))+1;
    b2= ceil(bl(2));
    b3= floor(bl(3));
    b4= min(fn2, ceil(bl(4)))-1;

    pf= log(1+ (b1:b4)/(n*f0))/lr;
    fp= floor(pf);
    pm= pf- fp;

    r= [fp(b2:b4), 1+fp(1:b3)];
    c= [b2:b4, 1:b3]+1;
    v= 2*[1-pm(b2:b4), pm(1:b3)];
    m= sparse(r,c,v,p, fn2+1);
end

%% =========================================================
%% FIND BEST SPEAKER
function [bestID, distVal]= findBestSpeaker(mfccMat, speakerModels)
% mfccMat => (numCoeffs x frames)
% codebook => (#centroids x numCoeffs)
    bestID=0; distVal= inf;
    for i=1:numel(speakerModels)
        cb= speakerModels{i};
        if isempty(cb), continue; end
        dVal= computeVQDist(mfccMat, cb);
        if dVal< distVal
            distVal= dVal;
            bestID= i;
        end
    end
end

function val= computeVQDist(mfccMat, codebook)
    [C, frames]= size(mfccMat);
    total= 0;
    for f=1:frames
        vec= mfccMat(:,f);
        diff= codebook - vec;  % (#centroids x numCoeffs) - (numCoeffs)
        dist2= sum(diff.^2,1);
        total= total+ min(dist2);
    end
    val= total/ frames;
end

%% =========================================================
%% LBG
function codebook= runLBG(X, codebookSize)
% X => (#frames x #coeff)
    epsVal=0.01; distThresh=1e-3;
    [N, D]= size(X);

    cbook= mean(X,1)';  % => (D x 1)
    count=1;
    while count< codebookSize
        cbook= [cbook.*(1+epsVal), cbook.*(1-epsVal)];
        count= size(cbook,2);

        prevDist= inf;
        while true
            distMat= zeros(count,N);
            for i=1:count
                diffVal= X - cbook(:,i)';  % NxD
                distMat(i,:)= sum(diffVal.^2,2);
            end
            [~, nearest]= min(distMat,[],1);

            newCB= zeros(D,count);
            for i=1:count
                idx= (nearest==i);
                if any(idx)
                    newCB(:,i)= mean(X(idx,:),1)';
                else
                    newCB(:,i)= cbook(:,i);
                end
            end

            distortion=0;
            for i=1:count
                idx= (nearest==i);
                if any(idx)
                    diffV= X(idx,:) - newCB(:,i)';
                    distortion= distortion+ sum(diffV.^2,'all');
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
    codebook= cbook;  % => (D x codebookSize)
end

%% =========================================================
%% NOTCH
function yNotch= applyNotch(y, fs, centerFreq, notchWidth)
    fn= fs/2;
    freqRatio= centerFreq/fn;
    theta= pi* freqRatio;
    r= 1- notchWidth;

    zerosCplx= [exp(1i*theta), exp(-1i*theta)];
    polesCplx= r* zerosCplx;

    b= poly(zerosCplx);  % => e.g. [1, -2cos(theta), 1]
    a= poly(polesCplx);

    yNotch= filter(b,a,y);
end

%% =========================================================
%% Ternary
function s= ternary(cond, sTrue, sFalse)
    if cond, s=sTrue; else s=sFalse; end
end

