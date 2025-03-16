function main_9_trainTest()
% main_9_trainTest
%
% Demonstration of training + testing speaker recognition 
% with newly added directories:
%   "D:\Program Files\Polyspace\R2021a\bin\EEC201\9train"
%   "D:\Program Files\Polyspace\R2021a\bin\EEC201\9test"
%
% Steps:
%   1) We define STFT & MFCC parameters
%   2) Train from 9train => gather all sX.wav => parse X => combine => LBG => speakerModels
%   3) Test from 9test => parse X => MFCC => find best speaker => measure accuracy
%
% Usage:
%   >> main_9_trainTest
%
% Author: GPT

    clear; clc; close all;

    trainDir= 'D:\Program Files\Polyspace\R2021a\bin\EEC201\9train';
    testDir = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\9test';

    % STFT + MFCC config
    N=256; Mstep=100; NFFT=512;
    numFilters=20; numCoeffs=12;
    codebookSize=8;  % LBG codebook

    % =========== (1) Train =========== 
    fprintf('=== Step 1: Train from %s ===\n', trainDir);
    speakerModels= train_9_lbg(trainDir, N, Mstep, NFFT, numFilters, numCoeffs, codebookSize);

    % =========== (2) Test =========== 
    fprintf('\n=== Step 2: Test from %s ===\n', testDir);
    acc= test_9_lbg(testDir, speakerModels, N, Mstep, NFFT, numFilters, numCoeffs);

    fprintf('\nFinal recognition accuracy = %.2f%%\n', acc);
end

%% =========================================================
function speakerModels= train_9_lbg(folderTrain, N, Mstep, NFFT, numFilters, numCoeffs, codebookSize)
% train_9_lbg => enumerates folderTrain => parse ID => combine => LBG => speakerModels
    fList= dir(fullfile(folderTrain,'s*.wav'));
    if isempty(fList)
        error('No s*.wav found in %s => cannot train', folderTrain);
    end

    % We use a dataMap => spkID => cell of waveFilePaths
    dataMap= containers.Map('KeyType','double','ValueType','any');

    for i=1:length(fList)
        fname= fList(i).name;
        fpath= fullfile(fList(i).folder, fname);
        token= regexp(fname,'^s(\d+)\.wav$','tokens','once');
        if isempty(token)
            fprintf('Skipping non-matching file: %s\n', fname);
            continue;
        end
        spkID= str2double(token{1});
        if ~isKey(dataMap, spkID)
            dataMap(spkID)= {};
        end
        arr= dataMap(spkID);
        arr{end+1}= fpath;
        dataMap(spkID)= arr;
    end

    spkIDs= sort(cell2mat(keys(dataMap)));
    speakerModels= cell(max(spkIDs),1);

    for s=1:length(spkIDs)
        spkID= spkIDs(s);
        waveList= dataMap(spkID);
        allMFCC= [];

        for k=1:length(waveList)
            wFile= waveList{k};
            [y, fs]= audioread(wFile);
            if size(y,2)>1
                y= y(:,1);
            end
            % preproc => remove DC + norm
            y= y - mean(y);
            pk= max(abs(y));
            if pk>1e-12
                y= y/pk;
            end

            mfccMat= audio2mfcc_9(y, fs, N, Mstep, NFFT, numFilters, numCoeffs);
            allMFCC= [allMFCC, mfccMat];
        end
        X= allMFCC';  % => (#frames x #coeffs)
        codebook= runLBG_9(X, codebookSize);
        speakerModels{spkID}= codebook;

        fprintf('Train spkID=%d => #files=%d => codebookSize=%d\n',...
            spkID, length(waveList), codebookSize);
    end
end

%% =========================================================
function accuracy= test_9_lbg(folderTest, speakerModels, N, Mstep, NFFT, numFilters, numCoeffs)
% test_9_lbg => enumerates folderTest => parse ID => compute MFCC => match => measure accuracy

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
            fprintf('Skipping non-matching file: %s\n', fname);
            continue;
        end

        trueID= str2double(token{1});

        [y, fs]= audioread(fpath);
        if size(y,2)>1
            y= y(:,1);
        end
        y= y- mean(y);
        pk= max(abs(y));
        if pk>1e-12
            y= y/pk;
        end

        mfccTest= audio2mfcc_9(y, fs, N, Mstep, NFFT, numFilters, numCoeffs);
        [bestID, distVal]= findBestSpeaker_9(mfccTest, speakerModels);

        isOk= (bestID==trueID);
        if isOk, correct= correct+1; end
        total= total+1;

        fprintf('%s => spk#%d (true:%d), dist=%.3f %s\n',...
            fname, bestID, trueID, distVal, ternary_9(isOk,'[OK]','[ERR]'));
    end

    if total>0
        accuracy= (correct/total)*100;
    else
        accuracy=0;
    end
    fprintf('Test done => correct=%d / %d => %.2f%%\n', correct, total, accuracy);
end

%% =========================================================
%% AUDIO -> MFCC
function mfccMat= audio2mfcc_9(y, fs, N, Mstep, NFFT, numFilters, numCoeffs)
    ps= audio2powerspec_9(y, fs, N, Mstep, NFFT);
    mfccMat= myMfcc_9(ps, fs, numFilters, numCoeffs, NFFT);
end

function ps= audio2powerspec_9(y, fs, N, Mstep, NFFT)
    alpha=0.95;
    for i=length(y):-1:2
        y(i)= y(i)- alpha*y(i-1);
    end
    overlap= N- Mstep;
    w= hamming(N);

    [S, ~, ~]= stft(y, fs,'Window',w,'OverlapLength',overlap,'FFTLength',NFFT,...
                    'FrequencyRange','onesided');
    ps= (abs(S).^2)./ NFFT;
end

function c= myMfcc_9(ps, fs, numFilters, numCoeffs, NFFT)
    [freqRows, nFrames]= size(ps);
    halfN= (NFFT/2)+1;
    if freqRows~= halfN
        error('myMfcc_9: freqRows=%d, expect %d => stft mismatch', freqRows, halfN);
    end

    melFB= melfb_9(numFilters, NFFT, fs);
    if size(melFB,2)~= freqRows
        error('Mismatch => melFB(%dx%d), ps(%dx%d)', ...
            size(melFB,1), size(melFB,2), freqRows, nFrames);
    end

    c= zeros(numCoeffs, nFrames);
    for fIndex=1:nFrames
        col= ps(:,fIndex);
        melE= melFB* col;
        melE(melE<1e-12)=1e-12;
        logMel= log(melE);
        dctC= dct(logMel);
        c(:,fIndex)= dctC(2:(numCoeffs+1));
    end
end

function m= melfb_9(p, n, fs)
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
    c= [b2:b4,     1:b3]+1;
    v= 2*[1-pm(b2:b4), pm(1:b3)];
    m= sparse(r,c,v,p, fn2+1);
end

%% =========================================================
%% LBG
function codebook= runLBG_9(X, codebookSize)
% X => (#frames x #coeffs)
    epsVal=0.01; distThresh=1e-3;
    [N, D]= size(X);

    % start centroid => mean => (D x 1)
    cbook= mean(X,1)';  
    count=1;
    while count< codebookSize
        cbook= [cbook.*(1+epsVal), cbook.*(1-epsVal)];
        count= size(cbook,2);

        prevDist= inf;
        while true
            distMat= zeros(count,N);
            for ci=1:count
                diffVal= X - cbook(:,ci)'; % => NxD
                distMat(ci,:)= sum(diffVal.^2,2);
            end
            [~, nearest]= min(distMat,[],1);

            newCB= zeros(D,count);
            for ci=1:count
                idx= (nearest==ci);
                if any(idx)
                    newCB(:,ci)= mean(X(idx,:),1)';
                else
                    newCB(:,ci)= cbook(:,ci);
                end
            end

            distortion=0;
            for ci=1:count
                idx= (nearest==ci);
                if any(idx)
                    diffV= X(idx,:) - newCB(:,ci)';
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
    codebook= cbook; % => (D x codebookSize)
end

%% =========================================================
%% FIND BEST SPEAKER
function [bestID, distVal]= findBestSpeaker_9(mfccMat, speakerModels)
% mfccMat => (numCoeffs x #frames)
    bestID=0; distVal= inf;

    for i=1:numel(speakerModels)
        cb= speakerModels{i};
        if isempty(cb), continue; end
        dVal= computeVQDist_9(mfccMat, cb);
        if dVal< distVal
            distVal= dVal;
            bestID= i;
        end
    end
end

function val= computeVQDist_9(mfccMat, codebook)
    [C, frames]= size(mfccMat);
    total=0;
    for f=1:frames
        vec= mfccMat(:,f);
        diff= codebook - vec;  % => (C x #centroids) - (C x 1)
        dist2= sum(diff.^2,1);
        total= total+ min(dist2);
    end
    val= total/frames;
end

%% =========================================================
%% Ternary
function s= ternary_9(cond,sTrue,sFalse)
    if cond
        s= sTrue;
    else
        s= sFalse;
    end
end
