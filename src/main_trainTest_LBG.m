function main_trainTest_LBG()
% main_trainTest_LBG
%
% A single integrated main code that:
%   1) Enumerates "Train" folder => sX.wav => parse X => compute MFCC => run LBG => store codebook
%   2) Enumerates "Test" folder => sY.wav => parse Y => compute MFCC => match codebook => measure accuracy
%
% Matching method: minimal average distortion (sum of min-distance^2 from frames to codewords).
% We do not use getFile; we do direct "audioread(...)" on each file path.
%
% Usage:
%   1) Place s1.wav..s8.wav (or more) in "Train", and s1.wav..s8.wav in "Test" if you have a test set.
%   2) Adjust "trainFolder" and "testFolder" if needed
%   3) Run >> main_trainTest_LBG
%
% Author: GPT, all in one version.

    clear; clc; close all;

    %% Step 0: Setup paths
    trainFolder = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Train';
    testFolder  = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Test';
    codebookSize= 8;  % # of centroids for LBG

    %% Step 1: Build training data => For each sX.wav in trainFolder => parse ID => gather MFCC
    [speakerModels, spkListTrain] = trainCodebooks(trainFolder, codebookSize);

    %% Step 2: Test => For each sX.wav in testFolder => parse ID => compute MFCC => match w/ codebooks
    testAccuracy = testCodebooks(testFolder, speakerModels);

    fprintf('\n** Final Recognition Accuracy = %.2f%% **\n', testAccuracy);
end

%% =========================================================
function [speakerModels, spkList] = trainCodebooks(folderPath, codebookSize)
% trainCodebooks:
%   - enumerates folderPath => sX.wav => parse ID => combine multiple files if same ID
%   - compute MFCC => LBG => store codebook
% Output:
%   speakerModels : cell array => speakerModels{id} is codebook
%   spkList       : numeric array of IDs found in train folder

    files = dir(fullfile(folderPath, 's*.wav'));
    if isempty(files)
        error('No s*.wav found in: %s', folderPath);
    end

    % parse ID from filename => s(\d+).wav
    % we group wave paths by ID
    dataMap = containers.Map('KeyType','double','ValueType','any');

    for i=1:length(files)
        fname= files(i).name; 
        fpath= fullfile(files(i).folder, fname);
        token= regexp(fname,'^s(\d+)\.wav$','tokens','once');
        if ~isempty(token)
            spkID = str2double(token{1});
            if ~isKey(dataMap, spkID)
                dataMap(spkID) = {};
            end
            arr= dataMap(spkID);
            arr{end+1}= fpath;
            dataMap(spkID)= arr;
        else
            fprintf('Skipping non-matching file: %s\n', fname);
        end
    end

    spkList = sort(cell2mat(keys(dataMap)));
    speakerModels = cell(max(spkList), 1);

    for i=1:length(spkList)
        spkID= spkList(i);
        waveList= dataMap(spkID);  % cell of file paths

        % Combine MFCC from all files that belong to spkID
        allMFCC= [];
        for k=1:length(waveList)
            waveFile= waveList{k};
            [y, fs] = audioread(waveFile);
            if size(y,2)>1
                y= y(:,1);
            end
            % preproc
            y= y - mean(y);
            pk= max(abs(y));
            if pk>1e-12
                y= y/pk;
            end

            mfccMat= computeMFCC_all(y, fs);  % (numCoeffs x frames)
            allMFCC= [allMFCC, mfccMat];      % accumulate horizontally
        end
        % shape => (numCoeffs x totalFrames)
        X= allMFCC';  % => (#totalFrames x numCoeffs)

        % LBG 
        codebook= runLBG(X, codebookSize);
        speakerModels{spkID}= codebook;

        fprintf('Trained spkID=%d (#files=%d) => codebookSize=%d\n',...
            spkID, length(waveList), codebookSize);
    end
end

%% =========================================================
function accuracy= testCodebooks(folderPath, speakerModels)
% testCodebooks:
%   - enumerates folderPath => sX.wav => parse ID => compute MFCC => 
%     do min-dist match => count correct => output accuracy

    files = dir(fullfile(folderPath, 's*.wav'));
    if isempty(files)
        warning('No s*.wav found in Test folder => 0% accuracy');
        accuracy=0; return;
    end

    correct=0; total=0;

    for i=1:length(files)
        fname= files(i).name; 
        fpath= fullfile(files(i).folder, fname);
        token= regexp(fname,'^s(\d+)\.wav$','tokens','once');
        if isempty(token)
            fprintf('Skipping non-matching file: %s\n', fname);
            continue;
        end

        trueID= str2double(token{1});
        [y, fs] = audioread(fpath);
        if size(y,2)>1
            y= y(:,1);
        end

        y= y - mean(y);
        pk= max(abs(y));
        if pk>1e-12
            y= y/pk;
        end

        mfccMat= computeMFCC_all(y, fs);
        Xtest  = mfccMat';

        % find best speaker by min-dist
        bestID= 0; 
        bestDist= inf;
        for cID=1:length(speakerModels)
            cb= speakerModels{cID};
            if isempty(cb), continue; end
            dVal= computeVQdist(Xtest, cb);
            if dVal< bestDist
                bestDist= dVal;
                bestID= cID;
            end
        end

        isOk= (bestID==trueID);
        if isOk, correct=correct+1; end
        total= total+1;

        fprintf('Test file=%s => true=%d => predict=%d => dist=%.3f %s\n',...
            fname, trueID, bestID, bestDist, ternary(isOk,'[OK]','[ERR]'));
    end

    if total>0
        accuracy= (correct/total)*100;
    else
        accuracy=0;
    end
    fprintf('\nTest completed => correct=%d / %d => accuracy=%.2f%%\n',...
        correct, total, accuracy);
end

function distVal= computeVQdist(X, codebook)
% X => (#frames x dim)
% codebook => (#centroids x dim)
    N= size(X,1);
    dists= zeros(N,1);
    for i=1:N
        diffVal= codebook - X(i,:);
        dist2= sum(diffVal.^2,2);
        dists(i)= min(dist2);
    end
    distVal= sum(dists)/N;
end

function s= ternary(cond,sTrue,sFalse)
    if cond
        s= sTrue;
    else
        s= sFalse;
    end
end

%% =========================================================
function codebook = runLBG(X, codebookSize)
% runLBG => Linde-Buzo-Gray vector quantization
%
% X => (#samples x #dimension)
% codebook => (#centroids x #dimension)
    epsVal=0.01; distThresh=1e-4;
    [N, dim]= size(X);

    cbook= mean(X,1); 
    count=1;
    while count< codebookSize
        cbook= [cbook.*(1+epsVal); cbook.*(1-epsVal)];
        count= size(cbook,1);

        prevDist=inf;
        while true
            distMat= zeros(count,N);
            for ci=1:count
                diffVal= X - cbook(ci,:);
                distMat(ci,:)= sum(diffVal.^2,2);
            end
            [~, nearest]= min(distMat,[],1);

            newCB= zeros(count,dim);
            for ci=1:count
                idx= (nearest==ci);
                if any(idx)
                    newCB(ci,:)= mean(X(idx,:),1);
                else
                    newCB(ci,:)= cbook(ci,:);
                end
            end

            distortion=0;
            for ci=1:count
                idx= (nearest==ci);
                if any(idx)
                    dv= X(idx,:) - newCB(ci,:);
                    distortion= distortion+ sum(dv.^2,'all');
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

%% =========================================================
function mfccMat = computeMFCC_all(y, fs)
% minimal example => c2..c13 => (12 x frames)
    N=256; overlap=128; NFFT=512;
    [S,~,~]= spectrogram(y, hamming(N), overlap, NFFT, fs);
    powerSpec= abs(S).^2;

    numFilters=26;
    melFB_ = melfb(numFilters, NFFT, fs);
    if size(melFB_,2)~= size(powerSpec,1)
        error('Dimension mismatch => melFB vs STFT freqBins');
    end

    melSpec= melFB_* powerSpec;
    melSpec(melSpec<1e-12)=1e-12;
    logMel= log(melSpec);

    dctAll= dct(logMel);
    mfccMat= dctAll(2:13,:);
end

function m= melfb(p, n, fs)
% standard melFB => (p x (n/2+1))
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
