function test10a_b_multiClass_speakerWord()
% test10a_b_multiClass_speakerWord
%
% This script demonstrates a "multi-class" approach to simultaneously
% identify both the speaker and the spoken word ("zero" or "twelve").
%
% Implementation:
%   1) We treat each (speakerID, wordType) as a unique class label.
%      For example, with 19 speakers, each speaker has 2 word types:
%        - zero
%        - twelve
%      Thus total classes = 19 * 2 = 38.
%
%   2) We assign each class an integer ID from 1..38:
%        classID = 2*(speakerID - 1) + wordCode
%        where
%          speakerID = i (1..19)
%          wordCode  = 1 for "zero", 2 for "twelve"
%
%   3) We build a trainList with all (i, wordCode) from:
%        D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Training\Zero_train{i}.wav
%        D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Training\Twelve_train{i}.wav
%      Each entry => { filePath, combinedClassID }.
%
%   4) We build a testList similarly from "Zero_Testing\Zero_test{i}" 
%      and "Twelve_Testing\Twelve_test{i}" => also assigned the same combinedClassID.
%
%   5) Train a codebook for each of the 38 classes via "runVQcodebook".
%   6) Test by picking min-dist among 38 codebooks => measure overall accuracy 
%      on identifying (speaker, word) combined.
%
% After classification, if bestID == trueClassID, we mark it as correct.
% We then compute overall multi-class accuracy.
%
% Usage:
%   Place this script in your MATLAB path or current folder,
%   confirm the folders exist with .wav files:
%     - Zero_Training\Zero_train{i}.wav  (i=1..19)
%     - Twelve_Training\Twelve_train{i}.wav
%     - Zero_Testing\Zero_test{i}.wav
%     - Twelve_Testing\Twelve_test{i}.wav
%   Then run:
%     >> test10a_b_multiClass_speakerWord
%
% The script will produce final accuracy on the combined test set.

    clear; clc; close all;

    nSpeakers = 19;  
    % We have 19 speakers, each says "zero" & "twelve" => total 2*n=38 classes.

    % ========== 1) Build trainList with combined classes ==========
    baseZeroTrain = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Training\';
    baseTwelveTrain= 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Training\';

    trainList = {};
    for i=1:nSpeakers
        classID_zero   = 2*(i-1) + 1;   % (speaker i, zero)
        pathZ = fullfile(baseZeroTrain, sprintf('Zero_train%d.wav', i));
        trainList{end+1,1} = { pathZ, classID_zero };

        classID_twelve = 2*(i-1) + 2;   % (speaker i, twelve)
        pathT = fullfile(baseTwelveTrain, sprintf('Twelve_train%d.wav', i));
        trainList{end+1,1} = { pathT, classID_twelve };
    end

    % ========== 2) Build testList (combined classes) ==========
    baseZeroTest    = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Testing\';
    baseTwelveTest  = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Testing\';

    testList = {};
    for i=1:nSpeakers
        cIDz = 2*(i-1)+1;   % (i, zero)
        fileZ = fullfile(baseZeroTest, sprintf('Zero_test%d.wav', i));
        testList{end+1,1} = {fileZ, cIDz};

        cIDt = 2*(i-1)+2;   % (i, twelve)
        fileT= fullfile(baseTwelveTest, sprintf('Twelve_test%d.wav', i));
        testList{end+1,1} = {fileT, cIDt};
    end

    % total training size= 2*nSpeakers, total testing size= 2*nSpeakers

    numFilters   = 20;
    numCoeffs    = 12;
    codebookSize = 8;

    disp('--- Training combined (speaker+word) classes => multi-class approach...');
    speakerModels = train_multiClass(trainList, numFilters, numCoeffs, codebookSize);

    disp('--- Testing combined classes => identify speaker + word...');
    [accuracy, ~] = test_multiClass(testList, speakerModels, numFilters, numCoeffs);

    fprintf('\nFinal multi-class accuracy (speaker + word) = %.2f%%\n', accuracy);
    disp('Done test10a_b_multiClass_speakerWord.');
end


%% train_multiClass
function speakerModels = train_multiClass(trainList, numFilters, numCoeffs, codebookSize)
    nTotal = size(trainList,1);

    classes = [];
    for row=1:nTotal
        item  = trainList{row};
        cID   = item{2};  
        if cID>length(classes), classes{cID}=[]; end
        wavPath = item{1};
        if exist(wavPath,'file')==2
            [mfccMat, ~] = computeMFCC_forFile(wavPath, numFilters, numCoeffs);
            classes{cID} = [classes{cID}, mfccMat]; 
        else
            warning('File missing => skip: %s', wavPath);
        end
    end

    nClass= length(classes);
    speakerModels= cell(nClass,1);

    for c=1:nClass
        if isempty(classes{c})
            speakerModels{c}=[];
            continue;
        end
        codebook = runVQcodebook(classes{c}, codebookSize);
        speakerModels{c}= codebook;
    end
end


%% test_multiClass
function [accuracy, predictions] = test_multiClass(testList, speakerModels, numFilters, numCoeffs)
    nTests = size(testList,1);
    predictions= zeros(nTests,1);
    correct=0;
    for t=1:nTests
        entry= testList{t};
        filePath= entry{1};
        realID  = entry{2};

        if exist(filePath,'file')==2
            [mfccTest, ~] = computeMFCC_forFile(filePath, numFilters, numCoeffs);
            [bestID, distVal] = findBestClass_demo(mfccTest, speakerModels);
            predictions(t)= bestID;

            isOk= (bestID==realID);
            if isOk, correct=correct+1; end
            fprintf('Test: %s => class#%d (true:%d), Dist=%.3f %s\n',...
                filePath, bestID, realID, distVal, ternary(isOk,'[OK]','[ERR]'));
        else
            warning('File not found => skip test: %s', filePath);
        end
    end

    used= sum(predictions>0);
    if used>0
        accuracy= (correct/used)*100;
    else
        accuracy=0;
    end
end

%% findBestClass_demo
function [bestID, distVal] = findBestClass_demo(mfccTest, speakerModels)
    bestID=0; distVal=inf;
    nClass= numel(speakerModels);
    for c=1:nClass
        cb= speakerModels{c};
        if isempty(cb), continue; end
        dVal= computeVQDistortion_demo(mfccTest, cb);
        if dVal< distVal
            distVal= dVal;
            bestID= c;
        end
    end
end

%% computeMFCC_forFile
function [mfccMat, fsOut] = computeMFCC_forFile(wavPath, numFilters, numCoeffs)
    [y, fsOut] = audioread(wavPath);
    mfccMat    = computeMFCC_fromSignal(y, fsOut, numFilters, numCoeffs);
end

%% computeMFCC_fromSignal
function c = computeMFCC_fromSignal(signal, fs, numFilters, numCoeffs)
    alpha=0.95;
    for i=length(signal):-1:2
        signal(i)= signal(i)- alpha*signal(i-1);
    end

    N=256; overlap=100; NFFT=512;
    [S,~,~] = stft(signal, fs,'Window',hamming(N),'OverlapLength',overlap,...
                   'FFTLength',NFFT,'FrequencyRange','onesided');
    ps= (abs(S).^2)/NFFT;

    melFB= melfb(numFilters, NFFT, fs);
    if size(melFB,2)~= size(ps,1)
        error('Dimension mismatch => melFB=%dx%d, ps=%dx%d',...
            size(melFB,1), size(melFB,2), size(ps,1), size(ps,2));
    end

    nFrames= size(ps,2);
    c= zeros(numCoeffs,nFrames);
    for fIndex=1:nFrames
        pcol= ps(:,fIndex);
        melE= melFB * pcol;
        melE(melE<1e-12)= 1e-12;
        logMel= log(melE);
        dctCoeffs= dct(logMel);
        c(:,fIndex)= dctCoeffs(2:(numCoeffs+1));
    end

    c= c - (mean(c,2)+1e-8);
end

%% runVQcodebook => LBG
function codebook= runVQcodebook(mfccMat, codebookSize)
    epsVal=0.01; distThresh=1e-3;
    [D,N]= size(mfccMat);
    cbook= mean(mfccMat,2);
    count=1;
    while count< codebookSize
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
                    dtmp= mfccMat(:,idx)- newCB(:,i);
                    distortion= distortion+ sum(dtmp.^2,'all');
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

%% computeVQDistortion_demo
function distVal= computeVQDistortion_demo(mfccMat, codebook)
    [~, N]= size(mfccMat);
    total=0;
    for n=1:N
        vec= mfccMat(:,n);
        diff= codebook- vec;
        dists= sum(diff.^2,1);
        total= total+ min(dists);
    end
    distVal= total/N;
end

%% melfb => mel filter bank
function m = melfb(p, n, fs)
    f0  = 700 / fs;
    fn2 = floor(n / 2);
    lr  = log(1 + 0.5 / f0) / (p + 1);

    bl = n*(f0*(exp([0,1,p,p+1]*lr)-1));
    b1 = floor(bl(1))+1;
    b2 = ceil(bl(2));
    b3 = floor(bl(3));
    b4 = min(fn2,ceil(bl(4)))-1;

    pf = log(1 + (b1:b4)/n/f0)/lr;
    fp = floor(pf);
    pm = pf - fp;

    r = [fp(b2:b4),1+fp(1:b3)];
    c = [b2:b4,    1:b3]+1;
    v = 2*[1-pm(b2:b4), pm(1:b3)];
    m = sparse(r,c,v,p,1+fn2);
end

%% ternary helper
function s= ternary(cond, sTrue, sFalse)
    if cond
        s=sTrue;
    else
        s=sFalse;
    end
end

