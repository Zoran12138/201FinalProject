function test10a_zeroTrain_twelveTest()
% test10a_zeroTrain_twelveTest
%
% This script trains a speaker recognition model using "zero" audios from:
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Training\Zero_train1.wav..Zero_train19.wav
%
% Then it tests that model on "twelve" audios from:
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Testing\Twelve_test1.wav..Twelve_test19.wav
%
% Each WAV file (1..19) is treated as one speaker ID = i. 
% We measure how well a system trained on "zero" can recognize the same 19 speakers saying "twelve."
%
% Steps:
%   1) Build a trainList for zero_train1..19.wav 
%   2) Build a testList for twelve_test1..19.wav 
%   3) Train all (19) speakers on zero data
%   4) Test on the same 19 IDs but "twelve" data
%   5) Print final accuracy
%
% Usage:
%   Place this script in your MATLAB path,
%   ensure the WAV files exist in:
%       D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Training\Zero_train1.wav..19
%       D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Testing\Twelve_test1.wav..19
%   Then run:
%       >> test10a_zeroTrain_twelveTest
%
% The code uses a basic LBG VQ approach with MFCC. 
% Results reflect how "zero" data training generalizes to "twelve" testing.

    clear; clc; close all;

    % We have 19 total IDs (speaker1..speaker19).
    nSpeakers = 19;

    % 1) Build trainList => zero_train1..19 => "Zero_Training"
    baseZeroTrain = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Training\';
    trainList = cell(nSpeakers,1);
    for i=1:nSpeakers
        trainList{i} = fullfile(baseZeroTrain, sprintf('Zero_train%d.wav', i));
    end

    % 2) Build testList => twelve_test1..19 => "Twelve_Testing"
    % ID matches i => so test is ID=i
    baseTwelveTest = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Testing\';
    testList = cell(nSpeakers,1);
    for i=1:nSpeakers
        filePath = fullfile(baseTwelveTest, sprintf('Twelve_test%d.wav', i));
        trueID   = i;
        testList{i} = { filePath, trueID };
    end

    % 3) Train => codebook size, mel filters, MFCC dimension
    numFilters   = 20;
    numCoeffs    = 12;
    codebookSize = 8;

    disp('--- Training on Zero_train1..19 (19) in Zero_Training folder ---');
    speakerModels = train_speakers_demo(trainList, numFilters, numCoeffs, codebookSize);

    % 4) Test => "twelve" 
    disp('--- Testing on Twelve_test1..19 from Twelve_Testing folder ---');
    [accuracy, ~] = test_speakers_demo(testList, speakerModels, numFilters, numCoeffs);

    fprintf('\nSystem accuracy (trained on "zero", tested on "twelve") = %.2f%%\n', accuracy);
    disp('Done test10a_zeroTrain_twelveTest.');
end


%% train_speakers_demo
function speakerModels = train_speakers_demo(trainList, numFilters, numCoeffs, codebookSize)
    nSpeak= numel(trainList);
    speakerModels= cell(nSpeak,1);
    for i=1:nSpeak
        fPath= trainList{i};
        if exist(fPath,'file')==2
            [mfccMat, ~] = computeMFCC_forFile(fPath, numFilters, numCoeffs);
            codebook = runVQcodebook(mfccMat, codebookSize);
            speakerModels{i}= codebook;
        else
            warning('File not found => skip training: %s', fPath);
            speakerModels{i}=[];
        end
    end
end

%% test_speakers_demo
function [accuracy, predictions] = test_speakers_demo(testList, speakerModels, numFilters, numCoeffs)
    nTests= numel(testList);
    predictions= zeros(nTests,1);
    correct=0;
    for t=1:nTests
        info= testList{t};
        filePath= info{1};
        realID  = info{2};

        if exist(filePath,'file')==2
            [mfccTest, ~] = computeMFCC_forFile(filePath, numFilters, numCoeffs);
            [bestID, distVal] = findBestSpeaker_demo(mfccTest, speakerModels);
            predictions(t)= bestID;

            isOk= (bestID==realID);
            if isOk, correct=correct+1; end
            fprintf('Test: %s => spk#%d (true:%d), Dist=%.3f %s\n',...
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

%% computeMFCC_forFile => read WAV => do MFCC => return [mfccMat, fsOut]
function [mfccMat, fsOut] = computeMFCC_forFile(wavPath, numFilters, numCoeffs)
    [y, fsOut] = audioread(wavPath);
    mfccMat    = computeMFCC_fromSignal(y, fsOut, numFilters, numCoeffs);
end

%% computeMFCC_fromSignal => stft -> powerSpec -> melFB -> dct => MFCC
function c = computeMFCC_fromSignal(signal, fs, numFilters, numCoeffs)
    alpha=0.95;
    for i=length(signal):-1:2
        signal(i)= signal(i)- alpha*signal(i-1);
    end

    N=256; overlap=100; NFFT=512;
    [S,~,~] = stft(signal, fs, 'Window',hamming(N),'OverlapLength',overlap,...
                   'FFTLength',NFFT,'FrequencyRange','onesided');
    ps= (abs(S).^2)/NFFT; 

    melFB = melfb(numFilters, NFFT, fs);
    if size(melFB,2)~= size(ps,1)
        error('Dimension mismatch => melFB=%dx%d, ps=%dx%d',...
            size(melFB,1), size(melFB,2), size(ps,1), size(ps,2));
    end

    nFrames= size(ps,2);
    c= zeros(numCoeffs, nFrames);
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

%% runVQcodebook => minimal LBG approach
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

%% findBestSpeaker_demo => minimal distance among codebooks
function [bestID, distVal]= findBestSpeaker_demo(mfccTest, speakerModels)
    bestID=0; distVal=inf;
    for sp=1:numel(speakerModels)
        cb= speakerModels{sp};
        if isempty(cb), continue; end
        dVal= computeVQDistortion_demo(mfccTest, cb);
        if dVal< distVal
            distVal= dVal;
            bestID= sp;
        end
    end
end

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

    bl = n * (f0 * (exp([0, 1, p, p+1] * lr) - 1));
    b1 = floor(bl(1)) + 1;
    b2 = ceil(bl(2));
    b3 = floor(bl(3));
    b4 = min(fn2, ceil(bl(4))) -1;

    pf = log(1 + (b1:b4)/n/f0)/ lr;
    fp = floor(pf);
    pm = pf - fp;

    r = [fp(b2:b4),     1+fp(1:b3)];
    c = [b2:b4,         1:b3] +1;
    v = 2 * [1-pm(b2:b4), pm(1:b3)];
    m = sparse(r,c,v,p, 1+fn2);
end

%% ternary => small utility
function s= ternary(cond, sTrue, sFalse)
    if cond
        s=sTrue;
    else
        s=sFalse;
    end
end
