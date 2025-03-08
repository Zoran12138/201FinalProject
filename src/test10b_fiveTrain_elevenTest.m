function test10b_fiveTrain_elevenTest()
% test10b_fiveTrain_elevenTest
%
% This script trains a speaker recognition model using "five" data
% from 23 students, then tests that model on "eleven" data 
% from those same 23 students.
%
% Directory structure (23 speakers: s1..s23):
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Five_Training\s{i}.wav
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Eleven_Test\s{i}.wav
%
% Steps:
%   1) Train codebooks for 23 speakers using Five_Training\s{i}.wav
%   2) Test on Eleven_Test\s{i}.wav => see if it can identify speaker i
%   3) Print final accuracy => "trained on five, tested on eleven."

    clear; clc; close all;

    nSpeakers= 23; 
    baseTrain= 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Five_Training\';
    baseTest = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Eleven_Test\';

    % 1) Build trainList => i=1..23 => each is a "five" training wave
    trainList= cell(nSpeakers,1);
    for i=1:nSpeakers
        filePath= fullfile(baseTrain, sprintf('s%d.wav', i));
        trainList{i}= filePath;
    end

    % Train codebooks
    numFilters=   20;
    numCoeffs=    12;
    codebookSize= 8;

    disp('--- Training speaker models on "five" => s1..s23 ---');
    speakerModels= train_speakers_demo(trainList, numFilters, numCoeffs, codebookSize);

    % 2) Build testList => each i=1..23 => from Eleven_Test
    testList= cell(nSpeakers,1);
    for i=1:nSpeakers
        fileTest= fullfile(baseTest, sprintf('s%d.wav', i));
        testList{i}= {fileTest, i};
    end

    % 3) Evaluate: system tries to identify speaker i from "eleven" test
    disp('--- Testing: (Trained on five) => now test on "eleven" data ---');
    [acc, ~] = test_speakers_demo(testList, speakerModels, numFilters, numCoeffs);

    fprintf('\nFinal speaker recognition accuracy (trained=five, tested=eleven): %.2f%%\n', acc);
    disp('Done test10b_fiveTrain_elevenTest.');
end

%% train_speakers_demo
function speakerModels = train_speakers_demo(trainList, numFilters, numCoeffs, codebookSize)
    nSpeak= numel(trainList);
    speakerModels= cell(nSpeak,1);
    for i=1:nSpeak
        fPath= trainList{i};
        if exist(fPath,'file')==2
            [mfccMat, ~]= computeMFCC_forFile(fPath, numFilters, numCoeffs);
            codebook= runVQcodebook(mfccMat, codebookSize);
            speakerModels{i}= codebook;
        else
            warning('Train file not found => skip: %s', fPath);
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
        item= testList{t};
        filePath= item{1};
        realID  = item{2};

        if exist(filePath,'file')==2
            [mfccTest, ~] = computeMFCC_forFile(filePath, numFilters, numCoeffs);
            [bestID, distVal]= findBestSpeaker_demo(mfccTest, speakerModels);
            predictions(t)= bestID;
            isOk= (bestID==realID);
            if isOk, correct= correct+1; end

            fprintf('Test: %s => spk#%d (true:%d), Dist=%.3f %s\n',...
                filePath, bestID, realID, distVal, ternary(isOk,'[OK]','[ERR]'));
        else
            warning('Test file missing => skip: %s', filePath);
        end
    end

    used= sum(predictions>0);
    if used>0
        accuracy= (correct/used)*100;
    else
        accuracy=0;
    end
end

%% computeMFCC_forFile
function [mfccMat, fsOut] = computeMFCC_forFile(wavPath, numFilters, numCoeffs)
    [y, fsOut]= audioread(wavPath);
    mfccMat   = computeMFCC_fromSignal(y, fsOut, numFilters, numCoeffs);
end

%% computeMFCC_fromSignal
function c= computeMFCC_fromSignal(signal, fs, numFilters, numCoeffs)
    alpha=0.95;
    for i=length(signal):-1:2
        signal(i)= signal(i)- alpha*signal(i-1);
    end

    N=256; overlap=100; NFFT=512;
    [S,~,~]= stft(signal, fs,'Window',hamming(N),'OverlapLength',overlap,...
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
        melE= melFB*pcol;
        melE(melE<1e-12)=1e-12;
        logMel= log(melE);
        dctCoeffs= dct(logMel);
        c(:,fIndex)= dctCoeffs(2:(numCoeffs+1));
    end

    c= c - (mean(c,2)+1e-8);
end

%% runVQcodebook => minimal LBG
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
                    tmp= mfccMat(:,idx)- newCB(:,i);
                    distortion= distortion+ sum(tmp.^2,'all');
                end
            end
            distortion= distortion/N;

            if abs(prevDist-distortion)/distortion<1e-3
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

%% findBestSpeaker_demo
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

%% computeVQDistortion_demo
function distVal= computeVQDistortion_demo(mfccMat, codebook)
    [~,N]= size(mfccMat);
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
function m= melfb(p, n, fs)
    f0= 700/fs;
    fn2= floor(n/2);
    lr= log(1+0.5/f0)/(p+1);

    bl= n*(f0*(exp([0,1,p,p+1]*lr)-1));
    b1= floor(bl(1))+1;
    b2= ceil(bl(2));
    b3= floor(bl(3));
    b4= min(fn2,ceil(bl(4)))-1;

    pf= log(1+ (b1:b4)/n/f0)/lr;
    fp= floor(pf);
    pm= pf- fp;

    r= [fp(b2:b4),1+fp(1:b3)];
    c= [b2:b4,1:b3]+1;
    v= 2*[1-pm(b2:b4),pm(1:b3)];
    m= sparse(r,c,v,p,1+fn2);
end

%% ternary => utility
function s= ternary(cond, sTrue, sFalse)
    if cond, s=sTrue; else, s=sFalse; end
end
