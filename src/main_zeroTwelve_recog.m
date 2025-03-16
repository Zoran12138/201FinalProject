function main_zeroTwelve_recog()
% main_zeroTwelve_recog
%
% Demonstration of a 2-class (zero vs. twelve) speaker recognition approach.
% We train from two folders:
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Training   => ID=1
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Training => ID=2
% Then test from two folders:
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Testing   => ID=1
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Testing => ID=2
%
% Each folder has files named e.g. Zero_train1.wav, Zero_train2.wav, ...
% or Twelve_test1.wav, etc. We'll parse them with simple rules below.
%
% We then measure the accuracy in terms of how well the system can
% distinguish zero vs. twelve sounds.
%
% Usage:
%   >> main_zeroTwelve_recog
%
% Author: GPT, with your previous MFCC+LBG logic

    clear; clc; close all;

    % 1) Folder paths (adjust if needed)
    folderZeroTrain   = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Training';
    folderTwelveTrain = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Training';
    folderZeroTest    = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Testing';
    folderTwelveTest  = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Testing';

    % 2) STFT + MFCC configuration
    N=256;             % frame length
    Mstep=100;         % frame step => overlap= N - Mstep
    NFFT=512;          % must be consistent w/ stft => freqRows= (NFFT/2 +1)
    numFilters=20;     % mel filters
    numCoeffs=12;      % keep c2..c13 => 12 MFCC
    codebookSize=8;    % LBG codebook

    % ========== Step A: TRAIN ==========

    % We'll combine two sets => ID=1 => Zero, ID=2 => Twelve
    % parse them => gather MFCC => LBG => get speakerModels{1}, speakerModels{2}
    fprintf('=== TRAIN: Zero->ID=1, Twelve->ID=2 ===\n');
    speakerModels= train_zeroTwelve(folderZeroTrain, folderTwelveTrain, ...
                                    N, Mstep, NFFT, numFilters, numCoeffs, codebookSize);

    % ========== Step B: TEST ==========
    fprintf('\n=== TEST: Zero_Testing (ID=1), Twelve_Testing (ID=2) ===\n');
    [acc, total, correct] = test_zeroTwelve(folderZeroTest, folderTwelveTest, ...
                                speakerModels, N, Mstep, NFFT, numFilters, numCoeffs);

    fprintf('\nFinal recognition accuracy = %.2f%%  (correct=%d / total=%d)\n', acc, correct, total);
end

%% =========================================================
function speakerModels= train_zeroTwelve(folderZeroTrain, folderTwelveTrain, ...
                                         N, Mstep, NFFT, numFilters, numCoeffs, codebookSize)
% TRAIN_ZeroTwelve:
%   read "Zero_trainX.wav" => spkID=1
%   read "Twelve_trainX.wav" => spkID=2
%   combine => do LBG => store in speakerModels{1} and speakerModels{2}

    % parse zero => ID=1
    waveList_zero= dir(fullfile(folderZeroTrain, 'Zero_train*.wav'));
    if isempty(waveList_zero)
        warning('No Zero_train*.wav found in %s => ID=1 will be empty', folderZeroTrain);
    end

    allMFCC_1= [];
    for i=1:length(waveList_zero)
        fname= waveList_zero(i).name;
        fpath= fullfile(waveList_zero(i).folder, fname);
        fprintf('Train[Zero->ID=1]: %s\n', fname);

        [y, fs]= audioread(fpath);
        if size(y,2)>1, y= y(:,1); end
        % preproc
        y= y- mean(y);
        pk= max(abs(y));
        if pk>1e-12
            y= y/pk;
        end

        mfccMat= audio2mfcc_zt(y, fs, N, Mstep, NFFT, numFilters, numCoeffs);
        allMFCC_1= [allMFCC_1, mfccMat];
    end

    % parse twelve => ID=2
    waveList_twelve= dir(fullfile(folderTwelveTrain, 'Twelve_train*.wav'));
    if isempty(waveList_twelve)
        warning('No Twelve_train*.wav found in %s => ID=2 will be empty', folderTwelveTrain);
    end

    allMFCC_2= [];
    for i=1:length(waveList_twelve)
        fname= waveList_twelve(i).name;
        fpath= fullfile(waveList_twelve(i).folder, fname);
        fprintf('Train[Twelve->ID=2]: %s\n', fname);

        [y, fs]= audioread(fpath);
        if size(y,2)>1, y= y(:,1); end
        y= y- mean(y);
        pk= max(abs(y));
        if pk>1e-12
            y= y/pk;
        end

        mfccMat= audio2mfcc_zt(y, fs, N, Mstep, NFFT, numFilters, numCoeffs);
        allMFCC_2= [allMFCC_2, mfccMat];
    end

    speakerModels= cell(2,1);

    if ~isempty(allMFCC_1)
        X1= allMFCC_1';  % => (#frames x #coeff)
        codebook1= runLBG_zt(X1, codebookSize);
        speakerModels{1}= codebook1;
    else
        speakerModels{1}= [];
    end

    if ~isempty(allMFCC_2)
        X2= allMFCC_2';
        codebook2= runLBG_zt(X2, codebookSize);
        speakerModels{2}= codebook2;
    else
        speakerModels{2}= [];
    end
end

%% =========================================================
function [acc, total, correct]= test_zeroTwelve(folderZeroTest, folderTwelveTest, ...
                                                speakerModels, N, Mstep, NFFT, numFilters, numCoeffs)
% test_zeroTwelve
%   parse Zero_testX => ID=1, parse Twelve_testX => ID=2
%   get MFCC => match => measure overall accuracy

    correct=0; total=0;

    % 1) zero => ID=1
    waveList_zero= dir(fullfile(folderZeroTest, 'Zero_test*.wav'));
    for i=1:length(waveList_zero)
        fname= waveList_zero(i).name;
        fpath= fullfile(waveList_zero(i).folder, fname);
        trueID= 1;
        [bestID, distVal, isOK]= testOneFile_zt(fpath, trueID, speakerModels, N,Mstep,NFFT,numFilters,numCoeffs);
        if isOK, correct=correct+1; end
        total= total+1;
        fprintf('[Zero->ID=1] %s => predicted=%d, dist=%.3f %s\n', ...
            fname, bestID, distVal, ternary_zt(isOK,'[OK]','[ERR]'));
    end

    % 2) twelve => ID=2
    waveList_twelve= dir(fullfile(folderTwelveTest, 'Twelve_test*.wav'));
    for i=1:length(waveList_twelve)
        fname= waveList_twelve(i).name;
        fpath= fullfile(waveList_twelve(i).folder, fname);
        trueID= 2;
        [bestID, distVal, isOK]= testOneFile_zt(fpath, trueID, speakerModels, N,Mstep,NFFT,numFilters,numCoeffs);
        if isOK, correct=correct+1; end
        total= total+1;
        fprintf('[Twelve->ID=2] %s => predicted=%d, dist=%.3f %s\n', ...
            fname, bestID, distVal, ternary_zt(isOK,'[OK]','[ERR]'));
    end

    if total>0
        acc= (correct/total)*100;
    else
        acc=0;
    end
    fprintf('Test done => correct=%d / total=%d => accuracy=%.2f%%\n', correct, total, acc);
end

%% =============== SUBFUNCTION: test one file ===============
function [bestID, distVal, isOK]= testOneFile_zt(filePath, trueID, speakerModels, ...
                                                N, Mstep, NFFT, numFilters, numCoeffs)
    [y, fs]= audioread(filePath);
    if size(y,2)>1
        y= y(:,1);
    end
    y= y- mean(y);
    pk= max(abs(y));
    if pk>1e-12
        y= y/pk;
    end

    mfccTest= audio2mfcc_zt(y, fs, N, Mstep, NFFT, numFilters, numCoeffs);
    [bestID, distVal]= findBestSpeaker_zt(mfccTest, speakerModels);
    isOK= (bestID==trueID);
end

%% =============== SUBFUNCTION: audio -> powerspec -> MFCC ===============
function mfccMat= audio2mfcc_zt(y, fs, N, Mstep, NFFT, numFilters, numCoeffs)
    ps= audio2powerspec_zt(y, fs, N, Mstep, NFFT);
    mfccMat= myMfcc_zt(ps, fs, numFilters, numCoeffs, NFFT);
end

function ps= audio2powerspec_zt(y, fs, N, Mstep, NFFT)
    alpha=0.95;
    for i= length(y):-1:2
        y(i)= y(i)- alpha*y(i-1);
    end
    overlap= N- Mstep;
    w= hamming(N);

    [S,~,~]= stft(y, fs,'Window',w,'OverlapLength',overlap,'FFTLength',NFFT, ...
                  'FrequencyRange','onesided');
    ps= (abs(S).^2)./NFFT;
end

function c= myMfcc_zt(ps, fs, numFilters, numCoeffs, NFFT)
    [freqRows, nFrames]= size(ps);
    halfN= (NFFT/2)+1;
    if freqRows~= halfN
        error('myMfcc_zt: freqRows=%d, expected %d => check stft config', freqRows, halfN);
    end

    melFB= melfb_zt(numFilters, NFFT, fs);
    if size(melFB,2)~= freqRows
        error('Mismatch => melFB(%dx%d) vs ps(%dx%d)', ...
            size(melFB,1), size(melFB,2), freqRows, nFrames);
    end

    c= zeros(numCoeffs, nFrames);
    for fIndex=1:nFrames
        col= ps(:,fIndex);
        melE= melFB*col;
        melE(melE<1e-12)=1e-12;
        logMel= log(melE);
        dctC= dct(logMel);
        c(:,fIndex)= dctC(2:(numCoeffs+1)); % c2..c13
    end
end

%% =============== MELFB ===============
function m= melfb_zt(p,n,fs)
    f0= 700/fs;
    fn2= floor(n/2);
    lr= log(1+0.5/f0)/(p+1);

    bl= n*(f0*(exp([0,1,p,p+1]*lr)-1));
    b1= floor(bl(1))+1;
    b2= ceil(bl(2));
    b3= floor(bl(3));
    b4= min(fn2,ceil(bl(4)))-1;

    pf= log(1+(b1:b4)/(n*f0))/lr;
    fp= floor(pf);
    pm= pf- fp;

    r= [fp(b2:b4), 1+fp(1:b3)];
    c= [b2:b4,     1:b3]+1;
    v= 2*[1-pm(b2:b4), pm(1:b3)];
    m= sparse(r,c,v,p, fn2+1);
end

%% =============== SUBFUNCTION: LBG ===============
function codebook= runLBG_zt(X, codebookSize)
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
            for ci=1:count
                diffVal= X - cbook(:,ci)'; % NxD
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
    codebook= cbook;  % => (D x codebookSize)
end

%% =============== SUBFUNCTION: Matching ===============
function [bestID, distVal]= findBestSpeaker_zt(mfccMat, speakerModels)
% mfccMat => (numCoeffs x #frames), speakerModels => cell(2,1)
%   speakerModels{1}= codebook of zero, {2}= codebook of twelve
    bestID=0; distVal= inf;

    for i=1:numel(speakerModels)
        cb= speakerModels{i};
        if isempty(cb), continue; end
        dVal= computeVQDist_zt(mfccMat, cb);
        if dVal< distVal
            distVal= dVal;
            bestID= i;
        end
    end
end

function val= computeVQDist_zt(mfccMat, codebook)
    [C, frames]= size(mfccMat);
    total=0;
    for f=1:frames
        vec= mfccMat(:,f); % (C x 1)
        diff= codebook - vec;  % (C x codebookSize) - (C x 1)
        dist2= sum(diff.^2,1);
        total= total+ min(dist2);
    end
    val= total/frames;
end

%% =============== Ternary ===============
function s= ternary_zt(cond, sTrue, sFalse)
    if cond
        s= sTrue;
    else
        s= sFalse;
    end
end
