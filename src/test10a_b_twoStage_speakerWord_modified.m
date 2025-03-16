function test10a_b_twoStage_speakerWord_modified()
% test10a_b_twoStage_speakerWord_modified
%
% This script demonstrates a two-stage approach to simultaneously identify:
%   (a) which word is spoken ("zero" or "twelve"), 
%   (b) which speaker ID (among 19 speakers).
%
% Main idea of the pipeline:
%   1) Stage A: Word-level classification. We gather all zero training data into 
%      one word class=1, and all twelve training data into word class=2.
%      We build 2 codebooks: codebook_word1 (zero) and codebook_word2 (twelve).
%      During testing, we compute Euclidean distance to each codebook, 
%      pick whichever is smaller => the predicted word.
%
%   2) Stage B: Speaker-level classification. Once the word is decided 
%      (i.e., zero or twelve), we choose the corresponding set of speaker codebooks. 
%      For example, if Stage A says "word=1 (zero)", we match against the 19 speaker 
%      codebooks dedicated to zero, pick the speaker with minimal distance. 
%      The final output is (wordPred, speakerPred).
%
%   3) We measure how many times both word and speaker predictions match 
%      the ground truth => final accuracy.
%
% The MFCC extraction includes an additional normalization step after DCT:
%   each MFCC matrix is scaled by its maximum absolute value => [-1,1] range.
% 
% We also build confusion matrices for:
%   - word classification (2 classes: zero, twelve),
%   - speaker classification (19 classes).
%
% The file organization is assumed as follows:
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\
%       Zero_Training\Zero_train1.wav ... Zero_train19.wav
%       Zero_Testing\Zero_test1.wav  ... Zero_test19.wav
%       Twelve_Training\Twelve_train1.wav ... Twelve_train19.wav
%       Twelve_Testing\Twelve_test1.wav  ... Twelve_test19.wav
%
% Usage:
%   >> test10a_b_twoStage_speakerWord_modified

    clear; clc; close all;

    % Number of speakers, each says "zero" and "twelve"
    nSpeakers = 19;

    % File paths
    baseZeroTrain    = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Training\';
    baseZeroTest     = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Zero_Testing\';
    baseTwelveTrain  = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Training\';
    baseTwelveTest   = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Twelve_Testing\';

    % Stage A: Train word-level codebooks (word=1 => zero, word=2 => twelve)
    disp('--- Stage A: Train word-level codebooks (zero=1, twelve=2) ---');
    zero_mfcc  = [];
    twelve_mfcc= [];

    for i=1:nSpeakers
        zPath = fullfile(baseZeroTrain,   sprintf('Zero_train%d.wav', i));
        tPath = fullfile(baseTwelveTrain, sprintf('Twelve_train%d.wav', i));

        if exist(zPath,'file')==2
            [zMat, ~] = computeMFCC_forFile(zPath, 20, 12, true);
            zero_mfcc  = [zero_mfcc, zMat];
        else
            warning('Missing Zero train file: %s', zPath);
        end
        if exist(tPath,'file')==2
            [tMat, ~] = computeMFCC_forFile(tPath, 20, 12, true);
            twelve_mfcc= [twelve_mfcc, tMat];
        else
            warning('Missing Twelve train file: %s', tPath);
        end
    end

    codebookSize_word = 8;
    codebook_word1 = runVQcodebook(zero_mfcc,   codebookSize_word); % word=1=zero
    codebook_word2 = runVQcodebook(twelve_mfcc, codebookSize_word); % word=2=twelve
    disp('Word-level training done.');

    % Stage B: Train speaker-level codebooks for "zero" and "twelve"
    disp('--- Stage B: Train speaker-level codebooks for zero & twelve ---');
    zero_speakerModels   = cell(nSpeakers,1);
    twelve_speakerModels = cell(nSpeakers,1);

    for i=1:nSpeakers
        zPath = fullfile(baseZeroTrain,   sprintf('Zero_train%d.wav', i));
        tPath = fullfile(baseTwelveTrain, sprintf('Twelve_train%d.wav', i));

        zero_speakerModels{i}   = [];
        twelve_speakerModels{i} = [];

        if exist(zPath,'file')==2
            [zMat, ~] = computeMFCC_forFile(zPath, 20, 12, true);
            zero_speakerModels{i} = runVQcodebook(zMat, 8);
        end
        if exist(tPath,'file')==2
            [tMat, ~] = computeMFCC_forFile(tPath, 20, 12, true);
            twelve_speakerModels{i} = runVQcodebook(tMat, 8);
        end
    end
    disp('Speaker-level training done.');

    % Testing: gather all "Zero_test i" and "Twelve_test i"
    allTrueWords    = [];
    allPredWords    = [];
    allTrueSpeakers = [];
    allPredSpeakers = [];

    correctCount= 0;
    totalCount  = 0;

    disp('--- Testing zero_test... (word=1) ---');
    for i=1:nSpeakers
        zPath= fullfile(baseZeroTest, sprintf('Zero_test%d.wav', i));
        if exist(zPath,'file')~=2
            warning('Missing zero test file: %s', zPath);
            continue;
        end
        wordGT= 1;  % zero
        spkGT= i;
        totalCount= totalCount +1;

        [mfccTest, ~] = computeMFCC_forFile(zPath, 20, 12, true);

        % Stage A => word classification
        dist1= computeVQDistortion_demo(mfccTest, codebook_word1);
        dist2= computeVQDistortion_demo(mfccTest, codebook_word2);
        if dist1 < dist2
            wordPred=1;
        else
            wordPred=2;
        end

        % Stage B => speaker classification
        if wordPred==1
            bestSpk=0; bestDist=inf;
            for s=1:nSpeakers
                cb= zero_speakerModels{s};
                if ~isempty(cb)
                    dVal= computeVQDistortion_demo(mfccTest, cb);
                    if dVal< bestDist
                        bestDist= dVal;
                        bestSpk= s;
                    end
                end
            end
            spkPred= bestSpk;
        else
            bestSpk=0; bestDist=inf;
            for s=1:nSpeakers
                cb= twelve_speakerModels{s};
                if ~isempty(cb)
                    dVal= computeVQDistortion_demo(mfccTest, cb);
                    if dVal< bestDist
                        bestDist= dVal;
                        bestSpk= s;
                    end
                end
            end
            spkPred= bestSpk;
        end

        allTrueWords(end+1)= wordGT;
        allPredWords(end+1)= wordPred;
        allTrueSpeakers(end+1)= spkGT;
        allPredSpeakers(end+1)= spkPred;

        isOK= (wordPred==wordGT && spkPred==spkGT);
        if isOK, correctCount= correctCount+1; end

        fprintf('TestZero:%s => Pred(word=%d, spk=%d) vs GT(word=1, spk=%d) %s\n',...
            zPath, wordPred, spkPred, spkGT, ternary(isOK,'[OK]','[ERR]'));
    end

    disp('--- Testing twelve_test... (word=2) ---');
    for i=1:nSpeakers
        tPath= fullfile(baseTwelveTest, sprintf('Twelve_test%d.wav', i));
        if exist(tPath,'file')~=2
            warning('Missing twelve test file: %s', tPath);
            continue;
        end
        wordGT= 2; 
        spkGT= i;
        totalCount= totalCount +1;

        [mfccTest, ~] = computeMFCC_forFile(tPath, 20, 12, true);

        dist1= computeVQDistortion_demo(mfccTest, codebook_word1);
        dist2= computeVQDistortion_demo(mfccTest, codebook_word2);
        if dist1<dist2
            wordPred=1;
        else
            wordPred=2;
        end

        if wordPred==1
            bestSpk=0; bestDist=inf;
            for s=1:nSpeakers
                cb= zero_speakerModels{s};
                if ~isempty(cb)
                    dVal= computeVQDistortion_demo(mfccTest, cb);
                    if dVal< bestDist
                        bestDist= dVal;
                        bestSpk= s;
                    end
                end
            end
            spkPred= bestSpk;
        else
            bestSpk=0; bestDist=inf;
            for s=1:nSpeakers
                cb= twelve_speakerModels{s};
                if ~isempty(cb)
                    dVal= computeVQDistortion_demo(mfccTest, cb);
                    if dVal< bestDist
                        bestDist= dVal;
                        bestSpk= s;
                    end
                end
            end
            spkPred= bestSpk;
        end

        allTrueWords(end+1)= wordGT;
        allPredWords(end+1)= wordPred;
        allTrueSpeakers(end+1)= spkGT;
        allPredSpeakers(end+1)= spkPred;

        isOK= (wordPred==wordGT && spkPred==spkGT);
        if isOK, correctCount= correctCount+1; end

        fprintf('TestTwelve:%s => Pred(word=%d, spk=%d) vs GT(word=2, spk=%d) %s\n',...
            tPath, wordPred, spkPred, spkGT, ternary(isOK,'[OK]','[ERR]'));
    end

    finalAccuracy= 0;
    if totalCount>0
        finalAccuracy= (correctCount / totalCount)*100;
    end
    fprintf('\nFinal pipeline accuracy (word+speaker) = %.2f%%\n', finalAccuracy);

    % Confusion matrix for word classification
    if ~isempty(allTrueWords)
        figure('Name','Word Classification Confusion','NumberTitle','off');
        cmWord = confusionmat(allTrueWords, allPredWords);
        confusionchart(cmWord, {'zero','twelve'}, ...
            'Title','Word-Level Confusion Matrix');
    end

    % Confusion matrix for speaker classification
    if ~isempty(allTrueSpeakers)
        figure('Name','Speaker Classification Confusion','NumberTitle','off');
        cmSpk = confusionmat(allTrueSpeakers, allPredSpeakers);
        confusionchart(cmSpk, ...
            'Title','Speaker-Level Confusion Matrix',...
            'RowSummary','row-normalized','ColumnSummary','column-normalized');
    end

    disp('Done test10a_b_twoStage_speakerWord_modified.');
end


%% ========================================================================
%% computeMFCC_forFile => includes doNormalize param
function [mfccMat, fsOut] = computeMFCC_forFile(wavPath, numFilters, numCoeffs, doNormalize)
% Reads WAV, computes MFCC, optionally normalizes the MFCC matrix amplitude.

    if nargin<4 || isempty(doNormalize)
        doNormalize= false;
    end
    [signal, fsOut] = audioread(wavPath);
    mfccMat = computeMFCC_fromSignal(signal, fsOut, numFilters, numCoeffs);

    if doNormalize
        mfccMat= normalizeMFCC_matrix(mfccMat);
    end
end

function mfccMat = computeMFCC_fromSignal(signal, fs, numFilters, numCoeffs)
% Basic method: 
%   1) pre-emphasis => y(t)= x(t)- 0.95*x(t-1)
%   2) stft => onesided => power => mel => log => dct
%   3) keep c2..c13 => #12 dims

    alpha=0.95;
    for i=length(signal):-1:2
        signal(i)= signal(i)- alpha*signal(i-1);
    end

    N=256; overlap=100; NFFT=512;
    [S,~,~]= stft(signal, fs,'Window',hamming(N),'OverlapLength',overlap,...
                  'FFTLength',NFFT,'FrequencyRange','onesided');
    ps= abs(S).^2 / NFFT;

    melFB= melfb(numFilters, NFFT, fs);
    if size(melFB,2)~= size(ps,1)
        error('Dimension mismatch => melFB=%dx%d, ps=%dx%d',...
            size(melFB,1), size(melFB,2), size(ps,1), size(ps,2));
    end

    nFrames= size(ps,2);
    mfccMat= zeros(numCoeffs,nFrames);
    for f=1:nFrames
        pcol= ps(:,f);
        pcol(pcol<1e-12)=1e-12;
        melE= melFB* pcol;
        melE(melE<1e-12)=1e-12;
        logMel= log(melE);
        dctCoeffs= dct(logMel);
        mfccMat(:,f)= dctCoeffs(2:(numCoeffs+1));
    end
end

function M = normalizeMFCC_matrix(M)
% normalizes MFCC by its max absolute value
    maxVal= max(abs(M(:)));
    if maxVal>1e-12
        M= M / maxVal;
    end
end

%% ========================================================================
%% LBG-based VQ

function codebook= runVQcodebook(mfccMat, codebookSize)
% codebook => #coeff x codebookSize

    epsVal= 0.01;
    distThresh= 1e-3;
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
                diffVal= mfccMat - cbook(:,i);
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
                    diffV= mfccMat(:,idx)- newCB(:,i);
                    distortion= distortion + sum(diffV.^2,'all');
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

function distVal= computeVQDistortion_demo(mfccMat, codebook)
% sum of min-dist^2 for each frame
    if isempty(codebook)
        distVal= inf; return;
    end
    [~,N]= size(mfccMat);
    total=0;
    for n=1:N
        vec= mfccMat(:,n);
        diff= codebook - vec;
        dists= sum(diff.^2,1);
        total= total+ min(dists);
    end
    distVal= total / N;
end

%% ========================================================================
%% mel filter bank
function m = melfb(p, n, fs)
    f0= 700/fs;
    fn2= floor(n/2);
    lr= log(1 + 0.5/f0)/(p+1);

    bl= n*(f0*(exp([0,1,p,p+1]*lr)-1));
    b1= floor(bl(1))+1;
    b2= ceil(bl(2));
    b3= floor(bl(3));
    b4= min(fn2,ceil(bl(4)))-1;

    pf= log(1 + (b1:b4)/(n*f0))/lr;
    fp= floor(pf);
    pm= pf- fp;

    r= [fp(b2:b4),1+fp(1:b3)];
    c= [b2:b4,1:b3]+1;
    v= 2*[1-pm(b2:b4), pm(1:b3)];
    m= sparse(r,c,v,p,fn2+1);
end

function s= ternary(cond, sTrue, sFalse)
    if cond
        s= sTrue;
    else
        s= sFalse;
    end
end
