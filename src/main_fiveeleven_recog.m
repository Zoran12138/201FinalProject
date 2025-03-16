function main_fiveEleven_recog()
% main_fiveEleven_recog
%
% Demonstrates a 2-class recognition system for "five" (ID=1) vs. "eleven" (ID=2).
% Each folder contains audio files named s1.wav, s2.wav, ..., s23.wav, etc.
%
% Folder structure (all .wav files named sX.wav):
%   D:\Program Files\Polyspace\R2021a\bin\EEC201
%       Five_Training\   (s1.wav, s2.wav, ...)
%       Five_Test\       (s1.wav, s2.wav, ...)
%       Eleven_Training\ (s1.wav, s2.wav, ...)
%       Eleven_Test\     (s1.wav, s2.wav, ...)
%
% We parse the folder to decide the word ID:
%   ID=1 => five, ID=2 => eleven
% Then measure accuracy in test stage by comparing predicted ID vs. ground‐truth ID.
%
% Usage:
%   >> main_fiveEleven_recog

    clear; clc; close all;

    % 1) Folder paths
    folderFiveTrain   = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Five_Training';
    folderFiveTest    = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Five_Test';
    folderElevenTrain = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Eleven_Training';
    folderElevenTest  = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Eleven_Test';

    % 2) STFT + MFCC configuration
    N=256;      % frame size
    Mstep=100;  % frame step => overlap= N - Mstep
    NFFT=512;   
    numFilters=20;
    numCoeffs=12;   % keep c2..c13 => 12 dims
    codebookSize=8; % LBG codebook

    % ========== STEP A: TRAIN ==========
    % ID=1 => five, ID=2 => eleven
    fprintf('=== TRAIN: five->ID=1, eleven->ID=2 ===\n');
    speakerModels = train_fiveEleven(folderFiveTrain, folderElevenTrain, ...
                                     N, Mstep, NFFT, numFilters, numCoeffs, codebookSize);

    % ========== STEP B: TEST ==========
    fprintf('\n=== TEST: Five_Test => ID=1, Eleven_Test => ID=2 ===\n');
    [acc, total, correct] = test_fiveEleven(folderFiveTest, folderElevenTest, ...
                                speakerModels, N, Mstep, NFFT, numFilters, numCoeffs);

    fprintf('\nFinal recognition accuracy = %.2f%%  (correct=%d / total=%d)\n', acc, correct, total);
end

%% =========================================================
function speakerModels= train_fiveEleven(folderFiveTrain, folderElevenTrain, ...
                                         N, Mstep, NFFT, numFilters, numCoeffs, codebookSize)
% train_fiveEleven:
%   For word=1 => gather from folderFiveTrain\*.wav
%   For word=2 => gather from folderElevenTrain\*.wav
%   Combine MFCC => run LBG => store in speakerModels{1} (five) & {2} (eleven)

    % gather "five" => ID=1
    fList_five= dir(fullfile(folderFiveTrain, 's*.wav'));
    if isempty(fList_five)
        warning('No s*.wav in Five_Training => ID=1 will be empty');
    end

    allMFCC_1= [];
    for i=1:length(fList_five)
        fname= fList_five(i).name;
        fpath= fullfile(fList_five(i).folder, fname);
        fprintf('Train[Five->ID=1]: %s\n', fname);

        [y, fs]= audioread(fpath);
        if size(y,2)>1, y= y(:,1); end
        y= y - mean(y);
        peakVal= max(abs(y));
        if peakVal>1e-12
            y= y / peakVal;
        end

        mfccMat= audio2mfcc_fe(y, fs, N, Mstep, NFFT, numFilters, numCoeffs);
        allMFCC_1= [allMFCC_1, mfccMat];
    end

    % gather "eleven" => ID=2
    fList_eleven= dir(fullfile(folderElevenTrain, 's*.wav'));
    if isempty(fList_eleven)
        warning('No s*.wav in Eleven_Training => ID=2 will be empty');
    end

    allMFCC_2= [];
    for i=1:length(fList_eleven)
        fname= fList_eleven(i).name;
        fpath= fullfile(fList_eleven(i).folder, fname);
        fprintf('Train[Eleven->ID=2]: %s\n', fname);

        [y, fs]= audioread(fpath);
        if size(y,2)>1, y= y(:,1); end
        y= y - mean(y);
        peakVal= max(abs(y));
        if peakVal>1e-12
            y= y / peakVal;
        end

        mfccMat= audio2mfcc_fe(y, fs, N, Mstep, NFFT, numFilters, numCoeffs);
        allMFCC_2= [allMFCC_2, mfccMat];
    end

    speakerModels= cell(2,1);

    % building codebook for word=1 => five
    if ~isempty(allMFCC_1)
        X1= allMFCC_1';
        cb1= runLBG_fe(X1, codebookSize);
        speakerModels{1}= cb1;
    else
        speakerModels{1}= [];
    end

    % building codebook for word=2 => eleven
    if ~isempty(allMFCC_2)
        X2= allMFCC_2';
        cb2= runLBG_fe(X2, codebookSize);
        speakerModels{2}= cb2;
    else
        speakerModels{2}= [];
    end
end

%% =========================================================
function [acc, total, correct] = test_fiveEleven(folderFiveTest, folderElevenTest, ...
                                                 speakerModels, N, Mstep, NFFT, numFilters, numCoeffs)
% test_fiveEleven:
%   parse all s*.wav in Five_Test => ID=1
%   parse all s*.wav in Eleven_Test => ID=2
%   measure how many times predicted ID == groundTruth ID

    correct=0; total=0;

    % ID=1 => five
    fList_five= dir(fullfile(folderFiveTest, 's*.wav'));
    for i=1:length(fList_five)
        fname= fList_five(i).name;
        fpath= fullfile(fList_five(i).folder, fname);
        trueID= 1;

        [bestID, distVal, isOK]= testOneFile_fe(fpath, trueID, speakerModels, ...
            N, Mstep, NFFT, numFilters, numCoeffs);

        if isOK, correct= correct+1; end
        total= total+1;
        fprintf('[Five->ID=1] %s => predicted=%d, dist=%.3f %s\n',...
            fname, bestID, distVal, ternary_fe(isOK,'[OK]','[ERR]'));
    end

    % ID=2 => eleven
    fList_eleven= dir(fullfile(folderElevenTest, 's*.wav'));
    for i=1:length(fList_eleven)
        fname= fList_eleven(i).name;
        fpath= fullfile(fList_eleven(i).folder, fname);
        trueID= 2;

        [bestID, distVal, isOK]= testOneFile_fe(fpath, trueID, speakerModels, ...
            N, Mstep, NFFT, numFilters, numCoeffs);

        if isOK, correct= correct+1; end
        total= total+1;
        fprintf('[Eleven->ID=2] %s => predicted=%d, dist=%.3f %s\n',...
            fname, bestID, distVal, ternary_fe(isOK,'[OK]','[ERR]'));
    end

    if total>0
        acc= (correct / total)*100;
    else
        acc=0;
    end
    fprintf('Test done => correct=%d / total=%d => accuracy=%.2f%%\n', correct, total, acc);
end

%% =============== testOneFile_fe ===============
function [bestID, distVal, isOK] = testOneFile_fe(filePath, trueID, speakerModels, ...
                                                 N, Mstep, NFFT, numFilters, numCoeffs)
% read file => MFCC => find best among speakerModels{1} or {2} => bestID => compare

    [sig, fs]= audioread(filePath);
    if size(sig,2)>1
        sig= sig(:,1);
    end
    sig= sig- mean(sig);
    pk= max(abs(sig));
    if pk>1e-12
        sig= sig/ pk;
    end

    mfccTest= audio2mfcc_fe(sig, fs, N, Mstep, NFFT, numFilters, numCoeffs);
    [bestID, distVal]= findBestClass_fe(mfccTest, speakerModels);
    isOK= (bestID == trueID);
end

%% =============== audio->powerspec->MFCC ===============
function mfccMat= audio2mfcc_fe(y, fs, N, Mstep, NFFT, numFilters, numCoeffs)
    ps= audio2powerspec_fe(y, fs, N, Mstep, NFFT);
    mfccMat= myMfcc_fe(ps, fs, numFilters, numCoeffs, NFFT);
end

function ps= audio2powerspec_fe(y, fs, N, Mstep, NFFT)
    alpha=0.95;
    for i= length(y):-1:2
        y(i)= y(i)- alpha*y(i-1);
    end
    overlap= N- Mstep;
    w= hamming(N);

    [S,~,~]= stft(y, fs, 'Window',w, 'OverlapLength',overlap, ...
                  'FFTLength',NFFT, 'FrequencyRange','onesided');
    ps= (abs(S).^2)/NFFT;
end

function c= myMfcc_fe(ps, fs, numFilters, numCoeffs, NFFT)
    [freqRows, nFrames]= size(ps);
    halfN= (NFFT/2)+1;
    if freqRows~= halfN
        error('myMfcc_fe => freqRows=%d, expected=%d', freqRows, halfN);
    end

    melFB= melfb_fe(numFilters, NFFT, fs);
    if size(melFB,2)~= freqRows
        error('Mismatch => melFB(%dx%d), ps(%dx%d)', ...
            size(melFB,1), size(melFB,2), freqRows, nFrames);
    end

    c= zeros(numCoeffs, nFrames);
    for f=1:nFrames
        col= ps(:,f);
        col(col<1e-12)= 1e-12;
        melE= melFB* col;
        melE(melE<1e-12)=1e-12;
        logMel= log(melE);
        dctVal= dct(logMel);
        c(:,f)= dctVal(2:(numCoeffs+1));
    end
end

function m= melfb_fe(p, n, fs)
    f0=700/fs;
    fn2= floor(n/2);
    lr= log(1+ 0.5/f0)/(p+1);

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

%% =============== runLBG_fe ===============
function codebook= runLBG_fe(X, codebookSize)
    epsVal=0.01; distThresh=1e-3;
    [N,D]= size(X);

    cbook= mean(X,1)';
    count=1;
    while count< codebookSize
        cbook= [cbook.*(1+epsVal), cbook.*(1-epsVal)];
        count= size(cbook,2);

        prevDist=inf;
        while true
            distMat= zeros(count,N);
            for ci=1:count
                diffVal= X- cbook(:,ci)';
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
                    dv= X(idx,:)- newCB(:,ci)';
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

%% =============== findBestClass_fe ===============
function [bestID, distVal]= findBestClass_fe(mfccMat, speakerModels)
    bestID=0; distVal= inf;
    for i=1:numel(speakerModels)
        cb= speakerModels{i};
        if isempty(cb), continue; end
        dVal= computeVQDist_fe(mfccMat, cb);
        if dVal< distVal
            distVal= dVal;
            bestID= i;
        end
    end
end

function val= computeVQDist_fe(mfccMat, codebook)
    [~, frames]= size(mfccMat);
    total=0;
    for f=1:frames
        vec= mfccMat(:,f);
        diff= codebook- vec;
        dist2= sum(diff.^2,1);
        total= total+ min(dist2);
    end
    val= total/ frames;
end

%% =============== Ternary ===============
function s= ternary_fe(cond, sTrue, sFalse)
    if cond
        s= sTrue;
    else
        s= sFalse;
    end
end


