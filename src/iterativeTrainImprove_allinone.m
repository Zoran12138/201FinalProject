function iterativeTrainImprove_allinone()
% ITERATIVETRAINIMPROVE_ALLINONE
%
% A single-file demo for repeatedly training speaker recognition with 
% different hyper-parameters, to (hopefully) improve matching accuracy.
%
% This script includes local functions:
%   1) train_speakers_robust
%   2) test_speakers_robust
%   3) audio2mfcc
%   4) trainVQ_LBG
%   5) findBestSpeaker
%   6) etc.


    clear; clc;

    %% 1) Define data directories
    trainDir = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\';
    testDir  = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\';

    %% 2) STFT (frame) config
    N = 256;       % frame length
    Mstep = 100;   % frame shift => overlap = N - Mstep
    NFFT = 512;    % must remain consistent => stft freqRows = NFFT/2+1 in onesided mode

    %% 3) We'll sweep over different param sets, e.g. codebook sizes or MFCC filters
    % For example:
    codebookSizes = [4, 8, 12];        % number of codewords in VQ
    numFiltersList= [20, 26, 32];      % number of mel filters
    numCoeffsList = [8, 12, 16];       % number of MFCC used (excluding c0)
    paramCount = 0;

    bestAcc  = 0;
    bestCode = 8;
    bestFilt = 20;
    bestCoef = 12;

    logData = {};  % for storing each experiment result

    %% 4) Nested loops to test all combos
    for cbSize = codebookSizes
        for nfilt = numFiltersList
            for ncoef = numCoeffsList
                paramCount = paramCount+1;
                fprintf('=== ParamSet #%d => codebook=%d, nfilt=%d, ncoef=%d ===\n',...
                    paramCount, cbSize, nfilt, ncoef);

                % (A) Train
                speakerModels = train_speakers_robust(trainDir, nfilt, ncoef, cbSize, N, Mstep, NFFT);

                % (B) Test
                [accTest, ~] = test_speakers_robust(testDir, speakerModels, nfilt, ncoef, N, Mstep, NFFT);

                fprintf(' => Test Accuracy=%.2f%%\n\n', accTest);

                % record
                logData{paramCount,1} = cbSize;
                logData{paramCount,2} = nfilt;
                logData{paramCount,3} = ncoef;
                logData{paramCount,4} = accTest;

                % check best
                if accTest> bestAcc
                    bestAcc  = accTest;
                    bestCode = cbSize;
                    bestFilt = nfilt;
                    bestCoef = ncoef;
                end
            end
        end
    end

    %% 5) Summarize results
    fprintf('===== Summary of Parameter Sweep =====\n');
    for i=1:paramCount
        fprintf('ParamSet #%d: (codebk=%d, nfilt=%d, ncoef=%d) => acc=%.2f%%\n',...
            i, logData{i,1}, logData{i,2}, logData{i,3}, logData{i,4});
    end
    fprintf('\n=> Best param => codebook=%d, melFilt=%d, mfccCoef=%d, acc=%.2f%%\n',...
            bestCode, bestFilt, bestCoef, bestAcc);

    %% 6) Optional final retrain with best param
    fprintf('\nRetraining final system with the best param...\n');
    speakerModels_best = train_speakers_robust(trainDir, bestFilt, bestCoef, bestCode, N, Mstep, NFFT);
    [accBest, ~] = test_speakers_robust(testDir, speakerModels_best, bestFilt, bestCoef, N, Mstep, NFFT);
    fprintf('Final system test accuracy = %.2f%%\n', accBest);

    fprintf('\nDone iterativeTrainImprove_allinone.\n');
end  % End main function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% train_speakers_robust
function speakerModels = train_speakers_robust(folderTrain, numFilters, numCoeffs, codebookSize, N, Mstep, NFFT)
% TRIAN_SPEAKERS_ROBUST
%  read all .wav in folderTrain, create a codebook for each file => store in cell array
%  (One-file-per-speaker logic: each .wav = one speaker)
%
    fList = dir(fullfile(folderTrain,'*.wav'));
    nFiles= length(fList);
    if nFiles<1
        error('No .wav found in %s => cannot train!', folderTrain);
    end

    speakerModels = cell(nFiles,1);

    for i=1:nFiles
        wavPath= fullfile(fList(i).folder, fList(i).name);
        fprintf('Train: file #%d => %s\n', i, fList(i).name);

        mfccMat = audio2mfcc(wavPath, numFilters, numCoeffs, N, Mstep, NFFT);
        codebook= trainVQ_LBG(mfccMat, codebookSize);
        speakerModels{i}= codebook;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% test_speakers_robust
function [accuracy, predictions] = test_speakers_robust(folderTest, speakerModels, numFilters, numCoeffs, N, Mstep, NFFT)
% TEST_SPEAKERS_ROBUST
%  read all .wav in folderTest, do the same feature => codebook distance => pick min => compare ID
%
    fList= dir(fullfile(folderTest,'*.wav'));
    nTest= length(fList);
    predictions= zeros(nTest,1);
    correct=0;

    if nTest<1
        warning('No .wav in %s => test=0%%', folderTest);
        accuracy=0; return;
    end

    for i=1:nTest
        wPath= fullfile(fList(i).folder, fList(i).name);
        [~, fname, ext] = fileparts(wPath);
        if ~strcmpi(ext, '.wav')
            warning('Skipping non-wav file: %s', wPath);
            continue;
        end

        mfccMat = audio2mfcc(wPath, numFilters, numCoeffs, N, Mstep, NFFT);

        % find best codebook
        [bestID, distVal] = findBestSpeaker(mfccMat, speakerModels);
        predictions(i)= bestID;

        % parse real ID from 'sX.wav'
        realID = parseTrueID(fname);  % if 's3' => 3
        isOk   = (bestID==realID && realID>0);
        if isOk, correct= correct+1; end

        fprintf('Test: %s => spk#%d (true:%d) Dist=%.3f %s\n',...
            [fname,ext], bestID, realID, distVal, ternary(isOk,'[OK]','[ERR]'));
    end

    used= sum(predictions>0);
    if used>0
        accuracy= (correct/used)*100;
    else
        accuracy=0;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parseTrueID => from 's3' => 3
function ID = parseTrueID(fname)
    pat= regexp(fname, '^s(\d+)$','tokens','once');
    if isempty(pat)
        ID=0;
    else
        ID= str2double(pat{1});
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% audio2mfcc => read wav => stft => power => mel => dct => final
function mfccMat = audio2mfcc(wavPath, numFilters, numCoeffs, N, Mstep, NFFT)
    [y, fs] = audioread(wavPath);

    % preEmphasis
    alpha= 0.95;
    for i=length(y):-1:2
        y(i)= y(i)- alpha*y(i-1);
    end

    % stft => 'onesided'
    overlap= N - Mstep;
    w= hamming(N);
    [S,F,T] = stft(y, fs, 'Window', w, 'OverlapLength', overlap,...
                   'FFTLength', NFFT, 'FrequencyRange','onesided');
    ps= (abs(S).^2)./NFFT;
    % ps => [freqRows x nFrames], freqRows= NFFT/2+1

    % melFB
    melFB= melfb(numFilters, NFFT, fs);
    if size(melFB,2)~= size(ps,1)
        error('Dimension mismatch in audio2mfcc => melFB=%dx%d, ps=%dx%d',...
            size(melFB,1), size(melFB,2), size(ps,1), size(ps,2));
    end

    nFrames= size(ps,2);
    mfccMat= zeros(numCoeffs, nFrames);
    for fIndex=1:nFrames
        pcol= ps(:,fIndex);
        mE= melFB * pcol;
        mE(mE<1e-12)= 1e-12;
        logMel= log(mE);
        dctCoeffs= dct(logMel);
        mfccMat(:,fIndex)= dctCoeffs(2:(numCoeffs+1));
    end

    % optional => mean norm
    mfccMat= mfccMat - (mean(mfccMat,2)+1e-8);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% trainVQ_LBG => create codebook by LBG approach
function codebook= trainVQ_LBG(mfccMat, codebookSize)
    epsVal= 0.01;
    distThresh= 1e-3;
    [dim, N]= size(mfccMat);

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

            newCB= zeros(dim,count);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% findBestSpeaker => compare MFCC with each speaker's codebook
function [bestID, distVal]= findBestSpeaker(mfccMat, speakerModels)
    bestID=0; distVal=inf;
    for sp=1:numel(speakerModels)
        cb= speakerModels{sp};
        if isempty(cb), continue; end
        dValue= computeVQDistortion(mfccMat, cb);
        if dValue< distVal
            distVal= dValue;
            bestID= sp;
        end
    end
end

function distVal= computeVQDistortion(mfccMat, codebook)
    [dim, N]= size(mfccMat);
    total=0;
    for n=1:N
        vec= mfccMat(:,n);
        diff= codebook- vec;
        dists= sum(diff.^2,1);
        total= total+ min(dists);
    end
    distVal= total/N;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ternary helper
function s= ternary(cond, sTrue, sFalse)
    if cond, s=sTrue; else, s=sFalse; end
end
