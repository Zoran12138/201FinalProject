function test7_run_modified()
% test7_run_modified
%
% Demo for Test 7 with:
%  - 11 training files => speakerModels of size 11
%  - 8 testing files
%  - Human score = 60%
%
% Requirements:
% 1) Put your train WAV files in D:\Program Files\Polyspace\R2021a\bin\EEC201\train\
% 2) Put your test WAV files  in D:\Program Files\Polyspace\R2021a\bin\EEC201\test\
% 3) File naming example: s1.wav, s2.wav, etc. Adjust if needed.
% 4) Adjust 'trainList' & 'testList' to match actual speaker IDs.

    clear; clc;

    %% ========== 1. Prepare train files for 11 speakers ==========
    % Suppose you have s1.wav ~ s11.wav in the 'train' folder.
    trainDir = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\';
    trainList = {
        fullfile(trainDir, 's1.wav');
        fullfile(trainDir, 's2.wav');
        fullfile(trainDir, 's3.wav');
        fullfile(trainDir, 's4.wav');
        fullfile(trainDir, 's5.wav');
        fullfile(trainDir, 's6.wav');
        fullfile(trainDir, 's7.wav');
        fullfile(trainDir, 's8.wav');
        fullfile(trainDir, 's9.wav');
        fullfile(trainDir, 's10.wav');
        fullfile(trainDir, 's11.wav');
    };

    % MFCC + LBG parameters
    numFilters = 20;   % e.g. 20 mel filters
    numCoeffs  = 12;   % e.g. 12 MFCC coeffs (excluding 0th)
    M = 8;             % LBG codebook size (#centroids)

    % Train codebooks
    speakerModels = train_speakers(trainList, numFilters, numCoeffs, M);

    fprintf('--- Finished training on 11 files. Codebook array size = %d ---\n\n', length(speakerModels));

    %% ========== 2. Prepare test files (8 total) with true speaker IDs ==========
    % Each row => { testFilePath, trueSpeakerID } 
    % Example: if 's1.wav' in test folder belongs to speaker #1, 's2.wav' => #2, etc.
    testDir = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\';
    testList = {
        { fullfile(testDir,'s1.wav'), 1 };
        { fullfile(testDir,'s2.wav'), 2 };
        { fullfile(testDir,'s3.wav'), 3 };
        { fullfile(testDir,'s4.wav'), 4 };
        { fullfile(testDir,'s5.wav'), 5 };
        { fullfile(testDir,'s8.wav'), 8 };
        { fullfile(testDir,'s9.wav'), 9 };
        { fullfile(testDir,'s11.wav'), 11 };
    };

    % Test & get system accuracy
    [accuracy, predictions] = test_speakers(testList, speakerModels, numFilters, numCoeffs);

    fprintf('--- System recognition rate on 8 test audios = %.2f%% ---\n\n', accuracy);

    %% ========== 3. Compare with human accuracy = 60% ==========
    humanScore = 60;  % e.g. human recognized 5 out of 8 => 62.5% ~ 60
    fprintf('System accuracy = %.2f%%, Human accuracy = %.2f%%\n', accuracy, humanScore);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTION: train_speakers
function speakerModels = train_speakers(trainList, numFilters, numCoeffs, M)
% train_speakers
%   Reads each speaker's train audio, extracts MFCC, trains LBG codebook.
%
%   trainList : cell array of train filenames
%   numFilters, numCoeffs : MFCC parameters
%   M : codebook size (for LBG)
%
%   speakerModels : cell array of codebooks => speakerModels{i} = [numCoeffs x M]

    nSpeakers = length(trainList);
    speakerModels = cell(nSpeakers, 1);

    for i = 1:nSpeakers
        filename = trainList{i};
        mfccMat = mfcc(filename, numFilters, numCoeffs);  % your existing mfcc.m
        codebook = trainVQ_LBG(mfccMat, M);               % see below
        speakerModels{i} = codebook;
        fprintf('Trained speaker #%d with file: %s\n', i, filename);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTION: test_speakers
function [accuracy, predictions] = test_speakers(testList, speakerModels, numFilters, numCoeffs)
% test_speakers
%   For each test audio, compute MFCC and compare with each speaker's codebook
%   picking minimal VQ distortion => recognized speaker ID
%
%   testList : cell array of { <testFilePath>, <trueSpeakerID> }
%   speakerModels : cell array of codebooks
%   numFilters, numCoeffs : same MFCC params as training
%
%   accuracy    : recognition rate in percent
%   predictions : Nx1 array of predicted IDs (1..nSpeakers)

    nTests = length(testList);
    predictions = zeros(nTests, 1);
    correct = 0;

    for t = 1:nTests
        info = testList{t};
        testFile = info{1};
        trueID   = info{2};

        mfccTest = mfcc(testFile, numFilters, numCoeffs);  
        bestSpk = 0;
        minDist = inf;

        for spk = 1:length(speakerModels)
            codebook = speakerModels{spk};
            distVal = computeVQDistortion(mfccTest, codebook);
            if distVal < minDist
                minDist = distVal;
                bestSpk = spk;
            end
        end

        predictions(t) = bestSpk;
        if bestSpk == trueID
            correct = correct + 1;
        end

        fprintf('Test file: %s => recognized as speaker %d (true: %d)\n', testFile, bestSpk, trueID);
    end

    accuracy = (correct / nTests) * 100;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTION: computeVQDistortion
function distVal = computeVQDistortion(mfccMat, codebook)
% computeVQDistortion 
%   Measures average Euclidean distance from each frame to the nearest codeword

    [~, N] = size(mfccMat);
    totalDist = 0;

    for n = 1:N
        frameVec = mfccMat(:,n);
        diffVals = codebook - frameVec;  
        dists = sum(diffVals.^2, 1);     
        totalDist = totalDist + min(dists);
    end

    distVal = totalDist / N;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTION: trainVQ_LBG
function codebook = trainVQ_LBG(data, M)
% trainVQ_LBG
%   A simple LBG approach => from single centroid => M codewords

    epsilon = 0.01;
    distThresh = 1e-3;
    [D, N] = size(data);

    cbook = mean(data, 2);
    count = 1;

    while count < M
        cbook = [cbook.*(1+epsilon), cbook.*(1-epsilon)];
        count = size(cbook, 2);

        prevDist = Inf;
        while true
            % Assign
            distMat = zeros(count, N);
            for i = 1:count
                diffVal = data - cbook(:, i);
                distMat(i,:) = sum(diffVal.^2, 1);
            end
            [~, nearest] = min(distMat, [], 1);

            % Update
            newCB = zeros(D, count);
            for i = 1:count
                idx = (nearest == i);
                if any(idx)
                    newCB(:, i) = mean(data(:, idx), 2);
                else
                    newCB(:, i) = cbook(:, i);
                end
            end

            % Distortion
            distortion = 0;
            for i = 1:count
                idx = (nearest == i);
                if any(idx)
                    diffVal = data(:, idx) - newCB(:, i);
                    distortion = distortion + sum(diffVal.^2,'all');
                end
            end
            distortion = distortion / N;

            if abs(prevDist - distortion)/distortion < distThresh
                cbook = newCB;
                break;
            else
                cbook = newCB;
                prevDist = distortion;
            end
        end
    end

    codebook = cbook;
end
