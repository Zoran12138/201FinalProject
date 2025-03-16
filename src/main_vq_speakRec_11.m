function main_vq_speakRec_11()
% main_vq_speakRec_11
%
% We do a "self-test" on the same Train folder => s1..s11.
% Step:
%   1) build trainList => {1;2;...;11} => train_speakers_vq => speakerModels
%   2) build testList  => the same => test_speakers_vq => check accuracy
%

    clear; clc; close all;

    codebookSize= 8;

    trainList= cell(11,1);
    for i=1:11
        trainList{i}= i;  % store ID
    end

    % testList => same
    testList= trainList;  % just reuse

    % Train
    speakerModels = train_speakers_vq(trainList, codebookSize);

    % Test => same folder => measure self-test accuracy
    [acc, preds] = test_speakers_vq(testList, speakerModels);

    fprintf('\nSelf-test accuracy (1..11 on same Train data) = %.2f%%\n', acc);
end
