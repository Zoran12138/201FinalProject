
function speakerModels = train_speakers_vq(trainList, codebookSize)
% train_speakers_vq
%
% Input:
%   trainList   : cell array of speaker IDs, e.g. {1;2;3;...;11}
%   codebookSize: # of centroids for LBG
%
% Output:
%   speakerModels: cell array => speakerModels{id} = codebook for that speaker ID
%
% Steps:
%   For each speaker ID in trainList => 
%       use getFile(id,"train") => compute MFCC => run LBG => store codebook.

    % Gather unique speaker IDs
    spkIDs = unique([trainList{:}]);  % e.g. 1..11
    speakerModels = cell(max(spkIDs),1);

    for i=1:length(spkIDs)
        spkID = spkIDs(i);
        % read from train
        [y, fs, ~] = getFile(spkID, "train");

        % preproc: remove DC, normalize
        y = y - mean(y);
        p = max(abs(y));
        if p>1e-12
            y = y/p;
        end

        % MFCC => (#dim x #frames)
        mfccMat = computeMFCC_all(y, fs);
        X = mfccMat';  % => (#frames x #dim)

        % LBG => codebook
        codebook = runLBG(X, codebookSize);
        speakerModels{spkID} = codebook;

        fprintf('Trained speaker ID=%d => codebookSize=%d\n', spkID, codebookSize);
    end
end
