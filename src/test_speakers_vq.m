function [accuracy, predictions] = test_speakers_vq(testList, speakerModels)
% test_speakers_vq
%
% Input:
%   testList     : cell array of speaker IDs, e.g. {1;2;3;...;11}
%   speakerModels: cell array => speakerModels{id} is codebook
%
% Output:
%   accuracy   : recognition rate (%)
%   predictions: array of length(#testList) w/ predicted speaker ID
%
% Steps:
%   For each ID in testList => getFile(id,"train") => MFCC => pick codebook => check correctness

    nTests= length(testList);
    predictions= zeros(nTests,1);
    correctCount=0;

    for t=1:nTests
        spkID_true= testList{t};  % an integer

        % read from train (we do self-test)
        [y, fs, ~] = getFile(spkID_true, "train");

        y= y - mean(y);
        pk= max(abs(y));
        if pk>1e-12
            y= y/pk;
        end

        mfccTest= computeMFCC_all(y, fs);
        Xtest   = mfccTest';

        bestID= 0;
        bestDist= inf;

        for cID=1:length(speakerModels)
            codebook = speakerModels{cID};
            if isempty(codebook), continue; end

            dVal= computeVQdist(Xtest, codebook);
            if dVal< bestDist
                bestDist= dVal;
                bestID= cID;
            end
        end

        predictions(t)= bestID;
        isOK= (bestID==spkID_true);
        if isOK
            correctCount= correctCount+1;
        end

        fprintf('Test #%d => ID=%d => predicted=%d, dist=%.3f %s\n',...
            t, spkID_true, bestID, bestDist, ternary(isOK,'[OK]','[ERR]'));
    end

    accuracy= (correctCount/nTests)*100;
    fprintf('Self-test done => accuracy=%.2f%%\n', accuracy);
end

%% subfunction
function distVal= computeVQdist(X, codebook)
% X => (#frames x dim)
% codebook => (#centroids x dim)
    N= size(X,1);
    dists= zeros(N,1);
    for i=1:N
        diffVal= codebook - X(i,:);
        dist2= sum(diffVal.^2,2);
        dists(i)= min(dist2);
    end
    distVal= sum(dists)/N;
end

function s= ternary(cond, sTrue, sFalse)
    if cond
        s=sTrue;
    else
        s=sFalse;
    end
end


