function MFCC = computeMFCC(melSpec, numCoeffs)
% computeMFCC
%
% Input:
%   melSpec   - (numFilters x numFrames) matrix
%   numCoeffs - number of MFCC to keep (e.g. 13)
%
% Output:
%   MFCC      - (numCoeffs x numFrames)
%
% Steps:
%   1) melSpec => log
%   2) dct(logMel)
%   3) keep first numCoeffs (often skip c0 => c2..c13)

    if nargin<2, numCoeffs=13; end

    melSpec(melSpec<1e-12) = 1e-12;  % avoid log(0)
    logMel = log(melSpec);

    % do DCT along freq axis => each column is one frame
    dctAll = dct(logMel);  % size => (numFilters x numFrames)

    % we typically keep c1..c(numCoeffs), or skip c0
    % here let's skip c0
    MFCC = dctAll(2:numCoeffs+1, :);
end
