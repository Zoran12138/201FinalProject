function melSpec = applyMelFB(S, fs, numFilters)
% applyMelFB
%
% Inputs:
%   S           - STFT complex matrix (freq x time)
%   fs          - sampling rate
%   numFilters  - number of Mel filters, e.g. 26
%
% Output:
%   melSpec     - (numFilters x time), the mel-filtered spectral magnitudes
%
% Procedure:
%   1) Convert STFT to power spectrum => |S|^2
%   2) Multiply by melFB => apply triangular filters
%   3) result => melSpec

    [freqBins, numFrames] = size(S);

    powerSpec = abs(S).^2; % freq x time
    NFFT = (freqBins-1)*2; % typical => if we used onesided
    melFB_ = melfb(numFilters, NFFT, fs);

    % note: melFB_ is (numFilters x (NFFT/2 +1)) => must match freqBins
    if size(melFB_,2)~= freqBins
        error('Dimension mismatch: melFB=%dx%d, STFT freqBins=%d',...
            size(melFB_,1), size(melFB_,2), freqBins);
    end

    % melSpec => (numFilters x time)
    melSpec = melFB_ * powerSpec;
end
