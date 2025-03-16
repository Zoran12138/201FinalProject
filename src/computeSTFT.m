function [S, f, t] = computeSTFT(signal, fs, N, noverlap, NFFT)
% computeSTFT
% Inputs:
%   signal    - time-domain samples (1D)
%   fs        - sampling rate
%   N         - window length
%   noverlap  - overlap length
%   NFFT      - FFT size
% Outputs:
%   S         - STFT complex matrix
%   f         - freq axis in Hz
%   t         - time axis in seconds
%
%  Example usage:
%   [S, f, t] = computeSTFT(y, 8000, 256, 128, 512);

    if nargin<5, NFFT = N; end
    window = hamming(N);

    % [S, F, T] from spectrogram => dimension S: freq x time
    [S, f, t] = spectrogram(signal, window, noverlap, NFFT, fs);

    % S is complex => magnitude^2 is periodogram
end
