function m = melfb(p, n, fs)
% MELFB constructs a Mel-spaced triangular filter bank (p x (n/2+1)) for MFCC extraction.
%
% Purpose and Impact:
%   This function maps a power spectrum (from an FFT of length n) onto the Mel scale,
%   approximating human auditory perception. By applying Mel-spaced triangular filters,
%   we compress linear frequency bins into perceptually spaced ones and reduce
%   dimensionality. The result is p filter outputs that emphasize lower-frequency
%   resolution while de‐emphasizing higher-frequency detail, improving speech or
%   speaker recognition tasks when followed by log + DCT => MFCC.
%
% Step-by-step:
%   (1) p: number of Mel filters, n: FFT length, fs: sampling rate.
%   (2) We compute f0=700/fs and define Mel scale log-range lr=log(1+0.5/f0)/(p+1).
%   (3) We find boundary indices b1..b4 that define the start/peak/end of triangular
%       filters. Then we create fractional positions pf, used to split filter amplitudes.
%   (4) Construct a sparse matrix m of size (p x (n/2+1)), where each row is a
%       triangular filter. Multiplying m by a power spectrum yields a p-dimensional
%       Mel-frequency representation.
%
% Usage in MFCC:
%   The output (Mel spectrum) is usually passed to log() then a DCT to produce
%   the final Mel-Frequency Cepstral Coefficients.

    % f0 is a scaling factor relating frequency to mel scale
    f0 = 700 / fs;               % typical mel formula portion
    fn2 = floor(n/2);            % Nyquist index
    lr  = log(1 + 0.5/f0) / (p+1);% log-range for mel intervals

    % bl => boundaries on a linear freq axis mapped back from mel scale
    bl = n * (f0 * (exp([0,1,p,p+1]*lr) - 1));
    b1 = floor(bl(1)) + 1;
    b2 = ceil( bl(2));
    b3 = floor(bl(3));
    b4 = min(fn2, ceil(bl(4))) - 1;

    % pf => fractional position on mel scale for bins from b1..b4
    pf = log(1 + (b1:b4)/(n*f0)) / lr;
    fp = floor(pf);
    pm = pf - fp;                % fraction to define slopes

    % Build sparse matrix:
    % r => row indices (which mel filter), c => column indices (which FFT bin)
    % v => amplitude values forming triangular filter shapes
    r = [ fp(b2:b4), 1+fp(1:b3) ];
    c = [ b2:b4,     1:b3 ] + 1;
    v = 2 * [ 1 - pm(b2:b4), pm(1:b3) ];
    m = sparse(r, c, v, p, fn2 + 1);
end
