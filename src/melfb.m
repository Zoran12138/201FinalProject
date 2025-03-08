function m = melfb(p, n, fs)
% MELFB  Construct a mel-spaced filter bank (triangle filters).
%
%   M = melfb(p, n, fs)
%
%   This function generates a sparse matrix M of size [p, 1+floor(n/2)].
%   Each of the p rows contains one triangular filter in the frequency domain,
%   mapped onto the first (n/2 + 1) points of an FFT (i.e. from DC up to Nyquist).
%
%   Inputs:
%       p   : Number of filters in the Mel filter bank
%       n   : FFT length
%       fs  : Sampling rate in Hz
%
%   Output:
%       m   : A sparse matrix (p x (n/2 + 1)) that, when multiplied by
%             the power spectrum |X(k)|^2 (k=1..n/2+1), produces the
%             Mel-scale spectrum with p bins.
%
%   Typical usage example:
%       f = fft(s);             % compute FFT of signal s
%       m = melfb(p, n, fs);    % construct mel filter bank
%       n2 = 1 + floor(n/2);
%       z = m * abs(f(1:n2)).^2;% compute the mel-scale spectrum
%
%   To plot the filter bank responses (on a linear frequency axis):
%       figure;
%       plot(linspace(0, fs/2, 1+floor(n/2)), melfb(20, 256, fs)');
%       xlabel('Frequency (Hz)'); ylabel('Amplitude');
%       title('Mel-spaced Filter Bank');
%
% ---------------------------------------------------------------------
% The core idea is:
% 1) Map a set of Mel-spaced points to linear frequency.
% 2) Convert those frequency points to their corresponding FFT bin indices.
% 3) For each triangular filter, fill in amplitude values around those bins
%    to form adjacent overlapping triangles.
% 4) Return the filter bank as a sparse matrix 'm'.
%
% Implementation references:
%   [1] S. Davis and P. Mermelstein, "Comparison of parametric representations 
%       for monosyllabic word recognition in continuously spoken sentences",
%       IEEE Transactions on Acoustics, Speech, and Signal Processing, 1980.
%   [2] Rabiner, L., & Juang, B. H. (1993). Fundamentals of Speech Recognition.
% ---------------------------------------------------------------------

    % f0 represents a base factor for converting frequencies to the mel scale
    % Note: common formulas use fmel = 1127 * ln(1 + f/700),
    %       here we adopt an approach with 700/fs as a scaled version.
    f0 = 700 / fs;

    % fn2 is the index of the Nyquist frequency bin (integer floor)
    fn2 = floor(n / 2);

    % 'lr' is the "log-range": the mel-interval from DC to half of 1/f0,
    %  then we divide by (p+1) because we want p filters plus 2 boundary points
    %  (one at the low end, one at the high end).
    lr = log(1 + 0.5 / f0) / (p + 1);

    % Next, compute the left and right boundary in terms of "mel-bins" 
    % and convert them to linear frequency bin indices.
    % bl will hold four values:
    %   [bl(1), bl(2), bl(3), bl(4)] = 
    %   n * (f0 * (exp([0,1,p,p+1]*lr) - 1))
    % They represent the bin indices (or fractional bin) where the triangular
    % filters start, peak, and end in the FFT spectrum.

    bl = n * (f0 * (exp([0, 1, p, p+1] * lr) - 1));
    b1 = floor(bl(1)) + 1;        % Start bin index
    b2 = ceil(bl(2));             % Bin index near the 1st filter peak
    b3 = floor(bl(3));            % Bin index near the (p-th filter) peak
    b4 = min(fn2, ceil(bl(4))) - 1;% End bin index, cannot exceed fn2

    % Now we map each bin from b1 to b4 back to mel-space. We get fractional
    % positions in the mel scale by using log(1 + freq/f0). Then we scale by 1/lr.
    pf = log(1 + (b1 : b4) / n / f0) / lr; 
    % 'pf' is a vector of mel-bin indices in the continuous sense, for each freq bin.

    fp = floor(pf);   % the integer part
    pm = pf - fp;     % the fractional part, determines how far into the next filter

    % 'r', 'c', 'v' are used to build the sparse matrix:
    %  - row indices (r): which filter row does this bin contribute to?
    %  - col indices (c): which frequency bin (FFT index) are we talking about?
    %  - values (v): the amplitude factor of that bin in that row's filter.

    % we create two segments: the bins that belong to filter #b2..#b4,
    % and the bins that belong to filters #1..#b3. 
    % This ensures each filter slopes up and slopes down in adjacent fashion.

    r = [fp(b2 : b4),  1 + fp(1 : b3)];
    c = [b2 : b4,      1 : b3] + 1;   % +1 because Matlab indexing starts at 1
    % amplitude is 2 * (1 - fractional) or 2 * fractional, shaping the triangles
    v = 2 * [1 - pm(b2 : b4), pm(1 : b3)];

    % Construct sparse filter matrix of size p x (1 + fn2).
    % Each row is a filter, each column is a frequency bin up to Nyquist.
    m = sparse(r, c, v, p, 1 + fn2);

    % 'm' is the desired Mel filter bank matrix.
    %
    % Summary of final usage:
    %   powerSpectrum = abs(fftFrame(1 : fn2+1)).^2;
    %   melSpectrum   = m * powerSpectrum;
    %
    % That yields 'p' mel-frequency bands (one per triangular filter).
end
