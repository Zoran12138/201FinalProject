function demo_plotMelFB()
% demo_plotMelFB
%
% This script shows how to plot mel-spaced filterbank responses
% up to roughly 7000 Hz, using:
%   p=20 filters, n=256 for FFT size, fs=12500 Hz
%
% Usage:
%   >> demo_plotMelFB

    % Number of mel filters
    p = 20;

    % FFT length
    n = 256;

    % Sampling frequency
    fs = 12500; 
    % (So fs/2 = 6250, close to 7000 in your figure you might use fs=14000 if you want more range)

    % Compute the mel filterbank
    M = melfb(p, n, fs);  
    % M will be a (p x n/2+1) sparse matrix

    % Frequency axis for plotting => from 0 to fs/2
    freqAxis = linspace(0, fs/2, size(M,2));

    % Plot each filter as a separate curve
    figure;
    plot(freqAxis, M');
    title('Mel-spaced Filterbank Responses');
    xlabel('Frequency (Hz)');
    ylabel('Amplitude');
    grid on;
end

%% Subfunction: melfb
function m = melfb(p, n, fs)
% melfb  Construct a mel-spaced filter bank 
%
%   m = melfb(p, n, fs) returns a p x (n/2+1) matrix of triangular
%   filter amplitudes, spaced on the mel scale between 0 and fs/2.

    f0  = 700 / fs;
    fn2 = floor(n / 2);
    lr  = log(1 + 0.5 / f0) / (p + 1);

    bl = n * (f0 * (exp([0 1 p p+1] * lr) - 1));
    b1 = floor(bl(1)) + 1;
    b2 = ceil(bl(2));
    b3 = floor(bl(3));
    b4 = min(fn2, ceil(bl(4))) - 1;

    pf = log(1 + (b1:b4)/(n * f0)) / lr;
    fp = floor(pf);
    pm = pf - fp;

    r = [fp(b2:b4),     1+fp(1:b3)];
    c = [b2:b4,         1:b3] + 1;
    v = 2 * [1-pm(b2:b4), pm(1:b3)];
    m = sparse(r, c, v, p, fn2 + 1);
end
