% File: plotSTFT.m
function plotSTFT(signal, fs, N)
    M = round(N/3);
    noverlap = N - M;
    figure;
    spectrogram(signal, hamming(N), noverlap, N, fs, 'yaxis');
    title(['Spectrogram (N=' num2str(N) ')']);
    colormap jet; colorbar;
end
