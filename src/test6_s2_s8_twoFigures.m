function test6_s2_s8_twoFigures()
% test6_s2_s8_twoFigures
%
% This script does the following:
%   1) For speaker ID=2 and ID=8, read their "train" wav files via getFile()
%   2) Extract MFCC (2D => dimension #2 and #8)
%   3) First figure => "2D scatter of MFCC features" for both speakers
%   4) Second figure => "both ID2 and ID8 frames + codewords using LBG," 
%       ID2 codewords => pentagram, ID8 codewords => triangle
%
% Requirements:
%   - getFile.m is in path, properly pointing to
%       D:\Program Files\Polyspace\R2021a\bin\EEC201\Train
%   - computeMFCC_all or similar function that yields MFCC 
%   - runLBG_2D function for LBG clustering in 2D
%
% Usage:
%   >> test6_s2_s8_twoFigures

    clear; clc; close all;

    %% 1) Setup 
    spkIDs = [2, 8];
    dims   = [2, 8];   % e.g. dimension #2 => x-axis, #8 => y-axis
    codebookSize = 4;  % number of codewords in LBG

    colorList_frames = {'b','r'};  % speaker2 => blue, speaker8 => red
    markerList_frames= {'o','x'};  % speaker2 => circle, speaker8 => x

    % For codewords, ID2 => pentagram, ID8 => triangle
    codewordMarkerID2 = 'p';  % pentagram
    codewordMarkerID8 = '^';  % triangle

    % We'll store data for each speaker in data2D{i} => shape 2 x N
    data2D = cell(numel(spkIDs), 1);

    %% 2) Loop each speaker, read audio, compute MFCC => store 2D
    for i=1:numel(spkIDs)
        spk = spkIDs(i);
        [y, fs, ~] = getFile(spk, "train");

        % simple pre-process: DC removal + normalize
        y = y - mean(y);
        pk = max(abs(y));
        if pk>1e-12, y = y/pk; end

        MFCC = computeMFCC_all(y, fs);  % user-defined function 
        % pick 2D
        xdata = MFCC(dims(1), :);
        ydata = MFCC(dims(2), :);

        data2D{i} = [xdata; ydata];
    end

    %% 3) First figure => 2D scatter of MFCC features (both speakers)
    figure('Name','(Fig1) 2D scatter of MFCC','NumberTitle','off');
    hold on; grid on;

    for i=1:numel(spkIDs)
        spk   = spkIDs(i);
        color = colorList_frames{i};
        mark  = markerList_frames{i};
        X2D   = data2D{i};
        if isempty(X2D), continue; end

        scatter(X2D(1,:), X2D(2,:), 30, color, mark, ...
            'DisplayName', sprintf('Speaker %d frames', spk));
    end

    xlabel(sprintf('MFCC dimension %d', dims(1)));
    ylabel(sprintf('MFCC dimension %d', dims(2)));
    title('2D Scatter of MFCC Features (Speaker 2 & 8)');
    legend('Location','best');
    hold off;

    %% 4) Second figure => frames + codewords 
    figure('Name','(Fig2) LBG codewords for s2 & s8','NumberTitle','off');
    hold on; grid on;

    for i=1:numel(spkIDs)
        spk   = spkIDs(i);
        color = colorList_frames{i};
        mark  = markerList_frames{i};
        X2D   = data2D{i};
        if isempty(X2D), continue; end

        % plot frames first
        scatter(X2D(1,:), X2D(2,:), 30, color, mark, ...
            'DisplayName', sprintf('Spk%d frames', spk));

        % run LBG => codewords
        X  = X2D';  % shape => N x 2
        CW = runLBG_2D(X, codebookSize);

        % pick codeword marker based on speaker
        switch spk
            case 2
                cwMarker = codewordMarkerID2;   % pentagram
            case 8
                cwMarker = codewordMarkerID8;   % triangle
            otherwise
                cwMarker = 's'; % fallback
        end

        scatter(CW(:,1), CW(:,2), 100, color, cwMarker, 'filled',...
            'DisplayName', sprintf('Spk%d codewords', spk));
    end

    xlabel(sprintf('MFCC dimension %d', dims(1)));
    ylabel(sprintf('MFCC dimension %d', dims(2)));
    title(sprintf('LBG codewords (M=%d) for s2 & s8', codebookSize));
    legend('Location','best');
    hold off;
end

%% computeMFCC_all (example)
function MFCC = computeMFCC_all(y, fs)
    % do short-time => mel => log => dct => pick c2..c13 => 12 dims
    N=256; overlap=128; NFFT=512;
    [S,~,~] = spectrogram(y, hamming(N), overlap, NFFT, fs);
    powerSpec = abs(S).^2;

    numFilters=26;
    melFB_ = melfb(numFilters, NFFT, fs);
    if size(melFB_,2)~= size(powerSpec,1)
        error('Dimension mismatch');
    end
    melSpec = melFB_ * powerSpec;
    melSpec(melSpec<1e-12)=1e-12;
    logMel = log(melSpec);
    dctAll = dct(logMel); 
    MFCC   = dctAll(2:13,:);  % c2..c13 => (12 x frames)
end

%% runLBG_2D
function codebook = runLBG_2D(X, K)
    % X => Nx2
    epsVal=0.01; distThresh=1e-4;
    [N, dim] = size(X);
    if dim~=2
        error('runLBG_2D => Nx2 input required');
    end

    cbook= mean(X,1); 
    count=1;
    while count< K
        cbook= [cbook.*(1+epsVal); cbook.*(1-epsVal)];
        count= size(cbook,1);

        prevDist=inf;
        while true
            distMat= zeros(count,N);
            for ci=1:count
                diffVal= X - cbook(ci,:);
                distMat(ci,:)= sum(diffVal.^2,2);
            end
            [~, nearest]= min(distMat,[],1);

            newCB= zeros(count,2);
            for ci=1:count
                idx= (nearest==ci);
                if any(idx)
                    newCB(ci,:)= mean(X(idx,:),1);
                else
                    newCB(ci,:)= cbook(ci,:);
                end
            end

            distortion=0;
            for ci=1:count
                idx= (nearest==ci);
                if any(idx)
                    diffVal= X(idx,:) - newCB(ci,:);
                    distortion= distortion+ sum(diffVal.^2,'all');
                end
            end
            distortion= distortion/N;

            if abs(prevDist-distortion)/distortion< distThresh
                cbook= newCB;
                break;
            else
                cbook= newCB;
                prevDist= distortion;
            end
        end
    end
    codebook= cbook;
end

%% melfb
function m = melfb(p, n, fs)
    f0= 700/fs;
    fn2= floor(n/2);
    lr= log(1+0.5/f0)/(p+1);

    bl= n*(f0*(exp([0,1,p,p+1]*lr)-1));
    b1= floor(bl(1))+1; 
    b2= ceil(bl(2));
    b3= floor(bl(3));
    b4= min(fn2, ceil(bl(4)))-1;

    pf= log(1 + (b1:b4)/(n*f0))/lr;
    fp= floor(pf);
    pm= pf - fp;

    r= [fp(b2:b4), 1+fp(1:b3)];
    c= [b2:b4,     1:b3]+1;
    v= 2*[1-pm(b2:b4), pm(1:b3)];
    m= sparse(r,c,v,p, fn2+1);
end

