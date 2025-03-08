function main_allInOne_speakRecProject()
% main_allInOne_speakRecProject
%
% A consolidated demonstration script that unifies:
%   - Basic "Test 2" style STFT demonstration
%   - "Test 6": 2D scatter of MFCC for 2 different speakers
%   - "Test 7": training+testing speaker recognition
%   - "Test 8": notch filter experiment
%   - Param sweep approach
%
% This version modifies SECTION C so that we train on s1.wav~s8.wav 
% and also test on s1.wav~s8.wav (IDs=1..8).
%
% Usage:
%   1) Place this file in "D:\Program Files\Polyspace\R2021a\bin\"
%   2) Put s1.wav..s8.wav in "...\EEC201\train\" for training, 
%      and also in "...\EEC201\test\" for testing, 
%      if you want separate data or you can keep the same files in both folders.
%   3) Run >> main_allInOne_speakRecProject
%   4) Check console and figure outputs.

    clear; clc; close all;

    %% ========== SECTION A: Basic STFT demonstration ==========
    fprintf('\n--- SECTION A: Basic STFT demonstration ---\n\n');
    fileA = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s1.wav'; 
    if exist(fileA,'file')
        [sA, fsA] = audioread(fileA);
        fprintf('Loaded audio: %s (duration=%.2f sec)\n', fileA, length(sA)/fsA);

        figure('Name','SECTION A: STFT Demonstration','NumberTitle','off');
        subplot(2,1,1);
        timeVec= (0:length(sA)-1)/fsA;
        plot(timeVec, sA);
        title('Time-Domain Waveform'); xlabel('Time(s)'); ylabel('Amplitude');

        N=256; overlap=128; NFFT=512;
        [S,F,T] = stft(sA, fsA, 'Window',hamming(N), 'OverlapLength',overlap, ...
                       'FFTLength',NFFT,'FrequencyRange','onesided');
        subplot(2,1,2);
        spectrogram_dB = 20*log10(abs(S)+1e-3);
        imagesc(T,F,spectrogram_dB); axis xy; colorbar;
        title('Spectrogram (onesided)'); xlabel('Time(s)'); ylabel('Freq(Hz)');
    else
        warning('File not found => skip SECTION A: %s', fileA);
    end


    %% ========== SECTION B: 2D scatter of MFCC (Test6 style) ==========
    fprintf('\n--- SECTION B: 2D scatter of MFCC ---\n\n');
    filesB = {
       'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s2.wav'
       'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s8.wav'
    };
    if all(cellfun(@(f) exist(f,'file')==2, filesB))
        numFiltB  = 20;  
        numCoeffB = 12; 
        [mfcc_s2, fs2] = computeMFCC_forFile(filesB{1}, numFiltB, numCoeffB);
        [mfcc_s8, fs8] = computeMFCC_forFile(filesB{2}, numFiltB, numCoeffB);
        dims = [2,8];  
        s2data = mfcc_s2(dims,:)';
        s8data = mfcc_s8(dims,:)';

        figure('Name','SECTION B: 2D MFCC','NumberTitle','off');
        plot(s2data(:,1), s2data(:,2),'bx'); hold on;
        plot(s8data(:,1), s8data(:,2),'ro');
        xlabel(sprintf('MFCC dim %d', dims(1)));
        ylabel(sprintf('MFCC dim %d', dims(2)));
        legend('Speaker2','Speaker8','Location','best');
        title('Test6: 2D scatter of MFCC');
        grid on;
    else
        warning('Either s2.wav or s8.wav not found => skip SECTION B');
    end


    %% ========== SECTION C: Train on s1..s8, then test on s1..s8 ==========
    fprintf('\n--- SECTION C: Train on s1->s8, then test => IDs=1..8 ---\n\n');
    trainList = {
       'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\s1.wav'
       'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\s2.wav'
       'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\s3.wav'
       'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\s4.wav'
       'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\s5.wav'
       'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\s6.wav'
       'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\s7.wav'
       'D:\Program Files\Polyspace\R2021a\bin\EEC201\train\s8.wav'
    };
    testList = {
       {'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s1.wav',1}
       {'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s2.wav',2}
       {'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s3.wav',3}
       {'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s4.wav',4}
       {'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s5.wav',5}
       {'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s6.wav',6}
       {'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s7.wav',7}
       {'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s8.wav',8}
    };

    numFilt=20; numCoeff=12; codebookSize=8;
    speakerModels = train_speakers_demo(trainList, numFilt, numCoeff, codebookSize);
    [accuracy, ~] = test_speakers_demo(testList, speakerModels, numFilt, numCoeff);
    fprintf('Test => overall accuracy=%.2f%% (s1..s8)\n', accuracy);


    %% ========== SECTION D: Notch filter experiment (Test8 concept) ==========
    fprintf('\n--- SECTION D: Notch filter experiment ---\n\n');
    notchFreqs= [500,1000,2000];
    testFileD= 'D:\Program Files\Polyspace\R2021a\bin\EEC201\test\s1.wav';
    if exist(testFileD,'file')
        [sigD, fsD] = audioread(testFileD);
        for f0 = notchFreqs
            yN= applyNotch(sigD, fsD, f0, 0.1);
            cNotch= computeMFCC_fromSignal(yN, fsD, numFilt, numCoeff);
            fprintf('Notch@%4d => computed MFCC size: %dx%d\n',...
                f0, size(cNotch,1), size(cNotch,2));
        end
    else
        warning('File not found => skip Notch test: %s', testFileD);
    end


    %% ========== SECTION E: Param sweep approach (iterative train) ==========
    fprintf('\n--- SECTION E: Param sweep approach ---\n\n');
    codebookSizes= [4,8];
    filterList   = [20,26];
    coeffList    = [8,12];

    bestAcc=0;
    bestParam= [8,20,12];
    for cb= codebookSizes
        for nf= filterList
            for nc= coeffList
                spkModel = train_speakers_demo(trainList, nf, nc, cb);
                [accTest, ~] = test_speakers_demo(testList, spkModel, nf, nc);
                fprintf('(cb=%d, nf=%d, nc=%d) => acc=%.2f%%\n', cb,nf,nc,accTest);
                if accTest> bestAcc
                    bestAcc= accTest;
                    bestParam= [cb,nf,nc];
                end
            end
        end
    end
    fprintf('Best param= codebook=%d, filter=%d, mfcc=%d => acc=%.2f%%\n',...
        bestParam(1), bestParam(2), bestParam(3), bestAcc);

    fprintf('\nAll sections done.\n');
end  % End main function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% train_speakers_demo
function speakerModels = train_speakers_demo(trainList, numFilters, numCoeffs, codebookSize)
    nSpeak= length(trainList);
    speakerModels = cell(nSpeak,1);
    for i=1:nSpeak
        fPath= trainList{i};
        if exist(fPath,'file')==2
            [mfccMat, ~] = computeMFCC_forFile(fPath, numFilters, numCoeffs);
            codebook = runVQcodebook(mfccMat, codebookSize);
            speakerModels{i}= codebook;
        else
            warning('File not found => skip training: %s', fPath);
            speakerModels{i}=[];
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% test_speakers_demo
function [accuracy, predictions] = test_speakers_demo(testList, speakerModels, numFilters, numCoeffs)
    nTests= length(testList);
    predictions= zeros(nTests,1);
    correct=0;
    for t=1:nTests
        entry= testList{t};
        filePath= entry{1};
        realID  = entry{2};

        if exist(filePath,'file')==2
            [mfccTest, ~] = computeMFCC_forFile(filePath, numFilters, numCoeffs);
            [bestID, distVal] = findBestSpeaker_demo(mfccTest, speakerModels);
            predictions(t)= bestID;

            isOk= (bestID==realID);
            if isOk, correct=correct+1; end
            fprintf('Test: %s => spk#%d (true:%d), Dist=%.3f %s\n',...
                filePath, bestID, realID, distVal, ternary(isOk,'[OK]','[ERR]'));
        else
            warning('File not found => skip testing: %s', filePath);
        end
    end

    used= sum(predictions>0);
    if used>0
        accuracy= (correct/used)*100;
    else
        accuracy=0;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% computeMFCC_forFile => read WAV -> do MFCC => return [mfccMat, fsOut]
function [mfccMat, fsOut] = computeMFCC_forFile(wavPath, numFilters, numCoeffs)
    [y, fsOut] = audioread(wavPath);
    mfccMat    = computeMFCC_fromSignal(y, fsOut, numFilters, numCoeffs);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% computeMFCC_fromSignal => stft -> powerSpec -> melFB -> dct -> MFCC
function c = computeMFCC_fromSignal(signal, fs, numFilters, numCoeffs)
    alpha=0.95;
    for i=length(signal):-1:2
        signal(i)= signal(i) - alpha*signal(i-1);
    end

    N=256; overlap=100; NFFT=512;
    [S,~,~] = stft(signal, fs, 'Window',hamming(N),'OverlapLength',overlap,...
                   'FFTLength',NFFT,'FrequencyRange','onesided');
    ps= (abs(S).^2)/NFFT; 

    melFB = melfb(numFilters, NFFT, fs);

    if size(melFB,2)~= size(ps,1)
        error('Dimension mismatch => melFB=%dx%d, ps=%dx%d',...
            size(melFB,1), size(melFB,2), size(ps,1), size(ps,2));
    end

    nFrames= size(ps,2);
    c= zeros(numCoeffs, nFrames);
    for fIndex=1:nFrames
        pcol= ps(:,fIndex);
        melE= melFB * pcol;
        melE(melE<1e-12)= 1e-12;
        logMel= log(melE);
        dctCoeffs= dct(logMel);
        c(:,fIndex)= dctCoeffs(2:(numCoeffs+1)); 
    end

    c= c - (mean(c,2)+1e-8);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% runVQcodebook => minimal LBG approach
function codebook= runVQcodebook(mfccMat, codebookSize)
    epsVal=0.01; distThresh=1e-3;
    [D,N]= size(mfccMat);
    cbook= mean(mfccMat,2);
    count=1;
    while count< codebookSize
        cbook= [cbook.*(1+epsVal), cbook.*(1-epsVal)];
        count= size(cbook,2);

        prevDist=inf;
        while true
            distMat= zeros(count,N);
            for i=1:count
                diffVal= mfccMat- cbook(:,i);
                distMat(i,:)= sum(diffVal.^2,1);
            end
            [~, nearest]= min(distMat,[],1);

            newCB= zeros(D,count);
            for i=1:count
                idx= (nearest==i);
                if any(idx)
                    newCB(:,i)= mean(mfccMat(:,idx),2);
                else
                    newCB(:,i)= cbook(:,i);
                end
            end

            distortion=0;
            for i=1:count
                idx= (nearest==i);
                if any(idx)
                    dtmp= mfccMat(:,idx)- newCB(:,i);
                    distortion= distortion+ sum(dtmp.^2,'all');
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% findBestSpeaker_demo => pick minimal distance among codebooks
function [bestID, distVal]= findBestSpeaker_demo(mfccTest, speakerModels)
    bestID=0; distVal=inf;
    for sp=1:numel(speakerModels)
        cb= speakerModels{sp};
        if isempty(cb), continue; end
        dVal= computeVQDistortion_demo(mfccTest, cb);
        if dVal< distVal
            distVal= dVal;
            bestID= sp;
        end
    end
end

function distVal= computeVQDistortion_demo(mfccMat, codebook)
    [~, N]= size(mfccMat);
    total=0;
    for n=1:N
        vec= mfccMat(:,n);
        diff= codebook- vec;
        dists= sum(diff.^2,1);
        total= total+ min(dists);
    end
    distVal= total/N;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% applyNotch => used in Notch experiment
function yNotch= applyNotch(y, fs, centerFreq, width)
    fn= fs/2;
    freqRatio= centerFreq/fn;
    notchZeros= [exp(1i*pi*freqRatio), exp(-1i*pi*freqRatio)];
    notchPoles= (1-width)*notchZeros;
    b= poly(notchZeros);
    a= poly(notchPoles);
    yNotch= filter(b,a,y);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% melfb => pasted code
function m = melfb(p, n, fs)
    f0  = 700 / fs;
    fn2 = floor(n / 2);
    lr  = log(1 + 0.5 / f0) / (p + 1);

    bl = n * (f0 * (exp([0, 1, p, p+1] * lr) - 1));
    b1 = floor(bl(1)) + 1;
    b2 = ceil(bl(2));
    b3 = floor(bl(3));
    b4 = min(fn2, ceil(bl(4))) - 1;

    pf = log(1 + (b1 : b4) / n / f0) / lr;
    fp = floor(pf);
    pm = pf - fp;

    r = [fp(b2 : b4),      1 + fp(1 : b3)];
    c = [b2 : b4,          1 : b3] + 1;  
    v = 2 * [1 - pm(b2 : b4),  pm(1 : b3)];

    m = sparse(r, c, v, p, 1 + fn2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Ternary helper
function s= ternary(cond, sTrue, sFalse)
    if cond, s=sTrue; else, s=sFalse; end
end

