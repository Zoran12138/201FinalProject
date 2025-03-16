function sumAll_s1_s11()
% sumAll_s1_s11
%
% Reads s1.wav, s2.wav, ..., s11.wav from:
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Train
%
% Then sums their waveforms (element-wise) to produce one "combined" signal.
% Finally, plots the result in the time domain and optionally writes it out.
%
% Requirements:
%   1) All .wav files have the same sample rate fs
%   2) Each file has at least 'minLen' samples for consistent summation
%   3) If stereo, we only take the first channel
%   4) If lengths differ, we handle the min length for safe summation

    clear; clc; close all;

    folderPath = 'D:\Program Files\Polyspace\R2021a\bin\EEC201\Train';
    nFiles = 11;  % from s1 to s11
    
    % 1) 先循环读入每个 wav 文件，记录其长度，以找“最小长度 minLen”
    lenArray = zeros(nFiles,1);
    fsArray  = zeros(nFiles,1);
    waves    = cell(nFiles,1);
    
    for i=1:nFiles
        wavName  = sprintf('s%d.wav', i);
        fullPath = fullfile(folderPath, wavName);
        
        if ~exist(fullPath, 'file')
            error('File not found: %s', fullPath);
        end
        
        [tempWave, fs] = audioread(fullPath);
        if size(tempWave,2) > 1
            tempWave = tempWave(:,1);  % only first channel
        end
        
        waves{i}   = tempWave;
        fsArray(i) = fs;
        lenArray(i)= length(tempWave);
    end
    
    % Check if sample rates are consistent
    if any(fsArray ~= fsArray(1))
        warning('Not all WAV files have the same sample rate!');
    end
    fsUsed = fsArray(1);
    
    % minLen => 取所有文件波形的最小长度
    minLen = min(lenArray);
    fprintf('Minimum wave length among s1..s11 is %d samples.\n', minLen);
    
    % 2) 初始化一个合成波形 sumWave 并加和
    sumWave = zeros(minLen,1);
    for i=1:nFiles
        sumWave = sumWave + waves{i}(1:minLen);
    end
    
    % 3) 输出结果
    fprintf('Created a combined sum of %d signals (s1..s11).\n', nFiles);
    % 例如, 打印其前几样本
    disp('First 10 samples of sumWave:');
    disp(sumWave(1: min(10,minLen)));
    
    % 4) 绘制时域波形
    t = (0 : minLen-1) / fsUsed;
    figure('Name','Sum of s1..s11','NumberTitle','off');
    plot(t, sumWave, 'LineWidth',1);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Summed Signal (s1 + s2 + ... + s11)');
    grid on;
    
    % 5) 可选：若想写出 WAV
    % outPath = fullfile(folderPath, 'sum_s1_s11.wav');
    % audiowrite(outPath, sumWave, fsUsed);
    % disp(['Wrote summed wave to: ' outPath]);

end
