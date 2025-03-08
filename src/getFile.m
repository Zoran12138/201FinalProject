function [s, fs, t] = getFile(id, fileType)
% GETFILE  Reads a WAV audio file from a specific folder structure.
%
%   [s, fs, t] = getFile(id, fileType)
%
%   Inputs:
%       id       : integer (e.g., 1, 2, 3, ...) used to form a file name 
%                  such as 's1.wav'
%       fileType : string "train" or "test" (default: "train"), 
%                  indicating the subfolder to look in.
%
%   Outputs:
%       s  : one-dimensional audio signal (amplitude array)
%       fs : sampling frequency in Hz
%       t  : time vector (seconds) of the same length as s
%
%   Example:
%       [s, fs, t] = getFile(1, "train");
%       sound(s, fs);

    if nargin < 2
        fileType = "train";  % Default to the 'train' folder if not specified
    end

    % Change this to your actual root path:
    rootPath = 'D:\Program Files\Polyspace\R2021a\bin\EEC201';

    % Construct the full path depending on fileType
    if fileType == "train"
        filePath = fullfile(rootPath, 'train', ['s' int2str(id) '.wav']);
    elseif fileType == "test"
        filePath = fullfile(rootPath, 'test',  ['s' int2str(id) '.wav']);
    else
        error('Unrecognized fileType. Must be "train" or "test".');
    end

    % Read the audio file
    [s, fs] = audioread(filePath);

    % If stereo, take only the first channel
    if size(s, 2) > 1
        s = s(:, 1);
    end

    % Create a time vector in seconds
    t = (0 : length(s) - 1) / fs;
end
