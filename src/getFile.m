function [s, fs, t] = getFile(id, fileType)
% getFile
%
% Reads s{id}.wav from
%   D:\Program Files\Polyspace\R2021a\bin\EEC201\Train
% ignoring "test" if we want single folder approach
%
% But we'll keep the "train"/"test" logic in case needed.

    if nargin<2, fileType="test"; end

    rootPath = 'D:\Program Files\Polyspace\R2021a\bin\EEC201';
    switch lower(fileType)
        case "train"
            folderName= "Test";
        case "test"
            folderName= "Test"; 
            % Force "Train" even if "test" => single folder approach
            % Or you can do: folderName= "Test"; if truly ignoring logic
        otherwise
            error('fileType must be "train" or "test"');
    end

    fileName= sprintf('s%d.wav', id);
    filePath= fullfile(rootPath, folderName, fileName);

    if ~isfile(filePath)
        error('File not found => %s', filePath);
    end

    [s, fs] = audioread(filePath);
    if size(s,2)>1
        s = s(:,1);
    end
    t= (0:length(s)-1)/fs;
end

