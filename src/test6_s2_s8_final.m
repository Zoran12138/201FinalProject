function test6_s2_s8_final()
% test6_s2_s8_final
%
% A single script that:
%  1) Uses your s2data and s8data ([2 x 20]).
%  2) Trains an LBG codebook (M=4) for each.
%  3) Plots frames + codewords + connecting lines,
%     without adding extra "dataN" entries to the legend.

    clear; clc; close all;

    %%%%%%%%% Step 1: Define the [2 x 20] matrices you showed %%%%%%%%%
    % s2data
    s2data = [
       -0.3298   -0.6058    0.4333    0.4822    0.6567   -0.1643   -0.1523    0.0837   -0.8471    0.1759   -0.2405    0.8807    0.5849   -0.2772    0.0017   -0.9685   -1.3086   -2.5781   -2.5658   -3.6266
        0.2403    0.3635    0.0508   -0.5632   -0.6999   -1.1508   -0.8111   -0.9576   -0.1134   -0.2827   -0.0016   -0.4103   -0.3490   -0.2949   -0.1978   -0.0324   -0.3650    0.0147   -0.9959   -1.2113
    ];

    % s8data
    s8data = [
        0         0         0         0         0         0         0         0         0         0         0    1.9293    1.7488    0.3198    0.4602    0.6762    1.0823    1.0799    0.9838    1.4714
        0         0         0         0         0         0         0         0         0         0         0    0.4080    0.5294   -0.0091    0.0398    0.1354    0.4409   -0.1985    0.1822    0.5892
    ];

    %%%%%%%%% Step 2: Plot s2 frames, s8 frames %%%%%%%%%
    figure;
    % Keep track of the plot handles for the legend
    h_s2 = plot(s2data(1,:), s2data(2,:), 'bx','MarkerSize',8); hold on;
    h_s8 = plot(s8data(1,:), s8data(2,:), 'ro','MarkerSize',8);

    xlabel('Dimension 1');
    ylabel('Dimension 2');
    title('Test 6: s2 & s8 data => LBG codewords (M=4)');
    grid on;

    %%%%%%%%% Step 3: Train codebooks with M=4 %%%%%%%%%
    M = 4;
    codebook_s2 = trainVQ_LBG(s2data, M);
    codebook_s8 = trainVQ_LBG(s8data, M);

    %%%%%%%%% Step 4: Plot codewords %%%%%%%%%
    h_c2 = plot(codebook_s2(1,:), codebook_s2(2,:), 'ks','MarkerSize',10,'LineWidth',2);
    h_c8 = plot(codebook_s8(1,:), codebook_s8(2,:), 'md','MarkerSize',10,'LineWidth',2);

    % Step 5: Connect each frame to its nearest codeword (lines with HandleVisibility off)
    drawAssignments(s2data, codebook_s2, 'b');
    drawAssignments(s8data, codebook_s8, 'r');

    % Step 6: Update legend to only show frames & codewords
    legend([h_s2, h_s8, h_c2, h_c8], ...
           {'s2 frames','s8 frames','s2 codewords','s8 codewords'}, ...
           'Location','best');

    hold off;
end


%%%%%%%%%%%%%%%%%%%%%%% LOCAL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%

function codebook = trainVQ_LBG(data, M)
% trainVQ_LBG
% LBG approach to get codebook of size [2 x M], for data [2 x N].

    epsVal = 0.01;
    distThresh = 1e-3;
    [D, N] = size(data);

    cbook = mean(data, 2);  % single centroid = global mean
    count = 1;

    while count < M
        % split
        cbook = [cbook.*(1+epsVal), cbook.*(1-epsVal)];
        count = size(cbook, 2);

        prevDist = Inf;
        while true
            % assign
            distMat = zeros(count, N);
            for i = 1:count
                diffVal = data - cbook(:, i);
                distMat(i,:) = sum(diffVal.^2, 1);
            end
            [~, nearest] = min(distMat, [], 1);

            % update
            newCB = zeros(D, count);
            for i = 1:count
                idx = (nearest == i);
                if any(idx)
                    newCB(:, i) = mean(data(:, idx), 2);
                else
                    newCB(:, i) = cbook(:, i);
                end
            end

            % compute distortion
            distortion = 0;
            for i = 1:count
                idx = (nearest == i);
                if any(idx)
                    diffVal = data(:, idx) - newCB(:, i);
                    distortion = distortion + sum(diffVal.^2,'all');
                end
            end
            distortion = distortion / N;

            if abs(prevDist - distortion)/distortion < distThresh
                cbook = newCB;
                break;
            else
                cbook = newCB;
                prevDist = distortion;
            end
        end
    end

    codebook = cbook;
end

function drawAssignments(samples, codebook, lineColor)
% drawAssignments
% Draw lines from each sample to its assigned codeword, no extra legend items.

    [~, N] = size(samples);
    [~, C] = size(codebook);

    % Find nearest centroid
    distMat = zeros(C, N);
    for i = 1:C
        diffVal = samples - codebook(:, i);
        distMat(i,:) = sum(diffVal.^2, 1);
    end
    [~, nearest] = min(distMat, [], 1);

    % Draw line from sample to centroid
    for n = 1:N
        cIdx = nearest(n);
        xS = samples(1,n); yS = samples(2,n);
        xC = codebook(1,cIdx); yC = codebook(2,cIdx);

        % Important: 'HandleVisibility','off' => doesn't clutter legend
        line([xS xC], [yS yC], 'Color', lineColor, ...
             'LineStyle','-', 'HandleVisibility','off');
    end
end

