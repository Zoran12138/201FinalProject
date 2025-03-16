function codebook = runLBG(X, codebookSize)
% runLBG
%
% X => (#frames x dimension)
% codebookSize => M
% Output => (M x dimension)

    epsVal=0.01;
    distThresh=1e-4;
    [N, dim]= size(X);

    cbook= mean(X,1); % 1 x dim
    count=1;
    while count< codebookSize
        % split
        cbook= [cbook.*(1+epsVal); cbook.*(1-epsVal)];
        count= size(cbook,1);

        prevDist= inf;
        while true
            distMat= zeros(count,N);
            for ci=1:count
                diffVal= X - cbook(ci,:);
                distMat(ci,:)= sum(diffVal.^2,2);
            end
            [~, nearest]= min(distMat,[],1);

            newCB= zeros(count,dim);
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
                    dv= X(idx,:) - newCB(ci,:);
                    distortion= distortion+ sum(dv.^2,'all');
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
