function mfccMat = computeMFCC_all(y, fs)
% computeMFCC_all => minimal example
% 
% 1) stft => powerSpec
% 2) mel => melSpec
% 3) log => dct => keep c2..c13 => 12 dims
%
% Output => (12 x #frames)

    N=256; overlap=128; NFFT=512;
    [S,~,~] = spectrogram(y, hamming(N), overlap, NFFT, fs);
    powerSpec= abs(S).^2;

    numFilters=26;
    melFB_ = melfb(numFilters, NFFT, fs);
    if size(melFB_,2)~= size(powerSpec,1)
        error('Dimension mismatch in computeMFCC_all');
    end

    melSpec= melFB_ * powerSpec;
    melSpec(melSpec<1e-12)= 1e-12;
    logMel= log(melSpec);

    dctAll= dct(logMel);   % => (26 x frames)
    mfccMat= dctAll(2:13,:);   % => c2..c13 => 12 dims
end

function m = melfb(p, n, fs)
    f0= 700/fs;
    fn2= floor(n/2);
    lr= log(1+0.5/f0)/(p+1);

    bl= n*(f0*(exp([0,1,p,p+1]*lr)-1));
    b1= floor(bl(1))+1;
    b2= ceil(bl(2));
    b3= floor(bl(3));
    b4= min(fn2, ceil(bl(4)))-1;

    pf= log(1+(b1:b4)/(n*f0))/lr;
    fp= floor(pf);
    pm= pf- fp;

    r= [fp(b2:b4), 1+fp(1:b3)];
    c= [b2:b4,     1:b3]+1;
    v= 2*[1-pm(b2:b4), pm(1:b3)];
    m= sparse(r,c,v,p, fn2+1);
end
