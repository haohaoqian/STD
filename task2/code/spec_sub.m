function output=spec_sub(signal, fs)
    IS = 0.25;
    SP = .4; %Shift percentage
    NoiseCounter = 0; NoiseLength = 9;
    Beta = .05;
    minalpha = 1; maxalpha = 3;
    minSNR = - 3; maxSNR = 20;
    W = fix(0.025 * fs); %Window length
    nfft = W;
    wnd = hamming(W);
    NIS = fix((IS * fs - W) / (SP * W) +1);% number of initial silence segments
    Gamma = 2;
    y = segment(signal, W, SP, wnd);
    Y = fft(y, nfft);
    YPhase = angle(Y(1: fix(end / 2) + 1, :)); %Noisy Speech Phase
    Y = abs(Y(1: fix(end / 2) + 1, :)) .^ Gamma; %Specrogram
    numberOfFrames = size(Y, 2);
    
    N = mean(Y(:, 1:NIS), 2); %initial Noise Power Spectrum mean

    alphaSlope = (minalpha - maxalpha) / (maxSNR - minSNR);
    alphaShift = maxalpha - alphaSlope * minSNR;

    BN = Beta * N;
    
    for i=1:numberOfFrames
        [SpeechFlag, NoiseCounter]=vad(Y(:,i) .^ (1 / Gamma), N .^ (1 / Gamma), NoiseCounter); %Magnitude Spectrum Distance VAD
        if SpeechFlag==0
            N = (NoiseLength * N + Y(:,i)) / (NoiseLength + 1); %Update and smooth noise
            BN = Beta*N;
        end
        
        SNR=10 * log(Y(:, i) ./ N);
        alpha = alphaSlope * SNR + alphaShift;
        alpha = max(min(alpha, maxalpha), minalpha);
        
        D=Y(:, i) - alpha .* N;
        
        X(:,i) = max(D, BN);
    end
    
    output = overlap(X .^ (1 / Gamma), YPhase, W, SP * W);
end    
    
function Seg=segment(signal,W,SP,Window)
    
    Window=Window(:);
    
    L=length(signal);
    SP = fix(W .* SP);
    N = fix((L - W) / SP +1);
    
    Index = (repmat(1:W, N, 1) + repmat((0:(N - 1))' * SP, 1, W))';
    hw = repmat(Window, 1, N);
    Seg = signal(Index) .* hw;
end   
        
function sig=overlap(XNEW, yphase, windowLen, ShiftLen);    
    FrameNum = size(XNEW, 2);
    Spec=XNEW .* exp(1j * yphase);
    
    if mod(windowLen , 2) %if FreqResol is odd
        Spec=[Spec; flipud(conj(Spec(2:end,:)))];
    else
        Spec=[Spec; flipud(conj(Spec(2:end-1,:)))];
    end
    sig = zeros((FrameNum-1) * ShiftLen + windowLen,1);
    for i = 1:FrameNum
        start = (i - 1) * ShiftLen + 1;
        spec = Spec(:, i);
        sig(start:start + windowLen - 1) = sig(start:start + windowLen - 1)+real(ifft(spec, windowLen));
    end
end
    
function [SpeechFlag, NoiseCounter]=vad(signal, noise, NoiseCounter)

    NoiseMargin = 3;
    Hangover = 5;
    SpectralDist= 20 * (log10(signal) - log10(noise));
    SpectralDist(SpectralDist < 0) = 0;
    
    Dist = mean(SpectralDist); 
    if (Dist < NoiseMargin) 
        NoiseCounter = NoiseCounter + 1;
    else
        NoiseCounter = 0;
    end

    if (NoiseCounter > Hangover) 
        SpeechFlag = 0;    
    else 
        SpeechFlag = 1; 
    end 
end
