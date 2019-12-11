% Extracting features from the audio signal
[audio_info, audio_features] = getInfoAndPlotGraph(331, 'recording_cv.m4a');
disp('Audio details'); disp(audio_info);

[frames, video] = getFramesFromVideo('video_introduction.mp4');
qualities = getImageQualities(video.NumFrames, frames);

mle_mfcc = mle(audio_info.mfcc);
mle_ssim = mle(qualities);

pd = makedist('Normal','mu',0,'sigma',1);
pdf_mfcc = pdf(pd, audio_info.mfcc);
pdf_mfcc(14 : 49) = 0;
pdf_ssim = pdf(pd, qualities);

llr = log(pdf_mfcc / pdf_ssim);
disp('LLR of the heterogenous system: '); disp(llr);

[p0, p1] = getCorrespondingProbabilities(llr);

kl = KLDiv(pdf_mfcc, pdf_ssim);
disp('Kullback-Leibler divergence: '); disp(kl);

cross_corr = xcorr(pdf_mfcc, pdf_ssim);
subplot(332); plot(cross_corr);

mfcc_shifted(2:50) = pdf_mfcc;
cross_self = xcorr(pdf_mfcc, mfcc_shifted);
cross_corr(98:99) = 0;
sum_corr = cross_self + cross_corr;
subplot(333); plot(sum_corr);

pdf_mfcc_shifted(2:50) = pdf_mfcc;

audio_bp = belief_propogation(length(audio_info.mfcc), audio_info.mfcc);
video_bp = belief_propogation(length(qualities), qualities);
llr_bp = belief_propogation(5, [1, -2, -1, -2, -1]);
llr_bp2 = belief_propogation(5, [1, -1, -2, -5, -6]);

subplot(334); plot(audio_bp);
subplot(335); plot(video_bp);
subplot(336); plot(llr_bp);
subplot(337); plot(llr_bp2);


function [p0, p1] = getCorrespondingProbabilities(log_likelyhood)
    p0 = exp(log_likelyhood) / (1 + exp(log_likelyhood)) ;
    p1 = 1 - p0; 
end

function indexes = getImageQualities(num_frames, frames)
    reference_image = getReferenceImage('video_introduction.mp4', 50);
    indexes = [];
    for i = 1:num_frames
        image = frames(1:720, 1280*(i-1) + 1:1280*i, 1:3); 
        ssimval = ssim(image, reference_image );
        indexes = [indexes ssimval];
    end 
end

function image = getReferenceImage(fileName, nTest)
    image = uint8(getBack(fileName, nTest, 'median'));
end

function backGrnd = getBack(fileName, nTest, method)
    if nargin < 2, nTest = 20; end
    if nargin < 3, method = 'median'; end
    v = VideoReader(fileName);
    nChannel = size(readFrame(v), 3);
    tTest = linspace(0, v.Duration-1/v.FrameRate , nTest);
    %allocate room for buffer
    buff = NaN([v.Height, v.Width, nChannel, nTest]);
    for fi = 1:nTest
        v.CurrentTime =tTest(fi);
        % read current frame and update model
        buff(:, :, :, mod(fi, nTest) + 1) = readFrame(v);
    end
    switch lower(method)
        case 'median'
            backGrnd = nanmedian(buff, 4);
        case 'mean'
            backGrnd = nanmean(buff, 4);
    end
end

function [frames, video]  = getFramesFromVideo(fileName)
    video = VideoReader(fileName);    
    frames = [];
    for img = 1:video.NumFrames
        b = read(video, img);
        frames = [frames b];
    end
end

function [audio_info, audio_features] = getInfoAndPlotGraph(subplotPosition, fileName) 
    plotPitchGraph(subplotPosition, fileName);
    [audio_info, audio_features] = getAudioInfo(fileName);
end


function plotPitchGraph(subplotPosition, fileName)
    [audioIn, fs] = audioread(fileName);
    [audioInfo, audioFeatures] = getAudioInfo(fileName);

    t = linspace(0,size(audioIn,1)/fs, size(audioFeatures, 1));
    subplot(subplotPosition); 
    plot(t, audioFeatures(:,audioInfo.pitch));
    title('Pitch');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
end


function [audio_info, audio_features] = getAudioInfo(fileName)
    [audio, fs] = audioread(fileName);

    aFE = audioFeatureExtractor( ...
    "SampleRate",fs, ...
    "Window",hamming(round(0.03*fs),"periodic"), ...
    "OverlapLength",round(0.02*fs), ...
    "mfcc",true, ...
    "mfccDelta",true, ...
    "mfccDeltaDelta",true, ...
    "pitch",true, ...
    "spectralCentroid",true);

    audio_features = extract(aFE, audio);
    audio_info = info(aFE);
end

function dist = KLDiv(P,Q)
    %  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
    %  distributions
    %  P and Q  are automatically normalised to have the sum of one on rows
    % have the length of one at each 
    % P =  n x nbins
    % Q =  1 x nbins or n x nbins(one to one)
    % dist = n x 1

    if size(P,2)~=size(Q,2)
        error('the number of columns in P and Q should be the same');
    end

    if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
       error('the inputs contain non-finite values!') 
    end

    % normalizing the P and Q
    if size(Q,1)==1
        Q = Q ./sum(Q);
        P = P ./repmat(sum(P,2),[1 size(P,2)]);
        temp =  P.*log(P./repmat(Q,[size(P,1) 1]));
        temp(isnan(temp))=0;% resolving the case when P(i)==0
        dist = sum(temp,2);


    elseif size(Q,1)==size(P,1)

        Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
        P = P ./repmat(sum(P,2),[1 size(P,2)]);
        temp =  P.*log(P./Q);
        temp(isnan(temp))=0; % resolving the case when P(i)==0
        dist = sum(temp,2);
    end
end

function f = gauss_distribution(x, mu, s)
    p1 = -.5 * ((x - mu)/s) ^ 2;
    p2 = (s * sqrt(2*pi));
    f = exp(p1) / p2;
end
