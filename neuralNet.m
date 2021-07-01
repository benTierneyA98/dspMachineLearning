%Project

%main project script
%read in data set from google folder
clc
clear all

dataFolder = 'Dataset';
ads = audioDatastore(fullfile(dataFolder), ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')

commands = categorical(["yes","no"]);


%%
%create subsets of data for training, testing

adsTrain = subset(ads,[1:750 2376:3125]);
countEachLabel(adsTrain)

adsValidation = subset(ads,[751:1500 3126:3875]);
countEachLabel(adsValidation)

%%
%convert the speech waveforms to auditory-based spectrograms

fs = 16e3; %sample rate of the data set.

segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
numBands = 40;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;

FFTLength = 512;
numBands = 50;

%audioFeatureExtractor object to perform feature extraction
afe = audioFeatureExtractor( ...
    'SampleRate',fs, ...
    'FFTLength',FFTLength, ...
    'Window',hann(frameSamples,'periodic'), ...
    'OverlapLength',overlapSamples, ...
    'barkSpectrum',true);
setExtractorParams(afe,'barkSpectrum','NumBands',numBands,'WindowNormalization',false);

%add zero padding to samples
x = read(adsTrain);
numSamples = size(x,1);
numToPadFront = floor( (segmentSamples - numSamples)/2 );
numToPadBack = ceil( (segmentSamples - numSamples)/2 );
xPadded = [zeros(numToPadFront,1,'like',x);x;zeros(numToPadBack,1,'like',x)];

%extract
features = extract(afe,xPadded);
[numHops,numFeatures] = size(features)

%distribute feature extraction for training set
if ~isempty(ver('parallel')) && ~reduceDataset
    pool = gcp;
    numPar = numpartitions(adsTrain,pool);
else
    numPar = 1;
end
%for each partition, read from the datastore, zero-pad the signal, and then extract the features.
parfor ii = 1:numPar
    subds = partition(adsTrain,numPar,ii);
    XTrain = zeros(numHops,numBands,1,numel(subds.Files));
    for idx = 1:numel(subds.Files)
        x = read(subds);
        xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
        XTrain(:,:,:,idx) = extract(afe,xPadded);
    end
    XTrainC{ii} = XTrain;
end

%array with spectrograms along the fourth dimension
XTrain = cat(4,XTrainC{:});
[numHops,numBands,numChannels,numSpec] = size(XTrain)


epsil = 1e-6; %offset
XTrain = log10(XTrain + epsil); %training dataset

%distribute feature extraction for validation set
if ~isempty(ver('parallel'))
    pool = gcp;
    numPar = numpartitions(adsValidation,pool);
else
    numPar = 1;
end

%for each partition, read from the datastore, zero-pad the signal, and then extract the features.
parfor ii = 1:numPar
    subds = partition(adsValidation,numPar,ii);
    XValidation = zeros(numHops,numBands,1,numel(subds.Files));
    for idx = 1:numel(subds.Files)
        x = read(subds);
        xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
        XValidation(:,:,:,idx) = extract(afe,xPadded);
    end
    XValidationC{ii} = XValidation;
end

%validation dataset
XValidation = cat(4,XValidationC{:});
XValidation = log10(XValidation + epsil);

%labels for use with neural network
YTrain = removecats(adsTrain.Labels);
YValidation = removecats(adsValidation.Labels);

%%
%plot spectrograms and play audio

specMin = min(XTrain,[],'all');
specMax = max(XTrain,[],'all');
idx = randperm(numel(adsTrain.Files),3);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
for i = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    subplot(2,3,i)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(i))))

    subplot(2,3,i+3)
    spect = (XTrain(:,:,1,idx(i))');
    pcolor(spect)
    caxis([specMin specMax])
    shading flat

    sound(x,fs)
    pause(2)
end