%% Clear workspace and Command Window
clc; clear all; close all;

%% Get Test data
%Read data from CSV file
data = readtable('MultipleInputST.csv');
data = [data{:,:}]
data = data.'

sequenceData = con2seq(data);

%Partition the Training data
numTimeStepsTrain = floor(0.8*numel(sequenceData));
dataTrain = sequenceData(1 : numTimeStepsTrain+1);

%Standardize the data
% mu = mean(dataTrain);
% sig = std(dataTrain);
% dataTrainStandardized = (dataTrain - mu) / sig;

%Predictors and Responses
XTrain = dataTrain(1 : end-1);
YTrain = dataTrain(2 : end);

%% LSTM Network Architecture
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

%% Train LSTM
net = trainNetwork(XTrain,YTrain,layers,options);

%% Forecast Future Time Steps
[XTest, YTest] = TestData();

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

rmse = sqrt(mean((YPred-YTest).^2))

%% Plot Forecast
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])

%% Plot Forecast and RMSE
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)