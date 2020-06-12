%% Clear workspace and Command Window
clc; clear all; close all;

%% Nonlinear Autoregressive Network with Exogenous Inputs (NARX)

data = readtable('standardDF.csv');
data = table2timetable(data);

%Split data(Training: 80%(Test: 20%)
dataTrain = data(1:3888, :); 
dataTest = data(3888:4860, :);

%XTrain = Training inputs as array
XTrain = dataTrain(:, 1:end);
XTrain = timetable2table(XTrain);
XTrain = table2array(XTrain(:, 2:end));
%YTrain = Training target as array
YTrain = dataTrain{:,2};

%XTest = Testing inputs as array
XTest = dataTest(:, 1:end);
XTest = timetable2table(XTest);
XTest = table2array(XTest(:, 2:end));
%YTrain = Training target as array
YTest = dataTest{:,2};

X = tonndata(XTrain, false, false);
T = tonndata(YTrain, false, false);

tic;
% Create a Nonlinear Autoregressive Network with External Input
trainFcn = 'trainbr';  % Bayesian Regularization
inputDelays = 2;
feedbackDelays = 10;
hiddenLayerSize = 50;

%NARX Network
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

%Processing functions (input and output)
net.inputs{1}.processFcns = {'removeconstantrows', 'mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows', 'mapminmax'};

%Model input
net.performFcn = 'mse'

%Stopping criteria
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 20;

% Prepare the Data for Training and Simulation
[x,xi,ai,t] = preparets(net,X,{},T);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;
net.divideParam.testRatio = 0;

% Train the Network
[net,tr] = train(net,x,t,xi,ai);

%Step-Ahead Prediction Network
nets = removedelay(net);
nets.name = [net.name ' - Predict One Step Ahead'];
view(nets)
[xs,xis,ais,ts] = preparets(nets,X,{},T);
ys = nets(xs,xis,ais);
stepAheadPerformance = perform(nets,ts,ys)
OptimisedRMSE = sqrt(stepAheadPerformance);

%Display RMSE
disp("NARX RMSE Results:")
disp(OptimisedRMSE)

%% Lont Short-Term Memory (LSTM)

% Read data from CSV file and convert to timetable format
data = readtable('standardDF.csv');
data = table2timetable(data);

%%Split data(Training: 80%(Train 70%, Validation 30%), Test: 20%)
dataTrain = data(1:3888, :); 

%XTrain = Training input as array
XTrain = dataTrain(:, 1:end);
XTrain = timetable2table(XTrain);
XTrain = table2array(XTrain(:, 2:end));
%YTrain = Training target as array
YTrain = dataTrain{:,2};

%Transpose Train data
XTrain = XTrain';
YTrain = YTrain';

TrainInput = XTrain(:,1:3028);
TrainTarget = YTrain(:,1:3028);
ValidationInput = XTrain(:,3029:end);
ValidationTarget = YTrain(:,3029:end);

tic
numFeatures = 5;
numResponses = 1;
learningRates = 0.003;
connectedLayers = 40;
dropoutValues = 0.028;
numHiddenUnits = 248;
MiniBatch = 182;

%network architecture
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(connectedLayers)
    dropoutLayer(dropoutValues)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize', MiniBatch, ...
    'ValidationData', {ValidationInput, ValidationTarget}, ...
    'ValidationFrequency', 25, ...
    'ValidationPatience', 10, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',learningRates, ... 
    'LearnRateSchedule','piecewise', ...
    'Verbose',1, ...
    'Shuffle', 'never');

%Train network
net = trainNetwork(TrainInput,TrainTarget,layers,options);