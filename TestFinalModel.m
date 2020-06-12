%% Clear workspace and Command Window
clc; clear all; close all;

%% Test NARX
%Load training results
load('NARX_TrainingResults.mat')

%Load data
data = readtable('TestData.csv');
data = table2timetable(data);

%Pepare data
dataTest = data(:, :);

%XTest = Testing inputs as array
XTest = dataTest(:, 1:end);
XTest = timetable2table(XTest);
XTest = table2array(XTest(:, 2:end));
%YTrain = Training target as array
YTest = dataTest{:,2};

% Test the Network
TestInput = tonndata(XTest, false, false);
TestTarget = tonndata(YTest, false, false);
[testinput,inputdelay,layerdelay,shiftedtarget] = preparets(nets,TestInput,{},TestTarget);
ytest = nets(testinput,inputdelay,layerdelay);
e = gsubtract(shiftedtarget,ytest);
eupdate = cell2mat(e);

%Get RMSE Score
OptimisedRMSEtest = sqrt(mean(eupdate.^2)); 
performance = perform(net,shiftedtarget,ytest);
disp("NARX RMSE Results:")
disp(OptimisedRMSEtest)

% Model vs Truth graph 
TS = size(shiftedtarget,2);
ytestCell = cell2mat(ytest);
shiftedtargetCell = cell2mat(shiftedtarget);
figure
subplot(2,1,1)
plot(ytestCell)
hold on
plot(shiftedtargetCell,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Price")
title("NARX: Model Output VS Ground Truth")
subplot(2,1,2)
stem(shiftedtargetCell - ytestCell)
ylabel("Error")
title("RMSE: " + OptimisedRMSE)

%% Clear workspace and Command Window
clc; clear all; close all;

%% Test LSTM
%Load training results
load('LSTM_TrainingResults.mat')

% Read data from CSV file and convert to timetable format
data = readtable('TestData.csv');
data = table2timetable(data);

% Prepare data
dataTest = data(:, :);

%XTest = Test input as array
XTest = dataTest(:, 1:end);
XTest = timetable2table(XTest);
XTest = table2array(XTest(:, 2:end));
%Ytest = Training target as array
YTest = dataTest{:,2};

TestInput = XTest';
TestTarget = YTest';

%Prediction
testPrediction = predict(net, TestInput);

%Get RMSE Score
OptimisedRMSE = sqrt(mean((TestTarget - testPrediction).^2));
disp("LSTM RMSE Results:")
disp(OptimisedRMSE)

%Get Training Time
timeTaken = toc;
disp("LSTM Time taken for training:")
disp(timeTaken)

% Model output and Ground Truth
figure
subplot(2,1,1)
plot(TestTarget)
hold on
plot(testPrediction,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Price")
title("LSTM: Model Output VS Ground Truth")

subplot(2,1,2)
stem(testPrediction - TestTarget)
ylabel("Error")
title("RMSE: " + OptimisedRMSE)