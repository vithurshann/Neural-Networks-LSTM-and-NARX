function [TrainInput,TrainTarget, ValidationInput, ValidationTarget] = LSTM_ForwardChaining(KValue)

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

%Return data based on the fold value
if KValue == 1 
    TrainInput = XTrain(:,1:1010);
    TrainTarget = YTrain(:,1:1010);
    ValidationInput = XTrain(:,1011:2020);
    ValidationTarget = YTrain(:,1011:2020);
    
elseif KValue == 2
    TrainInput = XTrain(:,1:2020);
    TrainTarget = YTrain(:,1:2020);
    ValidationInput = XTrain(:,2021:3028);
    ValidationTarget = YTrain(:,2021:3028);
    
else
    TrainInput = XTrain(:,1:3028);
    TrainTarget = YTrain(:,1:3028);
    ValidationInput = XTrain(:,3029:end);
    ValidationTarget = YTrain(:,3029:end);
end
end

