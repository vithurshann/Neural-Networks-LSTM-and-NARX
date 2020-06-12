function [XTrain,YTrain] = NARX_GetData()

% Read data from CSV file and convert to timetable format
data = readtable('standardDF.csv');
data = table2timetable(data);

%Split data(Training: 80%(Test: 20%)
dataTrain = data(1:3888, :); 
%dataVal = data(2721:3888, :);

%XTrain = Training inputs as array
XTrain = dataTrain(:, 1:end);
XTrain = timetable2table(XTrain);
XTrain = table2array(XTrain(:, 2:end));
%YTrain = Training target as array
YTrain = dataTrain{:,2};

%XValidation = Test inputs as array
% XValidation = dataVal(:, 1:end);
% XValidation = timetable2table(XValidation);
% XValidation = table2array(XValidation(:, 2:end));
%YValidation = Test target as array
% YValidation = dataVal{:,2};
end

