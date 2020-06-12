%% Clear workspace and Command Window
clc; clear all; close all;

%% Get Test data
[XTrain, YTrain] = NARX_GetData();

X = tonndata(XTrain, false, false);
T = tonndata(YTrain, false, false);

%% Closed Loop

% Network Architecture 
trainFcn = "trainbr";
inputDelays = [2 5 10];
feedbackDelays = [10 15 25];
hiddenLayerSize = [25 50 100];
Folds = 3; 

% Optimization 
NARXGrid = [];
Parameter = 1;

for trainingFunction = 1:length(trainFcn)
    for hiddenLayer = 1:length(hiddenLayerSize)
        for feedbackDelay = 1:length(feedbackDelays)
            for inputDelay = 1:length(inputDelays)
                StoreRMSEScore = [];
                StoreTimeTaken = [];
                
                for Chaining = 1:Folds
                    tic 
                    %NARX Network
                    net = narxnet(1:inputDelays(inputDelay),1:feedbackDelays(feedbackDelay),hiddenLayerSize(hiddenLayer),'open',trainFcn(trainingFunction));
                    [x,xi,ai,t] = preparets(net,X,{},T);
                    % Setup Division of Data for Training and Validation
                    if Chaining == 1
                        net.divideFcn = 'divideind';
                        net.divideParam.trainInd = 1:1010;
                        net.divideParam.valInd = 1011:2020;
                        %net.divideParam.testInd = 1011:2020;
                    elseif Chaining == 2
                        net.divideFcn = 'divideind';
                        net.divideParam.trainInd = 1:2020;
                        net.divideParam.valInd = 2021:3028;
                        %net.divideParam.testInd = 2021:3028;
                    else
                        net.divideFcn = 'divideind';
                        net.divideParam.trainInd = 1:3028;
                        net.divideParam.valInd = 3029:3888;
                        %net.divideParam.testInd = 3029:3888;
                    end
                    %Processing functions (input and output)
                    net.inputs{1}.processFcns = {'removeconstantrows', 'mapminmax'};
                    net.inputs{2}.processFcns = {'removeconstantrows', 'mapminmax'};
                    %Model input
                    net.performFcn = 'mse'
                    %Stopping criteria
                    net.trainParam.epochs = 2000;
                    net.trainParam.max_fail = 10;
                    %Train NARX
                    [net,tr] = train(net,x,t,xi,ai, 'useParallel', 'yes');
                    %Closed Loop Network
                    netc = closeloop(net);
                    [xc,xic,aic,tc] = preparets(netc,X,{},T);
                    yc = netc(xc,xic,aic);
                    closedLoopPerformance = perform(net,tc,yc);
                    %Get RMSE Score
                    OptimisedRMSE = sqrt(closedLoopPerformance);
                    StoreRMSEScore = [StoreRMSEScore OptimisedRMSE];
                    %Get Training Time
                    TrainingTime = toc;
                    StoreTimeTaken = [StoreTimeTaken TrainingTime];
                end
                %Update Table
                NARXGrid{Parameter,1} = char(trainFcn(trainingFunction));
                NARXGrid{Parameter,2} = hiddenLayerSize(hiddenLayer); 
                NARXGrid{Parameter,3} = feedbackDelays(feedbackDelay); 
                NARXGrid{Parameter,4} = inputDelays(inputDelay); 
                NARXGrid{Parameter,5} = mean(StoreRMSEScore);
                NARXGrid{Parameter,6} = mean(StoreTimeTaken);
                %Update parameter count
                Parameter = Parameter + 1;
            end
        end
    end
end

%% One-Step Ahead

% Network Architecture 
trainFcn = "trainbr";
inputDelays = [2 5 10];
feedbackDelays = [2 15 25];
hiddenLayerSize = [10 25 50];
Folds = 3; 

% Optimization 
NARXGrid = [];
Parameter = 1;

for trainingFunction = 1:length(trainFcn)
    for hiddenLayer = 1:length(hiddenLayerSize)
        for feedbackDelay = 1:length(feedbackDelays)
            for inputDelay = 1:length(inputDelays)
                StoreRMSEScore = [];
                StoreTimeTaken = [];
                
                for Chaining = 1:Folds
                    tic 
                    %NARX Network
                    net = narxnet(1:inputDelays(inputDelay),1:feedbackDelays(feedbackDelay),hiddenLayerSize(hiddenLayer),'open',trainFcn(trainingFunction));
                    [x,xi,ai,t] = preparets(net,X,{},T);
                    % Setup Division of Data for Training and Validation
                    if Chaining == 1
                        net.divideFcn = 'divideind';
                        net.divideParam.trainInd = 1:1010;
                        net.divideParam.valInd = 1011:2020;
                        %net.divideParam.testInd = 1011:2020;
                    elseif Chaining == 2
                        net.divideFcn = 'divideind';
                        net.divideParam.trainInd = 1:2020;
                        net.divideParam.valInd = 2021:3028;
                        %net.divideParam.testInd = 2021:3028;
                    else
                        net.divideFcn = 'divideind';
                        net.divideParam.trainInd = 1:3028;
                        net.divideParam.valInd = 3029:3888;
                        %net.divideParam.testInd = 3029:3888;
                    end
                    %Processing functions (input and output)
                    net.inputs{1}.processFcns = {'removeconstantrows', 'mapminmax'};
                    net.inputs{2}.processFcns = {'removeconstantrows', 'mapminmax'};
                    %Model input
                    net.performFcn = 'mse'
                    %Stopping criteria
                    net.trainParam.epochs = 2000;
                    net.trainParam.max_fail = 10;
                    %Train NARX
                    [net,tr] = train(net,x,t,xi,ai, 'useParallel', 'yes');
                    % Step-Ahead Prediction Network
                    nets = removedelay(net);
                    [xs,xis,ais,ts] = preparets(nets,X,{},T);
                    ys = nets(xs,xis,ais);
                    stepAheadPerformance = perform(nets,ts,ys)
                    %Get RMSE Score
                    OptimisedRMSE = sqrt(stepAheadPerformance);
                    StoreRMSEScore = [StoreRMSEScore OptimisedRMSE];
                    %Get Training Time
                    TrainingTime = toc;
                    StoreTimeTaken = [StoreTimeTaken TrainingTime];
                end
                %Update Table
                NARXGrid{Parameter,1} = char(trainFcn(trainingFunction));
                NARXGrid{Parameter,2} = hiddenLayerSize(hiddenLayer); 
                NARXGrid{Parameter,3} = feedbackDelays(feedbackDelay); 
                NARXGrid{Parameter,4} = inputDelays(inputDelay); 
                NARXGrid{Parameter,5} = mean(StoreRMSEScore);
                NARXGrid{Parameter,6} = mean(StoreTimeTaken);
                %Update parameter count
                Parameter = Parameter + 1;
            end
        end
    end
end

%% Plots

figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotresponse(t,y)
%figure, ploterrcorr(e)
%figure, plotinerrcorr(x,e)
