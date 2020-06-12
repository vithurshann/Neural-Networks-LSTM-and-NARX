%% Clear workspace and Command Window
clc; clear all; close all;

%% Get Data
[XTrain, YTrain] = NARX_GetData();

X = tonndata(XTrain, false, false);
T = tonndata(YTrain, false, false);

%% Tuning: Training Funtion
% Network Architecture 
trainFcn = ["trainlm", "trainbr", "trainscg"];
inputDelays = 1:2;
feedbackDelays = 1:2;
hiddenLayerSize = 10;
Folds = 3;

%grid search table
NARXGrid = [];
%count value
Parameter = 1;

for trainingFunction = 1:length(trainFcn)
    StoreRMSEScore = [];
    StoreTimeTaken = [];
    for Chaining = 1:Folds
        tic 
        %NARX Network
        net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn(trainingFunction));
        [x,xi,ai,t] = preparets(net,X,{},T);
        % Setup Division of Data for Training, Validation, Testing
        % (ForwardChaining)
        if Chaining == 1
            net.divideFcn = 'divideind';
            net.divideParam.trainInd = 1:707;
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
        net.trainParam.epochs = 1000;
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
    NARXGrid{Parameter,2} = mean(StoreRMSEScore); 
    NARXGrid{Parameter,3} = mean(StoreTimeTaken);
    %Update count
    Parameter = Parameter + 1;
end

%% Tuning: Input Delays and Feedback Delays
% Network Architecture 
trainFcn =  "trainbr";
inputDelays = [2 5 10 25 50 100];
feedbackDelays = [2 5 10 25 50 100];
hiddenLayerSize = 100;
Folds = 3;

NARXGrid = [];
Parameter = 1;

for InputDelay = 1:length(inputDelays)
    for FeedbackDelay = 1:length(feedbackDelays)
        StoreRMSEScore = [];
        StoreTimeTaken = [];
        for Chaining = 1:Folds
            tic 
            %NARX Network
            net = narxnet(inputDelays(InputDelay),feedbackDelays(FeedbackDelay),hiddenLayerSize,'open',trainFcn);
            [x,xi,ai,t] = preparets(net,X,{},T);
            % Setup Division of Data for Training, Validation, Testing
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
            net.trainParam.epochs = 1000;
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
        NARXGrid{Parameter,1} = inputDelays(InputDelay);
        NARXGrid{Parameter,2} = feedbackDelays(FeedbackDelay);
        NARXGrid{Parameter,3} = mean(StoreRMSEScore); 
        NARXGrid{Parameter,4} = mean(StoreTimeTaken);
        %Update count
        Parameter = Parameter + 1;
    end
end

%% Tuning: Hidden Units
% Network Architecture 
trainFcn =  "trainbr";
inputDelays = 2;
feedbackDelays = 50;
hiddenLayerSize = [1 5 10 25 50 100 250 500];
Folds = 3;

NARXGrid = [];
Parameter = 1;

for HiddenLayer = 1:length(hiddenLayerSize)
    StoreRMSEScore = [];
    StoreTimeTaken = [];
    for Chaining = 1:Folds
        tic 
        %NARX Network
        net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize(HiddenLayer),'open',trainFcn);
        [x,xi,ai,t] = preparets(net,X,{},T);
        % Setup Division of Data for Training, Validation, Testing
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
        net.trainParam.epochs = 1000;
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
    NARXGrid{Parameter,1} = hiddenLayerSize(HiddenLayer);
    NARXGrid{Parameter,2} = mean(StoreRMSEScore); 
    NARXGrid{Parameter,3} = mean(StoreTimeTaken);
    %Update count
    Parameter = Parameter + 1;
end

%% %% prepare varibales for 3d graph
hiddenLayerGraph = NARXGrid(:,1);
hiddenLayerGraph = cell2mat(hiddenLayerGraph);
rmseGraph = NARXGrid(:,2);
rmseGraph = cell2mat(rmseGraph);
timeGraph = NARXGrid(:,3);
timeGraph = cell2mat(timeGraph);
%% Plot 3D graph
plot3(hiddenLayerGraph, rmseGraph,timeGraph,'r','LineWidth',2);
xlabel("Hidden Layers");
ylabel("RMSE");
zlabel("Time (seconds)");
title('Hidden Units Optimisation');
grid on
