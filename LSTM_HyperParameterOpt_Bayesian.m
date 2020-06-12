%% Clear workspace and Command Window
clc; clear all; close all;

%% Specify Hyper-Parameters
optimVars = [
    optimizableVariable('numHiddenUnits',[25 500],'Type','integer')
    optimizableVariable('connectedLayers',[1 100],'Type','integer')
    optimizableVariable('learningRates',[1e-4 1e-2],'Transform','log')
    optimizableVariable('dropoutValues',[1e-2 1],'Transform','log')
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')
    optimizableVariable('MiniBatchSize',[25 500],'Type','integer')];

%% Optimize Hyper-Parameters
Fold = 3;
StoreRMSEScore = [];
for Chaining = 1 : Fold
    [TrainInput,TrainTarget, ValidationInput, ValidationTarget] = LSTM_ForwardChaining(Chaining);
    ObjFcn = @(T)LSTM_CrossValidation(TrainInput,TrainTarget,ValidationInput,ValidationTarget, ...
        T.numHiddenUnits, T.connectedLayers, T.learningRates, T.dropoutValues, T.L2Regularization, T.MiniBatchSize);
    BayesObject = bayesopt(ObjFcn,optimVars, ...
        'IsObjectiveDeterministic',false, ...
        'MaxObjectiveEvaluations', 100, ...
        'UseParallel',true);
    %Store best estimated feasible point for each fold
    if Chaining == 1
        foldOne = bestPoint(BayesObject);
    elseif Chaining == 2
        foldTwo = bestPoint(BayesObject); 
    else
        foldThree = bestPoint(BayesObject);
    end
end
%% Train and Test Best Model

numFeatures = 5;
numResponses = 1;
 
rng('default')

options = trainingOptions('adam', ...
'MaxEpochs',250, ...
'MiniBatchSize', 256, ...
'ValidationData', {ValidationInput, ValidationTarget}, ...
'ValidationFrequency', 25, ...
'ValidationPatience', 10, ...
'GradientThreshold',1, ...
'InitialLearnRate',learningRates, ... 
'LearnRateSchedule','piecewise', ...
'Verbose',1, ...
'Shuffle', 'never');

layers = [ ...
sequenceInputLayer(numFeatures)
lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
fullyConnectedLayer(connectedLayers)
dropoutLayer(dropoutValues)
fullyConnectedLayer(numResponses)
regressionLayer];
%Train network
net = trainNetwork(TrainInput,TrainTarget,layers,options);
%Predict
testPrediction = predict(net, TestInput);
%Get RMSE Score
OptimisedRMSE = sqrt(mean((TestTarget - testPrediction).^2));
RMSE = mean(OptimisedRMSE);

%%

