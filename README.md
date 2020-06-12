# Neural-Networks-LSTM-and-NARX

%%%%
Comparative Analysis of Long Short-Term Memory (LSTM) and NARX in Stock prediction
By: Vithurshan Vijayachandran
%%%%

Files in Folder: 

1. Test Data - TestData.csv

2. TestFinalModel.m - RUN this file to test the best models

3. TrainFinalModel.m - contains the best NARX and LSTM models used for training


NARX
4. NARX_GetData.m = Function used to get data for train the NARX models

5. NARX_BaseLineModel.m - Script file used to create NARX baseline model by following MathsWork tutorial and NNStart

6. NARX_HyperParameterOpt_StepAhead.m - Script file used to find optimal hyper-parameter through individual grid search for Step-Ahead network

7. NARX_HyperParameterOpt_ClosedLoop.m - Script file used to find optimal hyper-parameter through individual grid search for Closed loop Network

8. NARX_HyperParameterTuning.m - Script file used to find the optimal hyper-parameter through typical grid search

Open functions 5 in order to run scripts 6, 7, 8 and 9


LSTM
9. LSTM_ForwardChaining.m - Function used to split the data into 3-fold for forward chaining 

10. LSTM_CrossValidation.m - Function used to perform cross validation for bayesian optimization

11. LSTM_BaselineModel.m - Script file used to create LSTM baseline model by following MathsWork tutorial 

12. LSTM_HyperParameterOpt_Bayesian.m - Scrip used to find optical hyper-parameters for LSTM model through bayesian optimization 

Open functions 10 and 11 in order to run scripts 11 and 12.


RESULTS
13. NARX_TrainingResults.mat - NARX training results for best NARX model required to run the Testing

14. LSTM_TrainingResults.mat - LSTM training results for best LSTM model required to run the Testing


DATA PROCESSING 
15. DataProcessing.ipynb - Jupyter notebook used for data processing and data standardization  

16. StandardDF.csv - Training data.
