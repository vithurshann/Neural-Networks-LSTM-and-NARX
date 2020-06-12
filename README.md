# Neural-Networks-LSTM-and-NARX

Comparative Analysis of Long Short-Term Memory (LSTM) and NARX in Stock prediction<br/>

Files in Repository <br/>  
1. Test Data - TestData.csv <br/> 
2. TestFinalModel.m - RUN this file to test the best models <br/>
3. TrainFinalModel.m - contains the best NARX and LSTM models used for training <br/>

NARX (Open functions 5 in order to run scripts 6, 7, 8 and 9) <br/>
 4. NARX_GetData.m = Function used to get data for train the NARX models<br/>
 5. NARX_BaseLineModel.m - Script file used to create NARX baseline model by following MathsWork tutorial and NNStart<br/>
 6. NARX_HyperParameterOpt_StepAhead.m - Script file used to find optimal hyper-parameter through individual grid search for Step-Ahead network<br/>
 7. NARX_HyperParameterOpt_ClosedLoop.m - Script file used to find optimal hyper-parameter through individual grid search for Closed loop Network<br/>
 8. NARX_HyperParameterTuning.m - Script file used to find the optimal hyper-parameter through typical grid search<br/>

LSTM (Open functions 10 and 11 in order to run scripts 11 and 12) <br/>
9. LSTM_ForwardChaining.m - Function used to split the data into 3-fold for forward chaining <br/>
10. LSTM_CrossValidation.m - Function used to perform cross validation for bayesian optimization <br/>
11. LSTM_BaselineModel.m - Script file used to create LSTM baseline model by following MathsWork tutorial <br/>
12. LSTM_HyperParameterOpt_Bayesian.m - Scrip used to find optical hyper-parameters for LSTM model through bayesian optimization <br/>

RESULTS <br/>
13. NARX_TrainingResults.mat - NARX training results for best NARX model required to run the Testing <br/>
14. LSTM_TrainingResults.mat - LSTM training results for best LSTM model required to run the Testing <br/>

DATA PROCESSING <br/>
15. DataProcessing.ipynb - Jupyter notebook used for data processing and data standardization <br/> 
16. StandardDF.csv - Training data. <br/>
