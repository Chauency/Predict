# Predict

This project attempts to predict future location of a surrounding vehicle with historical data.

The model used is a LSTM-based deep model. It takes in a sequence of length 10 (0.9 second) with 6 features: x and y veloceties of ego and target vehicles, and relative x and y distance of target vechicle with respect to ego vehicle. The output of the model is 2 features: relative x and y distance of target vechicle with respect to ego vehicle, 1 second in the future.

The data used to train and test the model is obtained by processing the NGSIM dataset (https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) with scripts under "/data-process/".
