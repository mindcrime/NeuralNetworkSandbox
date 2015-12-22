#!/usr/bin/octave -q

# A basic Neural Network that implements the logical OR operation

printf( "Starting Neural Network simulation...\n")

#         [ bias var, x1, x2]
inputX1 = [1; 0 ; 0];
inputX2 = [1; 0 ; 1];
inputX3 = [1; 1 ; 0];
inputX4 = [1; 1 ; 1];

nnWeightsTheta = [ -10; 20; 20];

temp1 = (inputX1')*nnWeightsTheta;
temp2 = (inputX2')*nnWeightsTheta;
temp3 = (inputX3')*nnWeightsTheta;
temp4 = (inputX4')*nnWeightsTheta;

sigmoid(temp1)
sigmoid(temp2)
sigmoid(temp3)
sigmoid(temp4)

