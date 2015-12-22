#!/usr/bin/octave -q

# A basic Neural Network that implements the logical NOT operation

printf( "Starting Neural Network simulation...\n")

#         [ bias var, x1, x2]
inputX1 = [1; 0 ];
inputX2 = [1; 1 ];

nnWeightsTheta = [ 10; -20];

temp1 = (inputX1')*nnWeightsTheta;
temp2 = (inputX2')*nnWeightsTheta;


sigmoid(temp1)
sigmoid(temp2)


