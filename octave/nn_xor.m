#!/usr/bin/octave -q

# A basic Neural Network that implements the logical XOR operation
# This time we need to layers.  Basically, the hidden layer has two
# "neurons" one that tests an OR condition and one that tests a NAND
# condition.  Then output then looks for both of those to be true
# (that is, it does an AND on its inputs from the hidden layer).


printf( "Starting Neural Network simulation...\n")

printf( "\nInputs:\n")
#        [ bias var, x1, x2]
inputX1 = [1; 0 ; 0];  printf( "[ %d, %d ]\n", inputX1(2:3) )
inputX2 = [1; 0 ; 1];  printf( "[ %d, %d ]\n", inputX2(2:3) )
inputX3 = [1; 1 ; 0];  printf( "[ %d, %d ]\n", inputX3(2:3) )
inputX4 = [1; 1 ; 1];  printf( "[ %d, %d ]\n", inputX4(2:3) )
printf( "----------------------\n\n" )


nnWeightsTheta1 = [ -10 30; 20 -20; 20 -20];

temp1 = (inputX1')*nnWeightsTheta1;
temp2 = (inputX2')*nnWeightsTheta1;
temp3 = (inputX3')*nnWeightsTheta1;
temp4 = (inputX4')*nnWeightsTheta1;




printf( "OR\n" )
sigmoid(temp1(1))
sigmoid(temp2(1))
sigmoid(temp3(1))
sigmoid(temp4(1))

printf( "\nNAND\n" )
sigmoid(temp1(2))
sigmoid(temp2(2))
sigmoid(temp3(2))
sigmoid(temp4(2))



nnWeightsTheta2 = [-30; 20; 20];
temp1A1 = zeros(3,1);
temp2A1 = zeros(3,1);
temp3A1 = zeros(3,1);
temp4A1 = zeros(3,1);

temp1A1(1) = 1;
temp2A1(1) = 1;
temp3A1(1) = 1;
temp4A1(1) = 1;


temp1A1(2) = sigmoid(temp1(1));
temp2A1(2) = sigmoid(temp2(1));
temp3A1(2) = sigmoid(temp3(1));
temp4A1(2) = sigmoid(temp4(1));


temp1A1(3) = sigmoid(temp1(2));
temp2A1(3) = sigmoid(temp2(2));
temp3A1(3) = sigmoid(temp3(2));
temp4A1(3) = sigmoid(temp4(2));


temp1A1;
temp2A1;
temp3A1;
temp4A1;

temp1Output = (temp1A1')*nnWeightsTheta2;
temp2Output = (temp2A1')*nnWeightsTheta2;
temp3Output = (temp3A1')*nnWeightsTheta2;
temp4Output = (temp4A1')*nnWeightsTheta2;

#temp1Output
#temp2Output
#temp3Output
#temp4Output

printf( "\nXOR:\n")
sigmoid(temp1Output)
sigmoid(temp2Output)
sigmoid(temp3Output)
sigmoid(temp4Output)




