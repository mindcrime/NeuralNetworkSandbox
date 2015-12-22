function output = sigmoid( inX ),

  temp = (1 / (1 + (e^-inX)));

  if( temp >= .5 ),
    output = 1;
  else
    output = 0;
  end
  
end
