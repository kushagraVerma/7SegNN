# 7SegNN
My first neural network  

This neural network takes seven-segment display as 7-bit input and tries to figure out which number it is from 0-9  
Neural network implementation itself is all-purpose (but probably limited to computation resources)  
Made my own implementation for matrices and matrix operations here  
Neural network stuff inspired largely by 3Blue1Brown's videos  
Uses p5.js for drawing and animation  

By default, the neural network instance here has an input layer of 7 neurons, hidden layer of 11, output layer of 10 for demonstation purposes, but does its job without the hidden layer  
By default, the neural network class uses the sigmoid function to force neuron output to values between 0 and 1; a different function can be passed as its argument, but its derivative would need to be passed as another argument as well  
