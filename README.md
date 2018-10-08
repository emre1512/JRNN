# JRNN

JRNN is an easy to use recurrent neural network (RNN) library implemented in Java. 


## Example Usage

```java
// Load data
List<float[]> traindata = DataLoader.loadData("C:\\Users\\user\\Documents\\AI Projects\\JRNN\\traindata.txt", Seperator.COMMA);
		
// Create neural network
NeuralNetwork nn = new NeuralNetwork(0.3f, 5E-3f, 3000000, true, ErrorFunction.MSE);
nn.addLayer(new HiddenLayer(2, ActivationFunction.SIGMOID));
nn.addLayer(new OutputLayer(3, ActivationFunction.SIGMOID)); // Neuron count must be same with the input vector length
	
// Train and test
NetworkController nc = new NetworkController(nn);
nc.showIterations(20000);
nc.trainNetwork(traindata);	

// Do regression for 10 steps, starting from the last input of train data
nc.doRegression(10);

// OR

// Predict next 6 output for the given input sequence
List<float[]> inputSequence = new ArrayList<>();
inputSequence.add(new float[]{0, 0, 1});
inputSequence.add(new float[]{0, 1, 0});
inputSequence.add(new float[]{0, 1, 1});
inputSequence.add(new float[]{0, 0, 1});
inputSequence.add(new float[]{0, 1, 0});
inputSequence.add(new float[]{0, 1, 1});

nc.predictNext(inputSequence, 6); 
```

## Sample Output
```
======= Training Starts =======
Max Epochs        : 3000000
Max Error         : 0.005
Learning Rate     : 0.3
===============================

======== Training Ends ========
Epochs : 313
Error  : 0.004987748
===============================

======= Regression Starts =======
0	0	1	
0	1	0	
0	1	1	
0	0	1	
0	1	0	
0	1	1	
0	0	1	
0	1	0	
0	1	1	
0	1	1	
====== Regression Finished ======
```

## Usage

```java
// Load data
List<float[]> traindata = DataLoader.loadData("C:\\Users\\user\\Documents\\AI Projects\\JRNN\\traindata.txt", Seperator.COMMA);
		
// Create neural network
// Set learning rate, max error, input data type (true for binary) anderror function.
NeuralNetwork nn = new NeuralNetwork(0.3f, 5E-3f, 3000000, true, ErrorFunction.MSE); 

// Add layers. Minimum 1 hidden and 1 output layer is required.
nn.addLayer(new HiddenLayer(2, ActivationFunction.SIGMOID));
nn.addLayer(new OutputLayer(3, ActivationFunction.SIGMOID)); // Neuron count must be same with the input vector length
	
// Train and test
NetworkController nc = new NetworkController(nn);
nc.showIterations(20000); // Show iteration stats every 20000 iterations
nc.trainNetwork(traindata);	

// Do regression for 10 steps, starting from the last input of train data
nc.doRegression(10);

// OR

// Predict next 6 output for the given input sequence
List<float[]> inputSequence = new ArrayList<>();
inputSequence.add(new float[]{0, 0, 1});
inputSequence.add(new float[]{0, 1, 0});
inputSequence.add(new float[]{0, 1, 1});
inputSequence.add(new float[]{0, 0, 1});
inputSequence.add(new float[]{0, 1, 0});
inputSequence.add(new float[]{0, 1, 1});

nc.predictNext(inputSequence, 6); 
```

## Data

The training data and test data should be in the form of:
```
1, 0, 1, 1
1, 0, 0, 1
1, 0, 1, 1
0, 0, 0, 1
1, 0, 1, 0
...
...
...
```

**Note: Data columns can be seperated by space or tab too.**


## Contribution

This library is still under development. You can open issues for the bugs you found. Also you can send pull requests for enhancements/bug fixes.

## License

See more at [LICENSE](https://github.com/emre1512/JANN/blob/master/LICENSE) page.