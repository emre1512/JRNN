package main;

import java.util.ArrayList;
import java.util.List;
import io.DataLoader;
import io.DataLoader.Seperator;
import math.IActivation.ActivationFunction;
import math.IError.ErrorFunction;
import math.MeanSquaredError;
import model.HiddenLayer;
import model.NeuralNetwork;
import model.OutputLayer;
import model.NetworkController;

public class Main {
	
	public static void main(String[] args){

		// Load data
		List<float[]> traindata = DataLoader.loadData("C:\\Users\\user\\Documents\\AI Projects\\JRNN\\traindata.txt", Seperator.COMMA);
		
		// Create neural network
		NeuralNetwork nn = new NeuralNetwork(0.3f, 5E-3f, 3000000, true, ErrorFunction.MSE);
		nn.addLayer(new HiddenLayer(2, ActivationFunction.SIGMOID));
//		nn.addLayer(new HiddenLayer(1, ActivationFunction.LEAKED_RELU));
//		nn.addLayer(new HiddenLayer(1, ActivationFunction.SIGMOID));
		nn.addLayer(new OutputLayer(3, ActivationFunction.SIGMOID)); // Neuron count must be same with the input vector length
	
		// Train and test
		NetworkController nc = new NetworkController(nn);
		nc.showIterations(20000);
		nc.trainNetwork(traindata);	
		
		// Do regression for 10 steps, starting from the last input of train data
		nc.doRegression(10);
		
		// OR
		
		// Predict next 6 output for the given input sequence
//		List<float[]> inputSequence = new ArrayList<>();
//		inputSequence.add(new float[]{0, 0, 1});
//		inputSequence.add(new float[]{0, 1, 0});
//		inputSequence.add(new float[]{0, 1, 1});
//		inputSequence.add(new float[]{0, 0, 1});
//		inputSequence.add(new float[]{0, 1, 0});
//		inputSequence.add(new float[]{0, 1, 1});
//		
//		nc.predictNext(inputSequence, 6); 
		
	}

}
