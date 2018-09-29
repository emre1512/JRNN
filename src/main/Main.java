package main;

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
		NeuralNetwork nn = new NeuralNetwork(0.3f, 1E-3f, 3000000, false, ErrorFunction.MSE);
		nn.addLayer(new HiddenLayer(2, ActivationFunction.TANH));
		nn.addLayer(new OutputLayer(3, ActivationFunction.SIGMOID)); // Neuron count must be same with the input vector length
	
		// Train and test
		NetworkController nc = new NetworkController(nn);
		nc.showIterations(20000);
		nc.trainNetwork(traindata);	
		
		// Do regression for 6 steps, starting from the last input
		nc.doRegression(6);
		
		// Predict next 6 output with the given input
		nc.predictNext(new float[]{0, 0, 1}, 6); 
		

	}

}
