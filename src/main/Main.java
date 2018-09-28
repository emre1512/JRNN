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
		List<float[]> traindata = DataLoader.loadData("C:\\Users\\user\\Documents\\AI Projects\\JRNN\\traindata.txt", Seperator.TAB);
		
		// Create neural network
		NeuralNetwork nn = new NeuralNetwork(0.1f, 5E-3f, 5, false, ErrorFunction.MSE);
		nn.addLayer(new HiddenLayer(1, ActivationFunction.LINEAR));
		nn.addLayer(new HiddenLayer(1, ActivationFunction.SIGMOID));
		nn.addLayer(new OutputLayer(1, ActivationFunction.LINEAR));
	
		// Train and test
		NetworkController nc = new NetworkController(nn);
		nc.showIterations(5000);
		nc.trainNetwork(traindata);	
	//	nc.testNetwork(testdata);

		
		nc.doRegression(5);
		
//		for(int i = 0; i < 10; i++){
//			nc.predictNext(new int[]{1});	
//		}

	}

}
