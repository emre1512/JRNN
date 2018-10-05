package model;

import java.util.ArrayList;
import java.util.List;
import constant.Constants;
import log.Logger;
import util.ResultQuantizer;

public class NetworkController {

	private NeuralNetwork nn;
	private int epoch;
	private int iteration;
	private List<float[]> trainDataset;
	private int iterationLogStepCount = Constants.ITERATION_LOG_STEP_COUNT;
	
	public NetworkController(NeuralNetwork nn){
		this.nn = nn;
		this.epoch = nn.getEpoch();
	}
	
	public void trainNetwork(List<float[]> trainDataset){
		
		this.trainDataset = trainDataset;
		
		int sampleCount = trainDataset.size();
		int inputCount = trainDataset.get(0).length; // Because don't have class label
		int outputLayerIndex = nn.getLayers().size() - 1;
		int outputLayerNeuronCount = nn.getLayers().get(outputLayerIndex).getNeuronCount();
		
		try{
			initNeurons(inputCount);
			
			Logger.getInstance().showTrainingStartMessage(epoch, nn.getDesiredError(), nn.getLearningRate());
			
			int iter = 0;
			
			do {

	        	iter++;
	        	
	        	// Not using last one
	        	for(int i = 0; i < sampleCount - 1; i++){
	        		
	        		float[] sample = trainDataset.get(i);
	        		float[] nextSample = trainDataset.get(i + 1);
	        		
	        		float[] inputs = new float[inputCount];
	        		
	        		for(int j = 0; j < inputCount; j++){            		
	            		inputs[j] = sample[j];
	        		}
	        		
	        		float[] output = new float[inputCount]; 

	        		for(int j = 0; j < inputCount; j++){            		
	        			output[j] = nextSample[j];
	        		}
	        		     		
	        		nn.setInputs(inputs);
	        		
	        		nn.setDesiredOutput(output);
	        		
	        		nn.train();
	        	}
	        	        	
	            if (iterationLogStepCount != Constants.ITERATION_LOG_STEP_COUNT 
	            		&& iter % iterationLogStepCount == 0) {
	                Logger.getInstance().showIterationStats(iter, nn.getGlobalError());
	            }
	        	
			} while (!nn.hasLearnt() && iter < epoch);
			
			this.iteration = iter;
			
			Logger.getInstance().showTrainingEndMessage(iteration, nn.getGlobalError());
			
		}catch (ArrayIndexOutOfBoundsException e) {
			
			Logger.getInstance().
				showOutputNeuronCountMustBeSameWithInputVectorLengthError(
				outputLayerNeuronCount, inputCount);
			
			System.exit(1);
		}
				
	}
		
	public void doRegression(int stepCount){
		
		List<float[]> predictions = new ArrayList<>();
		
		int outputLayerIndex = nn.getLayers().size() - 1;
		int outputLayerNeuronCount = nn.getLayers().get(outputLayerIndex).getNeuronCount();
		
		float[] nextInput = trainDataset.get(trainDataset.size()-1);
		float[] predictedOutput = new float[outputLayerNeuronCount];
		
		try{
					
			for(int i = 0; i < stepCount; i++){						
				nn.setInputs(nextInput);
				predictedOutput = nn.predictNext();

				predictions.add(predictedOutput);
				
				nextInput = predictedOutput;
			}	
			
			if(nn.isBinary()){
				for(int i = 0; i < predictions.size(); i++){
					for(int j = 0; j < predictedOutput.length; j++){
						predictions.get(i)[j] = ResultQuantizer.quantizeResult(predictions.get(i)[j]);
					}
				}
			}
					
			Logger.getInstance().showRegressionResults(predictions);
			
		}catch (ArrayIndexOutOfBoundsException  e) {
			
			Logger.getInstance().
				showOutputNeuronCountMustBeSameWithInputVectorLengthError(
					outputLayerNeuronCount, trainDataset.get(0).length);
		}

		
	}
	
	public void predictNext(List<float[]> inputSequence, int stepCount){
		
		List<float[]> predictions = new ArrayList<>();
		
		int outputLayerIndex = nn.getLayers().size() - 1;
		int outputLayerNeuronCount = nn.getLayers().get(outputLayerIndex).getNeuronCount();
		
		float[] predictedOutput = new float[outputLayerNeuronCount];
		
		// Use input sequence
		for(int i = 0; i < inputSequence.size(); i++){
			nn.setInputs(inputSequence.get(i));
			predictedOutput = nn.predictNext(); // We ignore the predictions, only the last prediction is important.
		}
		
		predictions.add(predictedOutput);	
				
		float[] nextInput = new float[outputLayerNeuronCount];
		
		for(int i = 0; i < stepCount - 1; i++){						
			nn.setInputs(nextInput);
			predictedOutput = nn.predictNext();
					
			predictions.add(predictedOutput);
			
			nextInput = predictedOutput;
		}
		
		if(nn.isBinary()){
			if(nn.isBinary()){
				for(int i = 0; i < predictions.size(); i++){
					for(int j = 0; j < predictedOutput.length; j++){
						predictions.get(i)[j] = ResultQuantizer.quantizeResult(predictions.get(i)[j]);
					}
				}
			}
		}
		
		Logger.getInstance().showPredictionResults(predictions);
				
	}
	
//	public void predictNext(float[] input, int stepCount){
//		
//		// TODO: Prediction start message log
//		
//		List<float[]> predictions = new ArrayList<>();
//		
//		int outputLayerIndex = nn.getLayers().size() - 1;
//		int outputLayerNeuronCount = nn.getLayers().get(outputLayerIndex).getNeuronCount();
//		
//		float[] predictedOutput = new float[outputLayerNeuronCount];		
//		float[] nextInput = input;		
//		
//		for(int i = 0; i < stepCount; i++){						
//			nn.setInputs(nextInput);
//			predictedOutput = nn.predictNext();
//						
//			predictions.add(predictedOutput);
//			
//			nextInput = predictedOutput;
//		}
//		
//		if(nn.isBinary()){
//			if(nn.isBinary()){
//				for(int i = 0; i < predictions.size(); i++){
//					for(int j = 0; j < predictedOutput.length; j++){
//						predictions.get(i)[j] = ResultQuantizer.quantizeResult(predictions.get(i)[j]);
//					}
//				}
//			}
//		}
//		
//		for(int i = 0; i < predictions.size(); i++){
//			for(int j = 0; j < predictions.get(0).length; j++){
//				System.out.print(predictions.get(i)[j] + " - ");
//			}
//			System.out.println();
//		}
//		
//	}
	
	public void showIterations(int iterationLogStepCount){
		this.iterationLogStepCount = iterationLogStepCount;
	}
	
	public void initNeurons(int inputCount){

		int firstHiddenLayerIndex = 0;
		int lastHiddenLayerIndex = nn.getLayers().size() - 2;
		
		for(int i = 0; i < nn.getLayers().size(); i++){
		
			Layer layer = nn.getLayers().get(i);
			
			
			if(i == firstHiddenLayerIndex){
				for(int j = 0; j < layer.getNeuronCount(); j++){
					int lastHiddenLayerNeuronCount = nn.getLayers().get(lastHiddenLayerIndex).getNeuronCount();
					Neuron neuron = new Neuron(layer.getActivationFunction(), inputCount, lastHiddenLayerNeuronCount);				
					layer.addNeuron(neuron);
				}
			}
			else{
				for(int j = 0; j < layer.getNeuronCount(); j++){
					int weightCount = nn.getLayers().get(i-1).getNeuronCount();
					Neuron neuron = new Neuron(layer.getActivationFunction(), weightCount);
					layer.addNeuron(neuron);
				}

			}						
		}
	}
}
