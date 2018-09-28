package model;

import java.util.List;
import constant.Constants;
import log.Logger;
import util.ArrayToClassLabel;
import util.ClassLabelToArray;
import util.SuccessCalculator;

public class NetworkController {

	private NeuralNetwork nn;
	private int epoch;
	private int iteration;
	private int iterationLogStepCount = Constants.ITERATION_LOG_STEP_COUNT;
	
	public NetworkController(NeuralNetwork nn){
		this.nn = nn;
		this.epoch = nn.getEpoch();
	}
	
	public void trainNetwork(List<float[]> trainDataset){
		
		int sampleCount = trainDataset.size();
		int inputCount = trainDataset.get(0).length; // Because don't have class label
		//int classLabelIndex = trainDataset.get(0).length - 1;
		
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
        		output[0] = nextSample[0];
        		
        		nn.setInputs(inputs);
        		
        		//int classLabel = (int)sample[classLabelIndex];
        		//nn.setDesiredOutput(ClassLabelToArray.convert(classLabel, nn.getLayers().get(nn.getLayers().size()-1).getNeuronCount()));
        		
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
	}
	
//	public void testNetwork(List<float[]> testDataset){
//      
//		Logger.getInstance().showTestStartMessage();
//	       
//		int sampleCount = testDataset.size();
//		int inputCount = testDataset.get(0).length - 1;
//		int classLabelIndex = testDataset.get(0).length - 1;		
//		int outputLayerIndex = nn.getLayers().size() - 1;
//		
//		float[] nnOutput = new float[nn.getLayers().get(outputLayerIndex).getNeuronCount()];
//		
//	    for(int i = 0; i < sampleCount; i++){
//    		
//    		float[] sample = testDataset.get(i);
//    		
//    		float[] inputs = new float[inputCount];
//    		
//    		for(int j = 0; j < inputCount; j++){            		
//        		inputs[j] = sample[j];
//    		}
//    		
//    		nn.setInputs(inputs);
//    		
//    		int classLabel = (int)sample[classLabelIndex];
//    		nn.setDesiredOutput(ClassLabelToArray.convert(classLabel, nn.getLayers().get(nn.getLayers().size()-1).getNeuronCount()));
//    		
//    		nn.test();    	   		    		
//    		
//    		
//    		for(int j = 0; j < nn.getLayers().get(outputLayerIndex).getNeuronCount(); j++){
//    			nnOutput[j] = nn.getLayers().get(outputLayerIndex).getNeurons().get(j).getActivationOutput();
//    		}
//    		
//			int predictedClass = ArrayToClassLabel.convert(nnOutput);
//			int realClass = (int)sample[classLabelIndex];
//    		
//			SuccessCalculator.addResult(predictedClass, realClass);
//			
//    		Logger.getInstance().showTestCaseResult(inputs, predictedClass, realClass);
//			
//    	}
//	    	
//	    float successRate = SuccessCalculator.getSuccessRate(sampleCount);
//	    
//	    Logger.getInstance().showTestEndMessage(successRate);
//	    
//	}  
	
	public void doRegression(int stepCount){
		
		int outputLayerIndex = nn.getLayers().size() - 1;
		int outputLayerNeuronCount = nn.getLayers().get(outputLayerIndex).getNeuronCount();
		
		float[] prevOutput = new float[outputLayerNeuronCount];
		float[] nextOutput = new float[outputLayerNeuronCount];
		
		for(int i = 0; i < stepCount; i++){			
			for(int j = 0; j < outputLayerNeuronCount; j++){
				prevOutput[j] = nn.getLayers().get(outputLayerIndex).getNeurons().get(j).getActivationOutput();
				nn.setInputs(prevOutput);
				nextOutput = nn.predictNext();
				
				for(int k = 0; k < outputLayerNeuronCount; k++){
					System.out.print(nextOutput[k] + " - ");
				}

			}
		}
	
	}
	
//	public void predictNext(float[] input){
//		nn.setInputs(vector);
//	}
//	
//	public void predictNext(int[] vector){
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
					System.out.println(lastHiddenLayerNeuronCount);
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
