package model;

import java.util.ArrayList;
import java.util.List;
import constant.Constants;
import log.Logger;
import math.IError;
import math.MeanSquaredError;
import math.IError.ErrorFunction;

public class NeuralNetwork {

	private List<Layer> layers;
	private int epoch;
	private float[] inputs;
	private float[] desiredOutput;
    private float nu;
    private float desiredError;
    private boolean hasOutputLayer = false;
    private boolean hasHiddenLayer = false;
    private float globalError;
    private IError errorFunction;
    private boolean isBinary = false;

    
    public NeuralNetwork(float nu, float desiredError, int epoch, boolean isBinary,
    					ErrorFunction errorFunction){
        this.nu = nu;
        this.desiredError = desiredError;       
        this.epoch = epoch;
        this.layers = new ArrayList<>();
        this.isBinary = isBinary;

        if(errorFunction == ErrorFunction.MSE){
            this.errorFunction = new MeanSquaredError();
        }

    }
    
	public List<Layer> getLayers() {
		return layers;
	}

	public int getEpoch() {
		return epoch;
	} 

	public void setInputs(float[] inputs){
		this.inputs = inputs;
	}
	
	public float[] getDesiredOutput(){
		return this.desiredOutput;
	}
	
	public float getLearningRate(){
		return this.nu;
	}
	
	public void setDesiredOutput(float[] desiredOutput){
		this.desiredOutput = desiredOutput;
	}
	
	public void setErrorFunction(IError errorFunction){
		this.errorFunction = errorFunction;
	}

	public float getDesiredError(){
		return this.desiredError;
	}
	    
    public void addLayer(Layer layer){
    	if(layer.getClass().getSimpleName().equals(Constants.OUTPUT_LAYER_NAME)){
    		hasOutputLayer = true;
    		layers.add(layer);
    	}
    	else if(!hasOutputLayer){
        	layers.add(layer);
        	hasHiddenLayer = true;
    	}
    }
           
    public void train() {
    	if(!hasOutputLayer){
    		Logger.getInstance().showNoOutputLayerError();
    	}
    	else if(!hasHiddenLayer){
    		Logger.getInstance().showNoHiddenLayerError();
    	}
    	else{   
            feedForward();            
            calculateError();
            backPropagation();
    	}
    }
    
    public float[] predictNext() { 	
		feedForward();

        int outputLayerIndex = layers.size() - 1;
        int outputLayerNeuronCount = layers.get(outputLayerIndex).getNeuronCount();
        
        float[] output = new float[outputLayerNeuronCount];
        
        for(int i = 0; i < outputLayerNeuronCount; i++){
        	output[i] = layers.get(outputLayerIndex).getNeurons().get(i).getActivationOutput();
        }
        
        return output; 	
    }
    
    private void feedForward(){
    	
    	int layerCount = layers.size();
    	int firstHiddenLayerIndex = 0;
    	int lastHiddenLayerIndex = layerCount - 2;
    	
    	for(int i = 0; i < layerCount; i++){
    		
    		Layer currentLayer = layers.get(i);
    		   			
			for(int j = 0; j < currentLayer.getNeuronCount(); j++){
	    		
	    		Neuron currentNeuron = currentLayer.getNeurons().get(j);   
	    		
	    		float totalInput = 0;
	    		
	    		// Calculate feedforward input
	    		for(int k = 0; k < currentNeuron.getWeightCount(); k++){
	    			if(i == firstHiddenLayerIndex){
	    				totalInput += inputs[k] * currentNeuron.getWeight(k);
	    			}
	    			else{
	    				Neuron previousNeuron = layers.get(i-1).getNeurons().get(k);
	    				totalInput += previousNeuron.getActivationOutput() * currentNeuron.getWeight(k);
	    			}
	    		}	
	    		
	    		// Calculate recurrent input
	    		if(i == firstHiddenLayerIndex){
	    			Layer lastHiddenLayer = layers.get(lastHiddenLayerIndex);
	    			for(int k = 0; k < currentNeuron.getRecurrentWeightCount(); k++){
//		    			System.out.println(currentNeuron.getRecurrentWeight(k) );
		    			totalInput += lastHiddenLayer.getNeurons().get(k).getActivationOutput() 
		    												* currentNeuron.getRecurrentWeight(k);	    			
		    		}	
	    		}
	    		
	    		// Add bias
	    		totalInput += currentNeuron.getBias();
			
	    		currentNeuron.setNeuronOutput(totalInput);
	    		currentNeuron.setActivationOutput(totalInput);
	    		
			}									
    	}   	
    }
      
    private void calculateError() {
        
    	resetGlobalError();
        
    	int outputLayerIndex = layers.size() - 1;

        globalError = errorFunction.error(desiredOutput, layers.get(outputLayerIndex).getNeurons());
//    	System.out.println(desiredOutput[0] + " - " + layers.get(outputLayerIndex).getNeurons().get(0).getActivationOutput());
    }
    
    private void backPropagation() {
    	
    	int outputLayerIndex = layers.size() - 1;
    	int firstHiddenLayerIndex = 0;
    	int lastHiddenLayerIndex = outputLayerIndex - 1;
    	
    	for(int i = outputLayerIndex; i > -1; i--){

    		Layer currentLayer = layers.get(i);
    		
    		for(int j = 0; j < currentLayer.getNeuronCount(); j++){
    			
    			Neuron currentNeuron = currentLayer.getNeurons().get(j);  
    			float delta = 0;
    			
    			if(i == outputLayerIndex){   		
                    float derivativeOfError = errorFunction.derivative(currentNeuron.getActivationOutput(), desiredOutput[j], layers.get(i).getNeuronCount());
                    delta = derivativeOfError * currentNeuron.getActivationDerivative(currentNeuron.getActivationOutput());                
    			}			
    			else{
    				// Previous neuron is output layer for hidden layer!
        			Layer previousLayer = layers.get(i+1);  

    				for(int k = 0; k < previousLayer.getNeuronCount(); k++){
    					Neuron previousNeuron = previousLayer.getNeurons().get(k);
    	        		delta += previousNeuron.getDelta() * previousNeuron.getWeight(j) * 
    	        				currentNeuron.getActivationDerivative(currentNeuron.getActivationOutput());     
    				}
    			}

    			currentNeuron.setDelta(delta);   			
    			
    			if(i == firstHiddenLayerIndex){
    				
    				// BP for Normal input weights
    				for (int k = 0; k < inputs.length; k++) {
                		float weightDiff = nu * currentNeuron.getDelta() * inputs[k];
                        float biasDiff = nu * currentNeuron.getDelta();
                        currentNeuron.setWeight(k, currentNeuron.getWeight(k) - weightDiff);
                        currentNeuron.setBias(currentNeuron.getBias() - biasDiff);
                    }
    				
    				// BP for recurrent input weights
    				for (int k = 0; k < currentNeuron.getRecurrentWeightCount(); k++) {
                		float weightDiff = nu * currentNeuron.getDelta() 
                					* layers.get(lastHiddenLayerIndex).getNeurons().get(k).getActivationOutput();
                        currentNeuron.setRecurrentWeight(k, currentNeuron.getRecurrentWeight(k) - weightDiff);
                    }
    			}
    			else{
    				// Previous neuron is hidden layer for output layer!
    				Layer previousLayer = layers.get(i-1); 
        			for (int k = 0; k < previousLayer.getNeurons().size(); k++) {
        				
        				Neuron previousNeuron = previousLayer.getNeurons().get(k);
        				
                		float weightDiff = nu * currentNeuron.getDelta() * previousNeuron.getActivationOutput();
                        float biasDiff = nu * currentNeuron.getDelta();

                        currentNeuron.setWeight(k, currentNeuron.getWeight(k) - weightDiff);
                        currentNeuron.setBias(currentNeuron.getBias() - biasDiff);
                    }
    			}
		
    		}
    			
    	}   	   
        
    }
    	
    public boolean isBinary() {
		return isBinary;
	}
    
    public float getGlobalError() {
        return this.globalError;
    }
    
    public void resetGlobalError() {
        this.globalError = 0;
    }
     
    public boolean hasLearnt() {
        return (globalError <= desiredError);
    }

}
