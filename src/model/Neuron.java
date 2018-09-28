package model;

import constant.Constants;
import math.IActivation;
import util.RandomGenerator;

public class Neuron {

	private float[] weights;
    private float bias;
    private float[] recurrentWeights;
    private float neuronOutput = 0;
    private float activationOutput = 0;
    private IActivation activationFunction;
    private float delta;
    public float error;
    
    public Neuron(IActivation activationFunction, int weightCount, int lastHiddenlayerNeuronCount) {
        this.weights = new float[weightCount];
        this.activationFunction = activationFunction;
        this.recurrentWeights = new  float[lastHiddenlayerNeuronCount];
        
        this.delta = 0;
         
        initializeWeights();
        initializeBias();
    }
    
    public Neuron(IActivation activationFunction, int weightCount) {
        this.weights = new float[weightCount];
        this.activationFunction = activationFunction;
        
        this.delta = 0;
         
        initializeWeights();
        initializeBias();
    }
    
    public float getWeight(int index) {
        return weights[index];
    }
    
    public float getRecurrentWeight(int index) {
        return recurrentWeights[index];
    }
    
    public float getWeightCount(){
    	return weights.length;
    }
    
    public float getRecurrentWeightCount(){
    	return recurrentWeights.length;
    }
    
    public void setWeight(int index, float weight) {
        weights[index] = weight;     
    }
            
    public float getNeuronOutput() {
		return neuronOutput;
	}

	public void setNeuronOutput(float neuronOutput) {
		this.neuronOutput = neuronOutput;
	}

	public float getActivationOutput() {
		return activationOutput;
	}

	public void setActivationOutput(float totalInput) {
		this.activationOutput = activationFunction.activate(totalInput);
	}

	public float getActivationDerivative(float input){
		return activationFunction.derivative(input);
	}
	
	public float getDelta() {
        return delta;
    }
	
	public void setDelta(float delta) {
        this.delta = delta;
    }
	     
    public float getBias() {
		return bias;
	}

	public void setBias(float bias) {
		this.bias = bias;
	}

	private void initializeWeights() {    	  
    	for (int i = 0; i < weights.length; i++) {  		    		
    		float value = RandomGenerator.generateRandom(Constants.MIN_WEIGHT, 
    									Constants.MAX_WEIGHT) / Constants.WEIGHT_NORMALIZER;
            this.weights[i] = value;
        }  
    	
    	for (int i = 0; i < recurrentWeights.length; i++) {  		    		
        	recurrentWeights[i] = RandomGenerator.generateRandom(Constants.MIN_WEIGHT, 
    				Constants.MAX_WEIGHT) / Constants.WEIGHT_NORMALIZER;
        }    	
    }
    
    private void initializeBias() {     
        this.bias = RandomGenerator.generateRandom(Constants.MIN_BIAS, 
        									Constants.MAX_BIAS) / Constants.BIAS_NORMALIZER;
    }
	
}
