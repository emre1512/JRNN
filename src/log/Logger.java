package log;

import java.util.List;

public class Logger implements LogMessages{

	private static Logger instance;
	
	public static Logger getInstance(){
		if(instance == null){
			instance = new Logger();
			return instance;
		}else{
			return instance;
		}
	}
	
	@Override
	public void showNoOutputLayerError() {
		System.out.println("No output layer! Please add an output layer.");
		System.out.println();
	}

//	@Override
//	public void showInitMessage(int hiddenLayerCount, int classCount, float learningRate) {
//		System.out.println("Network has been initiated with:" + hiddenLayerCount + " hidden layers, " +
//							);
//		System.out.println("                                ");
//		
//	}

	@Override
	public void showTrainingStartMessage(int epoch, float maxError, float learningRate) {
		System.out.println("======= Training Starts =======");
		System.out.println("Max Epochs        : " + epoch);
		System.out.println("Max Error         : " + maxError);
		System.out.println("Learning Rate     : " + learningRate);
		System.out.println("===============================");	
	}

	@Override
	public void showTrainingEndMessage(int iteration, float error) {
		System.out.println();
		System.out.println("======== Training Ends ========");
		System.out.println("Epochs : " + iteration);
		System.out.println("Error  : " + error);
		System.out.println("===============================");
		System.out.println();	
	}

	@Override
	public void showIterationStats(int iteration, float error) {
		System.out.println();
		System.out.println("===============================");
        System.out.println("Current iteration :" + iteration);
        System.out.println("Current error     :" + error);
		System.out.println("===============================");
	}

	@Override
	public void showRegressionResults(List<float[]> predictions) {
		System.out.println("======= Regression Starts =======");	
		
		for(int i = 0; i < predictions.size(); i++){
			for(int j = 0; j < predictions.get(0).length; j++){
				System.out.print(predictions.get(i)[j] + "	");
			}
			System.out.println();
		}
		
		System.out.println("====== Regression Finished ======");
		System.out.println();		
	}

	@Override
	public void showPredictionResults(List<float[]> predictions) {
		System.out.println("======= Prediction Starts =======");	
		
		for(int i = 0; i < predictions.size(); i++){
			for(int j = 0; j < predictions.get(0).length; j++){
				System.out.print(predictions.get(i)[j] + "	");
			}
			System.out.println();
		}		
		
		System.out.println("====== Prediction Finished ======");
		System.out.println();	
	}

	@Override
	public void showNoHiddenLayerError() {
		System.out.println("No hidden layer! Please add at least one hidden layer.");
		System.out.println();		
	}


}
