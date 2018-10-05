package log;

import java.util.List;

public interface LogMessages {

	public void showNoOutputLayerError();
	public void showNoHiddenLayerError();
	public void showTrainingStartMessage(int epoch, float maxError, float learningRate);
	public void showTrainingEndMessage(int iteration, float error);
	public void showIterationStats(int iteration, float error);
	public void showRegressionResults(List<float[]> predictions);
	public void showPredictionResults(List<float[]> predictions);
	public void showOutputNeuronCountMustBeSameWithInputVectorLengthError(int outputNeuronCount, int inputVectorLength);
	
}
