package math;

public class ReluActivation implements IActivation{

	@Override
	public float activate(float input) {
		return Math.max(0, input);
	}

	@Override
	public float derivative(float input) {
		if (input <= 0) return 0f;
		else return 1.0f;
	}

}
