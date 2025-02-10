package ai.AALL.evaluation;
import ai.evaluation.EvaluationFunction;
import rts.GameState;

public class WeightedEvaluation extends EvaluationFunction {
    private float[] weights;
    @Override
    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'evaluate'");
    }

    @Override
    public float upperBound(GameState gs) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'upperBound'");
    }

    public void setWeight(float[] _weights)
    {
        weights = _weights;
    }
}
