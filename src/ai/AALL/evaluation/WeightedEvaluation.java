package ai.AALL.evaluation;

import ai.AALL.math.TensorMath;
import ai.evaluation.EvaluationFunction;

import org.ejml.simple.SimpleMatrix;

import rts.GameState;

public class WeightedEvaluation extends EvaluationFunction {
    protected float[] weights;

    public void setWeight(float[] _weights)
    {
        weights = _weights;
    }

    @Override
    public float upperBound(GameState gs) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'upperBound'");
    }

    @Override
    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'evaluate'");
    }
}
