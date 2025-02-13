package ai.AALL.evaluation;

import ai.AALL.math.TensorMath;
import ai.evaluation.EvaluationFunction;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import rts.GameState;

public abstract class WeightedEvaluation extends EvaluationFunction {
    protected float[] weights;
    public abstract void parseWeights();
    public void setWeight(float[] _weights)
    {
        //System.out.println("WeightedEvaluation::setWeight");
        weights = _weights;
        double[] doubleweights = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            doubleweights[i] = (double) weights[i];
        }
        parseWeights();
    }
    
    
}
