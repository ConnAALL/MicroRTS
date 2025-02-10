package ai.AALL.evaluation;

import ai.evaluation.EvaluationFunction;
import rts.GameState;
public class SSVDEvaluation extends WeightedEvaluation{

    @Override
    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        // TODO Auto-generated method stub
        int[][][] obs = gs.getVectorObservation(minplayer);
        //use ND4J!

        throw new UnsupportedOperationException("Unimplemented method 'evaluate'");
    }

    @Override
    public float upperBound(GameState gs) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'upperBound'");
    }
    
}
