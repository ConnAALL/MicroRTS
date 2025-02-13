package ai.AALL;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.lang.Math;
import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.evaluation.SimpleEvaluationFunction;
import ai.jni.JNIInterface;
import ai.units.NetworkHelpers;
import ai.units.NetworkOuput;
import ai.units.PerUnitAI;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.ResourceUsage;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;
public class JNIExpertAI extends AbstractionLayerAI implements JNIInterface{
    UnitTypeTable utt = null;
    double reward = 0.0;
    double oldReward = 0.0;
    boolean firstRewardCalculation = true;
    SimpleEvaluationFunction ef = new SimpleEvaluationFunction();
    int maxAttackRadius;

    public JNIExpertAI(int timeBudget, int iterationsBudget, UnitTypeTable a_utt) {
        super(new AStarPathFinding(), timeBudget, iterationsBudget);
        utt = a_utt;
        maxAttackRadius = utt.getMaxAttackRange() * 2 + 1;
    }

    @Override
    public double computeReward(final int maxplayer, final int minplayer, final GameState gs) throws Exception {
        // do something
        if (firstRewardCalculation) {
            oldReward = ef.evaluate(maxplayer, minplayer, gs);
            reward = 0;
            firstRewardCalculation = false;
        } else {
            double newReward = ef.evaluate(maxplayer, minplayer, gs);
            reward = newReward - oldReward;
            oldReward = newReward;
        }
        return reward;
    }

    private float[] MinMax(final int[] action)
    {
        float[] out = new float[action.length];
        int min = Integer.MAX_VALUE;
        for(int i=0;i<action.length;i++)
        {
            if (action[i] < min) {
                min = action[i];
            }
        }
        int max = Integer.MIN_VALUE;
        for(int i=0;i<action.length;i++)
        {
            if (action[i] > max) {
                max = action[i];
            }
        }
        int range = max - min;
        for(int i=0;i<action.length;i++)
        {
            out[i] = 2 * (action[i] - min) / (float) range - 1;
        }
        return out;
    }

    // Softmax a tensor
    // Does not affect action
    private float[] softmax(final int[] action) {
        float sum = 0;
        float[] scaled = MinMax(action);
        for (int i = 0; i < action.length; i++) {
            float n = scaled[i];
            float v = (float) Math.exp((double) n);
            //System.out.println(String.format("\tn v %f %f", n, v));
            sum += v;
            scaled[i] = v;
        }
        if (sum == 0)
        {
            throw new RuntimeException(String.format("Division by 0"));
        }
        for (int i = 0; i < scaled.length; i++) {
            scaled[i] = scaled[i] / sum;
        }
        return scaled;
    }

    private int multinomial(final float[] logits) throws RuntimeException
    {
        assert logits.length != 0 : "logits can not be 0";
        Random random = new Random();
        float r = random.nextFloat(); // Random number in [0, 1)
        float cumulativeSum = 0.0f;
        for (int i = 0; i < logits.length; i++) {
            cumulativeSum += logits[i];
            //System.out.println(String.format("\tAdding %f", logits[i]));
            if (r < cumulativeSum) {
                return i;
            }
        }
        //throw new RuntimeException(String.format("Sum is only %f", cumulativeSum));
        //System.out.println();
        return logits.length - 1; //probably floating point error...
    }
    
    @Override
    public PlayerAction getAction(final int player, final GameState gs, int[][] action) throws Exception {
        //System.out.printf("JNIExpertAI::getAction action.length %d%n", action.length);
        //System.out.printf("JNIExpertAI::getAction action[0].length %d%n", action[0].length);

        // PlayerAction pa = PlayerAction.fromVectorAction(action, gs, utt, player, maxAttackRadius);
        // pa.fillWithNones(gs, player, 1);
        // return pa;
        int[] flatAction = action[0];
        //Input a softmaxed tensor
        // output consists of 
        // output unit type selection (size of unit type count)
        // output tile position (size of board)^2
        // top unittype selection and top 4 tile position selection are chosen
        PhysicalGameState pgs = gs.getPhysicalGameState();

        List<UnitType> types = utt.getUnitTypes();
        int[] unitTypeAction = Arrays.copyOfRange(flatAction, 0, types.size());
        int[] tileAction = Arrays.copyOfRange(flatAction, types.size(), flatAction.length);
        if (unitTypeAction.length == types.size()) { // check if something wrong
            for (Unit u : pgs.getUnits()) {
                if (u.getPlayer() == player && gs.getActionAssignment(u) == null) {//for each friendly unit without action assigned
                    // Softmax to select tile policy
                    int pos = multinomial(softmax(tileAction));
                    int x = pos / pgs.getWidth();
                    int y = pos % pgs.getHeight();
                    int unitPos = u.getPosition(pgs);
                    int ux = unitPos / pgs.getWidth();
                    int uy = unitPos % pgs.getHeight();
                    
                    //if (Math.sqrt((float) ((ux - x) ^ 2 + (uy - y) ^ 2)) < maxAttackRadius) // Check if selected tile is close enough to the unit
                    
                    //System.out.printf("Unit at %d %d, acting on %d %d %n", ux, uy, x, y);
                    int ut = multinomial(softmax(unitTypeAction));
                    //System.out.printf("Evaluated Unit Type: %d %n", ut);
                    UnitType type = types.get(ut);
                    boolean shouldReduce = unitAction(player, gs, u, x, y, type);
                    if(shouldReduce)
                    {
                        flatAction[pos] *= 0.7f; //tile policy selected. Reduce it to reduce the probability of it getting reselected
                        // This is only called if the unit is movable unit
                    }
                    
                }
            }
        }
        PlayerAction pa =  translateActions(player, gs);
        pa.fillWithNones(gs, player, 1);
        return pa;
    }
    
    public boolean unitAction(int playerID, GameState gs, Unit selectedUnit, int x, int y, UnitType trainType) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit rawUnit = pgs.getUnitAt(x, y);
        UnitType unitType = selectedUnit.getType();
        //type of selectedunit? 
            //can train
                //can it train the given traintype?
                    //find spot to train
            //can harvest
                //is target position free
                    //move
                //is there resource at target position
                    //harvest
                //build
            //the rest
                //is target position free
                    //move
                //is there enemy unit in target position
                    //attack
                //is there friendly unit in target position
                    //move
        if (!unitType.canMove && trainType.canMove) { //building
            train(selectedUnit, trainType);
            return false;
        }
        else if (unitType.canHarvest) //harvester is builder
        {
            if (rawUnit == null || rawUnit.getPlayer() == playerID)
            {
                move(selectedUnit, x, y);
                return true;
            }
            else if (rawUnit.getType().isResource) {
                Unit base = null;
                double bestD = 0;
                for (Unit u : pgs.getUnits()) {
                    if (u.getPlayer() == playerID &&
                            u.getType().isStockpile) {
                        double d = (selectedUnit.getX() - u.getX()) + (selectedUnit.getY() - u.getY());
                        if (base == null || d < bestD) {
                            base = u;
                            bestD = d;
                        }
                    }
                }
                if (base != null) {
                    harvest(selectedUnit, rawUnit, base);
                    return true;
                }
            }
            else if(rawUnit.getPlayer() != playerID && selectedUnit.getType().canAttack) //check if the target tile has enemy
            {
                attack(selectedUnit, rawUnit); //make sure worker can attack
                return true; 
            }
            else if (!trainType.canMove)
            {
                build(selectedUnit, trainType, x, y);
                return true;
            }
        }
        else //unit can move and can not harvest
        {
            if (rawUnit == null || rawUnit.getPlayer() == playerID) {
                move(selectedUnit, x, y);
                return true;
            } else if (selectedUnit.getType().canAttack) {
                attack(selectedUnit, rawUnit);
                return true;
            }
        }
        return true;
    }

    @Override
	public int[][][] getObservation(final int player, final GameState gs) throws Exception {
        return gs.getVectorObservation(player);
    }

    @Override
    public void reset() {
        // TODO Auto-generated method stub
    }

    @Override
    public AI clone() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public String computeInfo(int player, GameState gs) throws Exception {
        // TODO Auto-generated method stub
        return null;
    }
}
