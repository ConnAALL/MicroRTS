package ai.AALL;

import java.util.List;

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

    
    @Override
    public PlayerAction getAction(final int player, final GameState gs, final int[][] action) {
        //Input a softmaxed tensor
        // output consists of 
        // output unit type selection (size of unit type count)
        // output tile position (size of board)^2
        // top unittype selection and top 4 tile position selection are chosen
        PhysicalGameState pgs = gs.getPhysicalGameState();
        
        int[] inputINT = NetworkHelpers.getFlattened(pgs, player);
        double[] input = new double[inputINT.length];
        for (int i = 0; i < inputINT.length; i++) {
            input[i] = inputINT[i]; // Cast int to double
        }
        for (Unit u : pgs.getUnits()) { 
            if (u.getPlayer() == player) {//for each friendly unit, think
                if (gs.getActionAssignment(u) == null) {

                    // u.getX() + u.getY() * pgs.getHeight() = action.output_tile
                    int x = action.output_tile % pgs.getHeight();
                    int y = (action.output_tile - x) / pgs.getHeight();

                    List<UnitType> types = unitTable.getUnitTypes();
                    if(action.output_type != types.size()) // check if nullop
                    {
                        UnitType type = types.get(action.output_type);
                        unitAction(player, gs, u, x, y, type);
                    }
                }
            }
        }
        return translateActions(player, gs);
    }
    
    public void unitAction(int playerID, GameState gs, Unit selectedUnit, int x, int y, UnitType trainType) {
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
        }
        else if (unitType.canHarvest) //harvester is builder
        {
            if (rawUnit.getType().isResource) {
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
                }
            }
            else if(rawUnit.getPlayer() != playerID && selectedUnit.getType().canAttack) //check if the target tile has enemy
            {
                attack(selectedUnit, rawUnit); //make sure worker can attack   
            }
            else if (!trainType.canMove)
            {
                build(selectedUnit, trainType, x, y);
            }
            else if (rawUnit == null || rawUnit.getPlayer() == playerID) { // empty 
                move(selectedUnit, x, y);
            }
        }
        else //unit can move and can not harvest
        {
            if(rawUnit == null || rawUnit.getPlayer() == playerID)
            {
                move(selectedUnit, x, y);
            }
            else if (selectedUnit.getType().canAttack)
            {
                attack(selectedUnit, rawUnit);
            }
        }
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
