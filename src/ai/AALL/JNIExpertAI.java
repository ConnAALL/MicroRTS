package ai.AALL;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
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
    protected HashMap<Unit, Boolean> workerTable;
    protected HashMap<Unit, Boolean> attackerTable;
    boolean useSimple;

    public JNIExpertAI(int timeBudget, int iterationsBudget, UnitTypeTable a_utt, boolean _useSimple) {
        super(new AStarPathFinding(), timeBudget, iterationsBudget);
        utt = a_utt;
        maxAttackRadius = utt.getMaxAttackRange() * 2 + 1;
        useSimple = _useSimple;
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
            //Division by 0
            return scaled;
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

    // old model. input size 16x16 and output size 16x16+6
    private PlayerAction getActionDetailed(final int player, final GameState gs, int[][] action)
    {
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
                    if (shouldReduce) {
                        flatAction[pos] *= 0.7f; //tile policy selected. Reduce it to reduce the probability of it getting reselected
                        // This is only called if the unit is movable unit
                    }

                }
            }
        }
        PlayerAction pa = translateActions(player, gs);
        pa.fillWithNones(gs, player, 1);
        return pa;
    }

    private PlayerAction getActionSimple(final int player, final GameState gs, int[][] action)
    {
        //System.out.println("Calculating Simple Action");
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int[] flatAction = null;
        int agentAction = 0;
        List<Map.Entry<Unit, Boolean>> unitList;
        int[] coords;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                registerNewUnit(u);
            }
        }
        try {
            flatAction = action[0];
            assert action.length == 1 : "Model action vector height must be 1";
            assert flatAction.length == 13 : "Model action vector does not match action count";
            agentAction = multinomial(softmax(flatAction));
            //System.out.println(String.format("Agent action is %d", agentAction));
        } catch(Exception e)
        {
            System.out.println("Error while parsing model output: " + e.getMessage());
        }
        try{
            // These actions override previous action unless it's a building action
            switch (agentAction) {
                case 0: // do nothing
                    break;
                case 1: // allocate 1 worker unit to resource gathering. The worker set to resource gathering does not take attack orders
                    unitList = new ArrayList<>(workerTable.entrySet());
                    Collections.shuffle(unitList); //random unit
                    for (Map.Entry<Unit, Boolean> entry : unitList) {
                        Unit u = entry.getKey();
                        if (entry.getValue() == false) {
                            registerToTables(u, true);
                            break;
                        }
                    }
                    break; // no units can be set as worker
                case 2: // deallocate all workers from resource gathering. (Worker pull)
                    for (Unit entry : workerTable.keySet()) {
                        registerToTables(entry, false);
                    }
                    break;
                // attack commands sends all units that arent resource gathers to a random position within the specified quadrant
                case 3: // attack quadrant 1 -> on default map this is the base territory
                    unitList = new ArrayList<>(attackerTable.entrySet());
                    attackersToQuad(1, pgs, unitList);
                    break;
                case 4: // attack quadrant 2 
                    unitList = new ArrayList<>(attackerTable.entrySet());
                    attackersToQuad(2, pgs, unitList);
                    break;
                case 5: // attack quadrant 3
                    unitList = new ArrayList<>(attackerTable.entrySet());
                    attackersToQuad(3, pgs, unitList);
                    break;
                case 6: // attack quadrant 4 -> on default map this is the enemy base
                    unitList = new ArrayList<>(attackerTable.entrySet());
                    attackersToQuad(4, pgs, unitList);
                    break;
                case 7: // build worker from random base
                    trainUnit(player, "Worker", pgs);
                    break;
                case 8: // build light from random barrack
                    trainUnit(player, "Light", pgs);
                    break;
                case 9: // build heavy from random barrack
                    trainUnit(player, "Heavy", pgs);
                    break;
                case 10: // build ranged from random barrack
                    trainUnit(player, "Ranged", pgs);
                    break;
                case 11: // expand to nearest base that is (1. further than 5 tiles from current base) (2. has resource within 3 tiles)
                    coords = findExpansionLocation(pgs, 5);
                    if (coords != null)
                    {
                        for(Unit u : workerTable.keySet())
                        {
                            UnitType ut = utt.getUnitType("Base");
                            if(u.getType().produces.contains(ut) && !isBuilding(u,gs)) // make sure the unit is not already building something
                            {
                                build(u, ut, coords[0], coords[1]);
                            }
                        }
                    }
                    break;
                case 12: // build barrack near a random base (1. within 5 tiles of an existing base) (2. at least 3 tiles away from existing resources)
                    coords = findBarrackLocation(pgs, 5);
                    if (coords != null)
                    {
                        for(Unit u : workerTable.keySet())
                        {
                            UnitType ut = utt.getUnitType("Barracks");
                            if(u.getType().produces.contains(ut) && !isBuilding(u,gs)) // make sure the unit is not already building something
                            {
                                build(u, ut, coords[0], coords[1]); 
                            }
                        }
                    }
                    break;
            }
        } catch(Exception e)
        {
            System.out.println("Error while parsing model output: " + e.getMessage());
        }
        try{
            //Auto Actions
            //Attackers auto attacking around them
            for(Unit u : attackerTable.keySet())
            {
                if(gs.getActionAssignment(u) == null)
                {
                    int minRange = Math.max(2, u.getAttackRange());
                    Collection<Unit> inRange = pgs.getUnitsAround(u.getX(), u.getY(), minRange);
                    for (Unit uu : inRange) {
                        if (uu.getPlayer() != player) {
                            attack(u, uu);
                        }
                    }
                }
            }
            //Workers auto harvest from around them.
            for(Unit u : workerTable.keySet())
            {
                Boolean wb = workerTable.get(u);
                if(wb != null && wb && gs.getActionAssignment(u) == null)
                {
                    int minRange = Math.max(4,u.getAttackRange());
                    Collection<Unit> inRange = pgs.getUnitsAround(u.getX(), u.getY(), minRange);
                    Unit base = null;
                    for(Unit uu : inRange) // find close base
                    {
                        if (uu.getPlayer() == player && uu.getType().isStockpile)
                        {
                            base = uu;
                        }
                    }
                    if (base == null) //find any base
                    {
                        for (Unit b : pgs.getUnits()) {
                            if(b.getType().isStockpile && b.getPlayer() == player)
                            {
                                move(u, b.getX(), b.getY()); 
                            }
                        }
                    }
                    for(Unit uuu : inRange)
                    {
                        if(uuu.getType().isResource)
                        {
                            harvest(u, uuu, base);
                        }
                    }
                }
            }
        } catch(Exception e)
        {
            System.out.println("Error while handling automatic commands: " + e.getMessage());
        }
        PlayerAction pa = translateActions(player, gs);
        pa.fillWithNones(gs, player, 1);
        return pa;
    }
    // this function is called with all new units. All new units are by default attackers, even if they can harvest
    private void registerNewUnit(Unit u)
    {
        boolean canHarvest = u.getType().canHarvest;
        boolean canAttack = u.getType().canAttack;
        if (attackerTable.get(u) == null && canAttack) {
            attackerTable.put(u, true);
        }
        if (workerTable.get(u) == null && canHarvest) {
            workerTable.put(u, false);
        }
    }
    
    // by default, workers can both attack and harvest. 
    private void registerToTables(Unit u, boolean worker)
    {
        boolean canHarvest = u.getType().canHarvest;
        boolean canAttack = u.getType().canAttack;
        if (worker) {
            if (canHarvest) {workerTable.put(u, true);}
            if (canAttack) {attackerTable.put(u, false);}
        } else
        {
            if (canHarvest) {workerTable.put(u, false);}
            if (canAttack) {attackerTable.put(u, true);}
        }
    }

    private void trainUnit(final int player, final String UnitTypeName, PhysicalGameState pgs)
    {
        UnitType workerType = utt.getUnitType(UnitTypeName);
        Collections.shuffle(pgs.getUnits()); //random unit
        for (Unit u : pgs.getUnits()) {
            if (u.getType().produces.contains(workerType) && u.getPlayer() == player) {
                train(u, workerType);
                break;
            }
        }
    }

    private void attackersToQuad(int quad, PhysicalGameState pgs, List<Map.Entry<Unit, Boolean>> unitList)
    {
        int maxX = pgs.getWidth();
        int maxY = pgs.getHeight();
        switch (quad) {
            case 1:
                for (Map.Entry<Unit, Boolean> entry : unitList) {
                    if (entry.getValue() == true) {
                        int randomX = ThreadLocalRandom.current().nextInt(1, maxX / 2);
                        int randomY = ThreadLocalRandom.current().nextInt(1, maxY / 2);
                        move(entry.getKey(), randomX, randomY);
                    }
                }
                return;
            case 2:
                for (Map.Entry<Unit, Boolean> entry : unitList) {
                    if (entry.getValue() == true) {
                        int randomX = ThreadLocalRandom.current().nextInt(maxX / 2, maxX);
                        int randomY = ThreadLocalRandom.current().nextInt(1, maxY / 2);
                        move(entry.getKey(), randomX, randomY);
                    }
                }
                return;
            case 3:
                for (Map.Entry<Unit, Boolean> entry : unitList) {
                    if (entry.getValue() == true) {
                        int randomX = ThreadLocalRandom.current().nextInt(1, maxX / 2);
                        int randomY = ThreadLocalRandom.current().nextInt(maxY / 2, maxY);
                        move(entry.getKey(), randomX, randomY);
                    }
                }
                return;
            case 4:
                for (Map.Entry<Unit, Boolean> entry : unitList) {
                    if (entry.getValue() == true) {
                        int randomX = ThreadLocalRandom.current().nextInt(maxX / 2, maxX);
                        int randomY = ThreadLocalRandom.current().nextInt(maxY / 2, maxY);
                        move(entry.getKey(), randomX, randomY);
                    }
                }
                return;
        }
    }

    private boolean resourceInRange(PhysicalGameState pgs, int x, int y, int range)
    {
        Collection<Unit> units = pgs.getUnitsAround(x, y, range);
        for (Unit u : units) {
            if (u.getType().isResource)
                return true;
        }
        return false;
    }
    
    private int[] findBarrackLocation(PhysicalGameState pgs, int range) {
        try{
            for (Unit u : pgs.getUnits()) {
                if(u.getType().isStockpile)
                {
                    int ux = u.getX();
                    int uy = u.getY();
                    for (int i = Math.max(0, ux - range); i < Math.min(pgs.getWidth()-1, ux + range); i++){
                        for (int j = Math.max(0, uy - range); j < Math.min(pgs.getHeight()-1, uy + range); j++){
                            if (pgs.getTerrain(i, j) == 0 && pgs.getUnitAt(i, j) == null && !resourceInRange(pgs,i,j,3)) {
                                return new int[] { i, j };
                            }
                        }
                    }   
                }
            }
            return null; // No valid expansion found
        } catch (IndexOutOfBoundsException e)
        {
            System.out.println("Index out of range while searching for barrack building position");
        }
        return null;
    }

    private int[] findExpansionLocation(PhysicalGameState pgs, int range) {
        try {
            for (Unit u : pgs.getUnits()) {
                if (u.getType().isStockpile) {
                    int ux = u.getX();
                    int uy = u.getY();
                    for (int i = Math.max(0, ux - range); i < Math.min(pgs.getWidth() - 1, ux + range); i++) {
                        for (int j = Math.max(0, uy - range); j < Math.min(pgs.getHeight() - 1, uy + range); j++) {
                            if (pgs.getTerrain(i, j) == 0 && pgs.getUnitAt(i, j) == null
                                    && resourceInRange(pgs, i, j, 3)) {
                                return new int[] { i, j };
                            }
                        }
                    }
                }
            }
            return null; // No valid expansion found
        } catch (IndexOutOfBoundsException e) {
            System.out.println("Index out of range while searching for expansion");
        }
        return null;
    }
    
    private boolean isBuilding(Unit u, GameState gs)
    {
        return gs.getActionAssignment(u).action.getType() == UnitAction.TYPE_PRODUCE;
    }

    @Override
    public PlayerAction getAction(final int player, final GameState gs, int[][] action) throws Exception {
        if (useSimple)
        {
            return getActionSimple(player, gs, action);
        }
        else {
            return getActionDetailed(player, gs, action);
        }
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
        workerTable = new HashMap<Unit, Boolean>(30);
        attackerTable = new HashMap<Unit, Boolean>(30);
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
