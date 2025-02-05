/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ai;

import ai.core.AI;
import ai.core.ParameterSpecification;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import rts.*;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;
import ai.units.NetworkHelpers;
import ai.units.NetworkOuput;
import ai.units.PerUnitAI;
import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.abstraction.pathfinding.PathFinding;
/**
 *
 * @author santi
 */
public class CooperativeAI extends AbstractionLayerAI {
    HashMap<Long, PerUnitAI> unitMap = new HashMap<>();
    UnitTypeTable unitTable = new UnitTypeTable();
    protected UnitTypeTable utt;
    public CooperativeAI(UnitTypeTable a_utt) {
        this(a_utt, new AStarPathFinding());
    }

    public CooperativeAI(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }
    
    @Override
    public void reset() {
        super.reset();
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
    }

    @Override
    public AI clone() {
        return new CooperativeAI(utt, pf);
    }

    @Override
    public List<ParameterSpecification> getParameters()
    {
        return new ArrayList<>();
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        // Update unitmap
        for (Unit u : pgs.getUnits()) {
            if(u.getHitPoints() > 0 && u.getPlayer() == player)
            {
                PerUnitAI ai = unitMap.get(u.getID());
                if(ai == null)
                {
                    //create new ai
                    PerUnitAI pai = new PerUnitAI(player, u.getType());
                    unitMap.put(u.getID(), pai);
                }
            }
        }
        
        
        int[] inputINT = NetworkHelpers.getFlattened(pgs, player);
        double[] input = new double[inputINT.length];
        for (int i = 0; i < inputINT.length; i++) {
            input[i] = inputINT[i]; // Cast int to double
        }
        for (Unit u : pgs.getUnits()) { 
            if (u.getPlayer() == player) {//for each friendly unit, think
                if (gs.getActionAssignment(u) == null) {
                    PerUnitAI ai = unitMap.get(u.getID());
                    if (ai == null)
                    {
                        continue;
                    }
                    NetworkOuput action = ai.Think(pgs, input, player);
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
}
