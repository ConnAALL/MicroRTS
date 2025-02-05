package ai.units;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import ai.jneat.*;
import ai.units.NetworkHelpers;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.UnitAction;
import rts.units.Unit;
import rts.units.UnitType;
import ai.units.NetworkOuput;
import ai.units.NetworkHelpers;
import ai.units.CoevolutionManager;

public class PerUnitAI {
    Organism organism;
    int player;
    UnitType type;

    public PerUnitAI(int _player, UnitType _type)
    {
        player = _player;
        type = _type;
        organism = CoevolutionManager.instance.getOrganism(_type.ID);

    }

    public NetworkOuput Think(PhysicalGameState pgs, double[] input, int player)
    {
        Vector<Double> output_vec = NetworkHelpers.evaluateNetworkSoftMax(organism.net, input);
        double[] outputs = new double[output_vec.size()];
        outputs = output_vec.stream().mapToDouble(Double::doubleValue).toArray();

        return new NetworkOuput(pgs.getWidth(), pgs.getHeight(), outputs);
    }
    
}
