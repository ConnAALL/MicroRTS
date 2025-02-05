package ai.units;

import rts.PhysicalGameState;
import rts.units.UnitType;
import ai.jneat.*;
import ai.jNeatCommon.*;
import ai.units.PerUnitAI;
//import ai.units.ObservationSpace;

public class NEATUnitAI extends PerUnitAI {

    NEATUnitAI(int _player, UnitType _type)
    {
        super(_player, _type);
    }

    public int Think(double[] input, int player)
    {
        NetworkHelpers.evaluateNetworkContinuous(organism.net,input);
        return 0;
    }
}
