package ai.units;

import java.util.Vector;

import javax.management.RuntimeErrorException;

import java.util.List;
import java.util.StringTokenizer;
import java.lang.Exception;

import ai.units.EpochManager;
import ai.jNeatCommon.IOseq;
import ai.jneat.*;
import rts.units.UnitType;

public class CoevolutionManager {
    public static CoevolutionManager instance = null;
    public static boolean instanceCreated = false;

    Vector<EpochManager> epochManagers; // one for each type of network to train
    Vector<Genome> genomes; // one for each type of network to train
    Vector<Population> populations; // one for each type of network to train
    String weightsPath;

    // Coevolution manager does not wait for all networks to have finished their epochs to move on 
    // because we expect each type to be spawned at different rates throughout the game

    public CoevolutionManager(List<UnitType> unitTypeList)
    {
        if (CoevolutionManager.instanceCreated)
        {
            throw new RuntimeException("Do not call Coevolution()");
        }
        for (UnitType ut : unitTypeList) {
            String popName = ut.name + "_genome";
            epochManagers.add(new EpochManager(popName, ut.ID));
        }
        //load genomes
        CoevolutionManager.instance = this;
        CoevolutionManager.instanceCreated = true;
    }
    public void startEpochAll() //called at the start of the experiment
    {
        for (EpochManager em : epochManagers)
        {
            em.startEpoch();
        }
    }
    private EpochManager getMatchingManager(int numType)
    {
        for (EpochManager manager : epochManagers) {
            if (manager.getTypeID() == numType) {
                return manager;
            }
        }
        throw new RuntimeException("Error Getting EpochManager fuck you");
    }
    public Organism getOrganism(int numType)
    {
        Organism org = null;
        org = getMatchingManager(numType).getEvaluationOrganism(numType); // Return the manager if type matches
        if (org == null) {
            throw new RuntimeException("Error Getting Organism fuck you");
        }
        return org;
    }
}
