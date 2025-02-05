package ai.units;

import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Iterator;
import java.io.File;

import ai.jNeatCommon.NeatConstant;
import ai.jneat.*;

// Responsible for epochs of one type of network
public class EpochManager extends Neat {
    HashMap<Integer, Organism> organismMap;
    HashMap<Integer, Double> fitnessMap;
    HashMap<Integer, Integer> evalMap;
    Population pop = null;
    boolean epochReady = false;
    int networkType = 0;
    int generation = 0;
    String name = "";
    final String path = "./weights/";
    final int popSize = 20;
    final int evaluationPerOrganism = 10;
    final double prb_link = 0.50;
    final int inputSize = 10;
    final int outputSize = 20;
    final int nMax = 2000;
    final boolean recurrent = true;

    public EpochManager(String _name, int _networkType)
    {
        name = _name;
        networkType = _networkType;
    }

    public void startEpoch()
    {
        String fname_prefix = path + name;
        File file = new File(fname_prefix + ".last");
        if (pop != null)
        {
            pop.verify();
        }
        if (file.exists()) {
            CreatePopulation(fname_prefix, NeatConstant.HOT);
        }
        else
        {
            CreatePopulation(fname_prefix, NeatConstant.COLD);
        }
        
        organismMap.clear();
        fitnessMap.clear();
        evalMap.clear();
        // put all organism in pop.organism into the hash map for easy access
        int iterID = 0;
        Iterator<Organism> itr_organism;
        itr_organism = pop.organisms.iterator();
        while (itr_organism.hasNext()) {
            organismMap.put(iterID, itr_organism.next());

            iterID++;
        }
        epochReady = true;
    }

    private boolean isEpochDone()
    {
        for (Integer id : organismMap.keySet()) {
            int evaluations = evalMap.getOrDefault(id, 0);
            if (evaluations < evaluationPerOrganism) {
                return false; // If any organism has fewer evaluations than required, it's not finished
            }
        }
        return true; // All organisms have been evaluated sufficiently
    }
    
    public void recordFitnessAccumulate(int id, double fitness)
    {
        if (fitnessMap.containsKey(id))
        {
            recordFitness(id, fitnessMap.get(id) + fitness);
        }
        else
        {
            recordFitness(id, fitness);
        }
    }

    public void recordFitness(int id, double fitness)
    {
        if (evalMap.containsKey(id)) {
            fitnessMap.put(id, fitness);
            evalMap.put(id, evalMap.get(id) + 1);
        } else {
            fitnessMap.put(id, fitness);
            evalMap.put(id, 1);
        }
        if(isEpochDone())
        {
            epochReady = false;
            System.out.print(String.format("\n---------------- Generation %s of %s ----------------------", generation, name));
            System.out.print("\n  Population : innov num   = " + pop.getCur_innov_num());
            System.out.print("\n             : cur_node_id = " + pop.getCur_node_id());
            //System.out.print("\n   result    : " + esito);
            pop.epoch(generation);
            savePop();
            startEpoch();
        }
    }
    
    /**
     * Returns an organism that has fewer evaluations than the required number.
     * @return An organism that needs more evaluations, or null if all are sufficiently evaluated.
     */
    public Organism getEvaluationOrganism(int _networkType) {
        if (_networkType != networkType) // incorrect type of network requested or epoch is not ready
            return null;
        for (Integer id : organismMap.keySet()) {
            int evaluations = evalMap.getOrDefault(id, 0);
            if (evaluations < evaluationPerOrganism) {
                return organismMap.get(id); // Return the first organism that needs more evaluations
            }
        }
        return null; // All organisms are sufficiently evaluated
    }

    /**
	 * This is a sample of creating a new Population with
	 * 'size_population' organisms , and simulation
	 * of XOR example
	 * This sample can be started in two modality :
	 * -cold : each time the population is re-created from 0;
	 * -warm : each time the population re-read last population
	 * created and restart from last epoch.
	 * (the population backup file is : 'c:\\jneat\\dati\\population.primitive'
	 */
    private void CreatePopulation(String fname_prefix, int mode) {
        if (pop != null) {
            System.out.println("Population already loaded");
            return;
        }
        System.out.println(String.format("------ Creating Population of %s -------.", name));
        System.out.println(" Spawned population off genome");

        // default cold is : 3 sensor (1 for bias) , 1 out , 5 nodes max, no recurrent
        if (mode == NeatConstant.COLD)
            pop = new Population(popSize, inputSize, outputSize, nMax, recurrent, prb_link); // cold start-up
        // pop = new Population(size_population, 3, 1, 5, recurrent, prb_link); // cold
        // start-up
        else
            pop = new Population(fname_prefix + ".last"); // warm start-up

        pop.verify();
        System.out.print("\n---------------- Generation starting with----------");
        System.out.print("\n  Population : innov num   = " + pop.getCur_innov_num());
        System.out.print("\n             : cur_node_id = " + pop.getCur_node_id());
        System.out.print("\n---------------------------------------------------");

        savePop();
        System.out.println(String.format("------ Population of %s Ready-------.", name));
    }
    
    public void savePop()
    {
        // backup of population for warm startup
        System.out.println(String.format("------ Saving Population of %s -------.", name));
        pop.print_to_filename(path + name + ".last");
    }

    public int getTypeID()
    {
        return networkType;
    }
}