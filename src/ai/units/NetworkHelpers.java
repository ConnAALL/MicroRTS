package ai.units;

import java.util.Arrays;
import java.util.List;
import java.util.Vector;
import java.util.Random;

import ai.jneat.NNode;
import ai.jneat.Network;
import rts.PhysicalGameState;
import rts.units.Unit;
import rts.units.UnitType;

public class NetworkHelpers {
    
    // Input Space Statics
    public static final int UNIT_NULLOP = 0;
    public static final int UNIT_BASE = 100;
    public static final int UNIT_BARRACK = 200;
    public static final int UNIT_WORKER = 300;
    public static final int UNIT_LIGHT = 400;
    public static final int UNIT_HEAVY = 500;
    public static final int UNIT_RANGED = 600;
    public static final int UNIT_RESOURCE = 900;
    

    // Rest of the output space will be the representation of the board to position

    private static int[] concatenateArrays(int[]... arrays) {
        // Calculate the total length
        int totalLength = 0;
        for (int[] array : arrays) {
            totalLength += array.length;
        }

        // Create a new array to hold the result
        int[] result = new int[totalLength];

        // Copy elements from each array into the result array
        int currentIndex = 0;
        for (int[] array : arrays) {
            System.arraycopy(array, 0, result, currentIndex, array.length);
            currentIndex += array.length;
        }

        return result;
    }

    public static int[] getFlattened(PhysicalGameState pgs, int player)
    {
        int[] friendlyBoard = new int[pgs.getWidth() * pgs.getHeight()];
        // Recreate the grid
        for (int y = 0; y < pgs.getHeight(); y++) {
            for (int x = 0; x < pgs.getWidth(); x++) {
                // Get terrain type at (x, y)
                friendlyBoard[x + y * pgs.getHeight()] = pgs.getTerrain(x, y);
            }
        }
        int[] enemyBoard = friendlyBoard.clone();

        for (Unit u : pgs.getUnits()) {
            UnitType t = u.getType();
            int typeID = 0;
            switch (t.name) {
                case "Base":
                    typeID = 100;
                    break;
                case "Barracks":
                    typeID = 200;
                    break;
                case "Worker":
                    typeID = 300;
                    break;
                case "Light":
                    typeID = 400;
                    break;
                case "Heavy":
                    typeID = 500;
                    break;
                case "Ranged":
                    typeID = 600;
                    break;
                case "Resource":
                    typeID = 900;
                    break;
            }
            float health_ratio = (float) u.getHitPoints() / (float) u.getMaxHitPoints();
            int health_simple = (int) (health_ratio * 10);
            typeID += 10 * health_simple;

            if (typeID > 900) // if resource add to both maps
            {
                friendlyBoard[u.getX() + u.getY() * pgs.getHeight()] = typeID;
                enemyBoard[u.getX() + u.getY() * pgs.getHeight()] = typeID;
            } else if (u.getPlayer() == player) {
                friendlyBoard[u.getX() + u.getY() * pgs.getHeight()] = typeID;
            } else {
                enemyBoard[u.getX() + u.getY() * pgs.getHeight()] = typeID;
            }
        }
        return concatenateArrays(friendlyBoard, enemyBoard);
    }
    
    public static Vector<Double> computeSoftmax(Vector<Double> outputs) {
        Vector<Double> softmax = new Vector<Double>();
        double sumExp = 0.0;
        // Compute the maximum value for numerical stability
        double maxValue = outputs.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        // Compute the sum of exponentials
        for (double o : outputs) {
            sumExp += Math.exp(o - maxValue); // Subtract max for stability
        }
        // Compute softmax values
        for (double o : outputs) {
            double softmaxValue = Math.exp(o - maxValue) / sumExp;
            softmax.add(softmaxValue);
        }
        return softmax;
    }
    public static Vector<Double> computeSoftmax(double[] outputs) {
        Vector<Double> softmax = new Vector<Double>();
        double sumExp = 0.0;
		// Compute the maximum value for numerical stability
		double maxValue = Arrays.stream(outputs).max().orElse(0.0);
        // Compute the sum of exponentials
        for (double o : outputs) {
            sumExp += Math.exp(o - maxValue); // Subtract max for stability
        }
        // Compute softmax values
        for (double o : outputs) {
            double softmaxValue = Math.exp(o - maxValue) / sumExp;
            softmax.add(softmaxValue);
        }
        return softmax;
    }

	public static Vector<Double> evaluateNetworkSoftMax(Network network, double[] in)
	{
		return computeSoftmax(evaluateNetworkContinuous(network, in));
	}

	public static Vector<Double> evaluateNetworkContinuous(Network network, double[] in)
    {
        network.flush();
        int net_depth = network.max_depth();
        network.load_sensors(in);
        boolean success = network.activate();
        for (int relax = 0; relax <= net_depth; relax++)
            success = network.activate();
        if (success) {
            Vector<NNode> outNodes = network.getOutputs();
            Vector<Double> distr = new Vector<Double>();
            for (NNode node : outNodes) {
                distr.add(node.getActivation());
            }
            network.flush();
            return distr;
        } else {
            network.flush();
            return null;
        }
    }
    
    public static int selectAction(Vector<Double> probabilities) {
        Random random = new Random();
        double rand = random.nextDouble(); // Random value between 0.0 and 1.0

        double cumulativeProbability = 0.0;
        for (int i = 0; i < probabilities.size(); i++) {
            cumulativeProbability += probabilities.get(i);
            if (rand < cumulativeProbability) {
                return i; // Return the index of the selected action
            }
        }
        throw new IllegalStateException("Probabilities do not sum to 1.");
    }
}
