package ai.units;

import java.util.Arrays;

// Responsible for splitting the network into parts for softmaxing into several different parts
// Inspired by the mouse control, the AI will control the units with existing abstraction layer
public class NetworkOuput {
    //Total unit type count is 8
    //Selecting unit 8 means nullop

    //width * height
    //target tile
    public int output_type;
    public int output_tile;
    
    NetworkOuput(int boardX, int boardY, double[] output)
    {
        if(output.length != 8 + boardX * boardY)
        {
            throw new IllegalArgumentException("Lengths exceed array size");
        }
        double[] unitTypeOutput = Arrays.copyOfRange(output, 0, 8);
        double[] tileOutput = Arrays.copyOfRange(output, 8, 8 + boardX * boardY);

        output_type = NetworkHelpers.selectAction(NetworkHelpers.computeSoftmax(unitTypeOutput));
        output_tile = NetworkHelpers.selectAction(NetworkHelpers.computeSoftmax(tileOutput));
    }
}
