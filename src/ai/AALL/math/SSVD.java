package ai.AALL.math;

public class SSVD {
    private int inputSizeW;
    private int inputSizeH;
    private int outputSize;
    private int preSTensors;
    private int postSTensors;

    // Constructor
    public SSVD(int inputW, int inputH, int outputSize, int[] structure) {
        this.inputSizeW = inputW;
        this.inputSizeH = inputH;
        this.outputSize = outputSize;
        this.preSTensors = structure[0];
        this.postSTensors = structure[1];
    }

    // Method to calculate the chromosome size
    public int getChromosomeSize() {
        return this.preSTensors * Math.min(this.inputSizeH, this.inputSizeW) * Math.min(this.inputSizeH, this.inputSizeW) +
               this.postSTensors * Math.max(this.inputSizeH, this.inputSizeW) * Math.max(this.inputSizeH, this.inputSizeW) +
               this.outputSize * this.inputSizeW * this.inputSizeH;
    }
    
    /**
     * 
     * @param chromosome
     * @return array of 
     * {
     *  list of weights1 (size preSTensors)
     *  list of weights2 (size postSTensors)
     *  list of weightO (size 1)
     * }
     */
    public double[][][][] chromosomeToWeights(double[] chromosome) {
        int expectedSize = getChromosomeSize();
        if (chromosome.length != expectedSize) {
            throw new IllegalArgumentException("Vector size must be " + expectedSize + ", but got " + chromosome.length);
        }
        System.out.println("WeightedEvaluation::weight1");
        // Convert chromosome to weights_1
        int size_1 = this.preSTensors * Math.min(this.inputSizeH, this.inputSizeW) * Math.min(this.inputSizeH, this.inputSizeW);
        double[][][] weights_1 = new double[this.preSTensors][Math.min(this.inputSizeH, this.inputSizeW)][Math.min(this.inputSizeH, this.inputSizeW)];
        for (int i = 0; i < size_1; i++) {
            int tensorIndex = i / (Math.min(this.inputSizeH, this.inputSizeW) * Math.min(this.inputSizeH, this.inputSizeW));
            int rowIndex = (i % (Math.min(this.inputSizeH, this.inputSizeW) * Math.min(this.inputSizeH, this.inputSizeW))) / Math.min(this.inputSizeH, this.inputSizeW);
            int colIndex = i % Math.min(this.inputSizeH, this.inputSizeW);
            weights_1[tensorIndex][rowIndex][colIndex] = chromosome[i];
        }
        System.out.println("WeightedEvaluation::weight2");
        // Convert chromosome to weights_2
        int size_2 = this.postSTensors * Math.max(this.inputSizeH, this.inputSizeW) * Math.max(this.inputSizeH, this.inputSizeW);
        double[][][] weights_2 = new double[this.postSTensors][Math.max(this.inputSizeH, this.inputSizeW)][Math.max(this.inputSizeH, this.inputSizeW)];
        for (int i = size_1; i < size_1 + size_2; i++) {
            int tensorIndex = (i - size_1) / (Math.max(this.inputSizeH, this.inputSizeW) * Math.max(this.inputSizeH, this.inputSizeW));
            int rowIndex = ((i - size_1) % (Math.max(this.inputSizeH, this.inputSizeW) * Math.max(this.inputSizeH, this.inputSizeW))) / Math.max(this.inputSizeH, this.inputSizeW);
            int colIndex = (i - size_1) % Math.max(this.inputSizeH, this.inputSizeW);
            weights_2[tensorIndex][rowIndex][colIndex] = chromosome[i];
        }
        System.out.println("WeightedEvaluation::weightO1");
        // Convert chromosome to weightO
        double[][][] weightO = new double[1][this.outputSize][this.inputSizeW * this.inputSizeH];
        for (int i = size_1 + size_2; i < chromosome.length; i++) {
            int rowIndex = (i - size_1 - size_2) / (this.inputSizeW * this.inputSizeH);
            int colIndex = (i - size_1 - size_2) % (this.inputSizeW * this.inputSizeH);
            weightO[0][rowIndex][colIndex] = chromosome[i];
        }
        System.out.println("WeightedEvaluation::weightO2");
        // Return the weights
        return new double[][][][] {weights_1, weights_2, weightO};
    }
}
