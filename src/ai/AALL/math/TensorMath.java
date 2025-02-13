package ai.AALL.math;

import java.util.Random;
import org.ejml.data.Matrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

public class TensorMath {
    public static double[][][] intTensorToDoubleTensor(int[][][] intArray) {
        int rows = intArray.length;
        int cols = intArray[0].length;
        int depth = intArray[0][0].length;

        double[][][] doubleArray = new double[rows][cols][depth];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < depth; k++) {
                    doubleArray[i][j][k] = (double) intArray[i][j][k];
                }
            }
        }
        return doubleArray;
    }

    // Generates a random 3D tensor (simulating the input tensor)
    public static double[][][] generateRandomTensor(int rows, int cols, int depth) {
        double[][][] tensor = new double[rows][cols][depth];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < depth; k++) {
                    tensor[i][j][k] = Math.random();
                }
            }
        }
        return tensor;
    }

    // Converts a 3D array to an array of 2D SimpleMatrix (each channel is a SimpleMatrix)
    public static SimpleMatrix[] convertToMatrices(double[][][] tensor) {
        //int depth = tensor[0][0].length;
        System.out.printf("convertToMatrices:: %d %d %d\n", tensor.length, tensor[0].length, tensor[0][0].length); // 6 16 16
        SimpleMatrix[] matrices = new SimpleMatrix[tensor.length];
        for (int k = 0; k < tensor.length; k++)
        {
            double[][] slice = tensor[k];
            matrices[k] = new SimpleMatrix(slice);
        }
        return matrices;
    }

    // Extracts and processes features using weighted summation
    public static SimpleMatrix[] processFeatures(SimpleMatrix[] input, int[] featureSizes) {
        int p = 0;
        SimpleMatrix[] features = new SimpleMatrix[featureSizes.length];
        for (int i = 0; i < featureSizes.length; i++) {
            int size = featureSizes[i];

            // Weighted summation
            SimpleMatrix featureSum = new SimpleMatrix(input[0].getNumRows(), input[0].getNumCols());
            for (int j = 0; j < size; j++) {
                System.out.printf("%d %d %d\n", p, j, p + j);
                featureSum = featureSum.plus(input[p + j].scale(j)); // Multiply by weight
            }

            features[i] = featureSum;
            p += size;
        }
        return features;
    }

    // Concatenates processed features along depth axis
    public static SimpleMatrix concatenateFeatures(SimpleMatrix[] features) {
        SimpleMatrix concatenated = features[0];
        for (int i = 1; i < features.length; i++) {
            concatenated = concatenated.combine(0, concatenated.getNumCols(), features[i]);
        }
        return concatenated;
    }

    // Adds zero-padding around a matrix
    public static SimpleMatrix applyPadding(SimpleMatrix input, int padding) {
        int newCols = input.getNumCols() + 2 * padding;
        SimpleMatrix padded = new SimpleMatrix(input.getNumRows(), newCols);

        // Copy original matrix into the center of the padded matrix
        for (int i = 0; i < input.getNumRows(); i++) {
            for (int j = 0; j < input.getNumCols(); j++) {
                padded.set(i, j + padding, input.get(i, j));
            }
        }
        return padded;
    }

    // Apply zero-padding to a 3D tensor (array of SimpleMatrix)
    public static SimpleMatrix[] applyPadding(SimpleMatrix[] input, int[] padding) {
        int depth = input.length;
        int rows = input[0].getNumRows();
        int cols = input[0].getNumCols();

        int newDepth = depth + 2 * padding[2];
        int newRows = rows + 2 * padding[0];
        int newCols = cols + 2 * padding[1];

        SimpleMatrix[] padded = new SimpleMatrix[newDepth];
        for (int d = 0; d < newDepth; d++) {
            padded[d] = new SimpleMatrix(newRows, newCols);

            // Fill center with original data
            if (d >= padding[2] && d < depth + padding[2]) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        padded[d].set(i + padding[0], j + padding[1], input[d - padding[2]].get(i, j));
                    }
                }
            }
        }
        return padded;
    }

    public static SimpleMatrix squeeze(SimpleMatrix[] tensor) {
        // Case 1: Remove first dimension if it's 1 (tensor[0] is the 2D matrix)
        if (tensor.length == 1) {
            return tensor[0]; // Shape becomes [X, Y]
        }

        // Case 2: Remove last dimension if it's 1 (in this case, each SimpleMatrix has
        // 1 column)
        if (tensor[0].getNumCols() == 1) {
            // Handle the 2D case with the same number of rows, but only 1 column
            SimpleMatrix squeezed = new SimpleMatrix(tensor.length, tensor[0].getNumRows());
            for (int i = 0; i < tensor.length; i++) {
                for (int j = 0; j < tensor[i].getNumRows(); j++) {
                    squeezed.set(i, j, tensor[i].get(j, 0)); // Copy value from the first column
                }
            }
            return squeezed; // New [X, Y]
        }

        // Case 3: Remove middle dimension if it's 1 (tensor has 1 row)
        if (tensor[0].getNumRows() == 1) {
            // Handle the 2D case where only 1 row is present
            SimpleMatrix squeezed = new SimpleMatrix(tensor.length, tensor[0].getNumCols());
            for (int i = 0; i < tensor.length; i++) {
                for (int j = 0; j < tensor[i].getNumCols(); j++) {
                    squeezed.set(i, j, tensor[i].get(0, j)); // Copy from the only row
                }
            }
            return squeezed; // New [X, Y]
        }
        // If no singleton dimensions, return the original tensor as a SimpleMatrix (2D)
        return tensor[0]; // Or handle differently
    }

    public static DMatrixRMaj multiply(DMatrixRMaj A, DMatrixRMaj B) {
        DMatrixRMaj result = new DMatrixRMaj(A.numRows, B.numCols);
        CommonOps_DDRM.mult(A, B, result);
        return result;
    }

    public static void softMax(DMatrixRMaj matrix) {
        // Calculate the sum of exponentials across the entire matrix
        double sumExp = 0;
        for (int row = 0; row < matrix.numRows; row++) {
            for (int col = 0; col < matrix.numCols; col++) {
                sumExp += Math.exp(matrix.get(row, col));
            }
        }

        // Update the matrix with softmax values (exp(x) / sum(exp(x)))
        for (int row = 0; row < matrix.numRows; row++) {
            for (int col = 0; col < matrix.numCols; col++) {
                double softmaxValue = Math.exp(matrix.get(row, col)) / sumExp;
                matrix.set(row, col, softmaxValue);
            }
        }
    }

    // ReLU activation function (element-wise)
    public static DMatrixRMaj relu(DMatrixRMaj input) {
        DMatrixRMaj output = input.copy();
        for (int i = 0; i < input.getNumRows(); i++) {
            for (int j = 0; j < input.getNumCols(); j++) {
                if (output.get(i, j) < 0) {
                    output.set(i, j, 0); // Apply ReLU (set negative values to zero)
                }
            }
        }
        return output;
    }

    public static DMatrixRMaj flatten(DMatrixRMaj input)
    {
        double[] data = input.getData();
        return new DMatrixRMaj(data);
    }

    public static SimpleMatrix[] generateKernel3D(int[] k_s)
    {
        int depths = k_s[0];int rows = k_s[1];int cols = k_s[2];
        double lowerBound = -1 * depths * rows * cols;
        double upperBound = depths * rows * cols;

        // Create the matrix with the given dimensions
        SimpleMatrix[] output = new SimpleMatrix[depths];
        Random random = new Random(1);
        // Fill the matrix with random values from uniform distribution
        for (int k = 0; k < depths; k++) {
            output[k] = new SimpleMatrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    // Generate a random value between lowerBound and upperBound
                    double randomValue = lowerBound + (upperBound - lowerBound) * random.nextDouble();
                    output[k].set(i, j, randomValue);
                }
            }
        }
        return output;
    }

    public static SimpleMatrix[] conv3D(SimpleMatrix[] input, int[] kernelSize, int[] stride, int[] padding) {
        int inputDepth = input.length;
        int inputRow = input[0].getNumRows();
        int inputCol = input[0].getNumCols();
        
        int kernelDepth = kernelSize[0];int kernelRow = kernelSize[1];int kernelCol = kernelSize[2];
        int strideDepth = stride[0];int strideRow = stride[1];int strideCol = stride[2];
        int paddingDepth = padding[0];int paddingRow = padding[1];int paddingCol = padding[2];

        int paddedDepth = inputDepth + 2 * paddingDepth;
        int paddedRow = inputRow + 2 * paddingRow;
        int paddedCol = inputCol + 2 * paddingCol;
        //System.out.printf("\nNEW SIZE: %d %d %d", outDepth, outRow, outCol);

        SimpleMatrix[] paddedInput = new SimpleMatrix[paddedDepth];
        for (int d = 0; d < paddedDepth; d++) {
            paddedInput[d] = new SimpleMatrix(inputRow, inputCol);
            //paddedDepth = 10
            //padding = 3
            //starting index = 3 (pads are 0 1 2)
            //ending index = 6 (pads are 7 8 9)
            //paddedDepth - padding = 7
            if (d >= paddingDepth && d < paddedDepth - paddingDepth)
            {
                for (int r = 0; r < inputRow; r++) {
                    for (int c = 0; c < inputCol; c++) {
                        paddedInput[d + paddingDepth].set(r + paddedRow, c + paddedCol, input[d].get(r, c));
                    }
                }
            }
        }
        
        // Calculate output dimensions
        int outDepth = (paddedDepth - kernelDepth) / strideDepth + 1;
        int outRow = (paddedRow - kernelRow) / strideRow + 1;
        int outCol = (paddedCol - kernelCol) / strideCol + 1;
        System.out.printf("\nNEW SIZE: %d %d %d", outDepth, outRow, outCol);
        
        // Create output tensor
        SimpleMatrix[] output = new SimpleMatrix[outDepth];
        SimpleMatrix[] kernel = generateKernel3D(kernelSize);
        // Convolution operation
        for (int d = 0; d < outDepth; d++) {
            output[d] = new SimpleMatrix(outRow, outCol);
            for (int r = 0; r < outRow; r++) {
                for (int c = 0; c < outCol; c++) {
                    double sum = 0.0;
                    
                    // Apply kernel to the current patch of input tensor
                    for (int kd = 0; kd < kernelDepth; kd++) {
                        for (int kr = 0; kr < kernelRow; kr++) {
                            for (int kc = 0; kc < kernelCol; kc++) {
                                int inputD = d * strideDepth + kd;
                                int inputR = r * strideRow + kr;
                                int inputC = c * strideCol + kc;

                                // Perform element-wise multiplication and summation
                                sum += paddedInput[inputD].getIndex(inputR, inputC) * kernel[kd].get(kr, kc);
                            }
                        }
                    }
                    output[d].set(r, c, sum);
                }
            }
        }
        return output;
    }
}
