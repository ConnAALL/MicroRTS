package ai.AALL.math;
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

    // Converts a 3D array to an array of 2D SimpleMatrix (each channel is a matrix)
    public static SimpleMatrix[] convertToMatrices(double[][][] tensor) {
        int depth = tensor[0][0].length;
        SimpleMatrix[] matrices = new SimpleMatrix[depth];
        for (int k = 0; k < depth; k++) {
            double[][] slice = new double[tensor.length][tensor[0].length];
            for (int i = 0; i < tensor.length; i++) {
                for (int j = 0; j < tensor[0].length; j++) {
                    slice[i][j] = tensor[i][j][k];
                }
            }
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

    // Perform 3D convolution on an array of matrices (tensor)
    public static SimpleMatrix[] applyConv3D(SimpleMatrix[] input, int[] kernelSize, int[] stride, int[] padding) {
        int depth = input.length;
        int rows = input[0].getNumRows();
        int cols = input[0].getNumCols();

        // Apply padding
        SimpleMatrix[] paddedInput = applyPadding(input, padding);

        // Compute new depth, height, and width after convolution
        int newDepth = (depth - kernelSize[2] + 2 * padding[2]) / stride[2] + 1;
        int newRows = (rows - kernelSize[0] + 2 * padding[0]) / stride[0] + 1;
        int newCols = (cols - kernelSize[1] + 2 * padding[1]) / stride[1] + 1;

        // Initialize output tensor
        SimpleMatrix[] output = new SimpleMatrix[newDepth];
        for (int d = 0; d < newDepth; d++) {
            output[d] = new SimpleMatrix(newRows, newCols);
        }

        // Convolution operation
        for (int d = 0; d < newDepth; d++) {
            for (int i = 0; i < newRows; i++) {
                for (int j = 0; j < newCols; j++) {
                    double sum = 0.0;

                    // Apply 3D kernel
                    for (int kd = 0; kd < kernelSize[2]; kd++) {
                        for (int ki = 0; ki < kernelSize[0]; ki++) {
                            for (int kj = 0; kj < kernelSize[1]; kj++) {
                                int rowIdx = i * stride[0] + ki;
                                int colIdx = j * stride[1] + kj;
                                int depthIdx = d * stride[2] + kd;

                                if (rowIdx < paddedInput[0].getNumRows() && colIdx < paddedInput[0].getNumCols()
                                        && depthIdx < paddedInput.length) {
                                    sum += paddedInput[depthIdx].get(rowIdx, colIdx);
                                }
                            }
                        }
                    }

                    output[d].set(i, j, sum);
                }
            }
        }
        return output;
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
}
