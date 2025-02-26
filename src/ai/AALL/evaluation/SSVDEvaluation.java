package ai.AALL.evaluation;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.dense.row.mult.MatrixMatrixMult_DDRM;
import org.ejml.interfaces.decomposition.QRDecomposition;
import java.lang.Math;
import java.util.Arrays;

import ai.AALL.math.TensorMath;
import ai.AALL.math.SSVD;
import rts.GameState;
public class SSVDEvaluation extends WeightedEvaluation{

    // Must match with python counterpart
    private int[] featureSizes = { 5, 5, 3, 8, 6, 2 };
    private int featureTotal = 29;
    private SSVD ssvd;
    private DMatrixRMaj[] weights1;
    private DMatrixRMaj[] weights2;
    private DMatrixRMaj weightsO;

    private SimpleMatrix[] conv1;
    private SimpleMatrix[] conv2;

    public SSVDEvaluation()
    {
        //System.out.println("Started initializing SSVDEvaluation");
        int inputW = 16;
        int inputH = 16;
        int output = 1; //Using continuous output for MCTS
        ssvd = new SSVD(inputW, inputH, output, new int[] { 2, 2 });
        //System.out.println("Started initializing SSVDEvaluation 1 ");
        int[] k1 = { 4, 1, 1 };
        int[] k2 = { 2, 1, 1 };
        conv1 = TensorMath.generateKernel3D(k1);
        conv2 = TensorMath.generateKernel3D(k2);
    }
    @Override
    public void parseWeights()
    {
        double[] doubleweights = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            doubleweights[i] = (double) weights[i];
        }
        //System.out.println("SSVDEvaluation::parseWeights");
        //System.out.println("Started initializing SSVDEvaluation 2 ");
        double[][][][] weightTensors = ssvd.chromosomeToWeights(doubleweights);
        //System.out.println("SSVDEvaluation::parseWeights::ssvd");
        //System.out.println("Started initializing SSVDEvaluation 3 ");
        weights1 = new DMatrixRMaj[weightTensors[0].length];
        for (int i = 0; i < weightTensors[0].length; i++) {
            weights1[i] = new DMatrixRMaj(weightTensors[0][i]);  // Initialize weights1
        }
        weights2 = new DMatrixRMaj[weightTensors[1].length];
        for (int i = 0; i < weightTensors[1].length; i++) {
            weights2[i] = new DMatrixRMaj(weightTensors[1][i]);  // Initialize weights1
        }
        weightsO = new DMatrixRMaj(weightTensors[2][0]);
        //System.out.println("SSVDEvaluation::parseWeights::END");
        //System.out.println("Started initializing SSVDEvaluation END ");
    }
    public double evaluateSSVD(double[][][] obs)
    {
        //System.out.println("SSVDEvaluation::evaluateSSVD::START");
        // Convert to SimpleMatrix array
        SimpleMatrix[] inputTensor = TensorMath.convertToMatrices(obs);
        //System.out.println(Arrays.deepToString(obs));
        //System.out.printf("conv size %d %d %d\n", inputTensor.length, inputTensor[0].getNumRows(), inputTensor[0].getNumCols());
        // Feature sizes as in PyTorch
        //int[] featureSizes = {5, 5, 3, 8, 6, 2}; //10 11 8 29

        // Process features
        //SimpleMatrix[] processedFeatures = TensorMath.processFeatures(inputTensor, featureSizes);
        //System.out.printf("conv size %d %d %d\n", inputTensor.length, inputTensor[0].getNumRows(), inputTensor[0].getNumCols());
        // Apply 3D convolutions
        // SimpleMatrix convolved = TensorMath.applyConv3D(inputConcat, 4, 2); // First conv (1,1,4) with stride (1,1,2), padding (0,0,2)
        // convolved = TensorMath.applyConv3D(convolved, 4, 2);
        // convolved = TensorMath.applyConv3D(convolved, 4, 2);
        // int[] k1 = { 1, 1, 4 };
        // int[] s1 = { 1, 1, 2 };
        // int[] p1 = { 0, 0, 2 };
        
        int[] s1 = { 2, 1, 1 };
        int[] p1 = { 2, 0, 0 };
        SimpleMatrix[] convolved = TensorMath.conv3D(inputTensor, conv1, s1, p1);
        //System.out.printf("conv size %d %d %d\n", convolved.length, convolved[0].getNumRows(),convolved[0].getNumCols());
        
        convolved = TensorMath.conv3D(convolved, conv1, s1, p1);
        //System.out.printf("conv size %d %d %d\n", convolved.length, convolved[0].getNumRows(),convolved[0].getNumCols());
        
        convolved = TensorMath.conv3D(convolved, conv1, s1, p1);
        //System.out.printf("conv size %d %d %d\n", convolved.length, convolved[0].getNumRows(),convolved[0].getNumCols());
        
        // int[] k2 = { 1, 1, 2 };
        // int[] s2 = { 1, 1, 1 };
        // int[] p2 = { 0, 0, 0 };
        int[] s2 = { 1, 1, 1 };
        int[] p2 = { 0, 0, 0 };
        convolved = TensorMath.conv3D(convolved, conv2, s2, p2);
        //System.out.printf("conv size %d %d %d\n", convolved.length, convolved[0].getNumRows(), convolved[0].getNumCols());
        SimpleMatrix flattened = TensorMath.squeeze(convolved);
        // Compute Singular Value Decomposition
        SimpleSVD<SimpleMatrix> svd = flattened.svd();

        // Extract U, S, and V matrices
        SimpleMatrix U = svd.getU();
        SimpleMatrix S = svd.getW(); // Diagonal matrix of singular values
        SimpleMatrix V = svd.getV();
        //System.out.println("SSVDEvaluation::evaluateSSVD::MID");
        // Apply QR decomposition to stabilize U and Vh
        QRDecomposition<DMatrixRMaj> decomposer = DecompositionFactory_DDRM.qr();
        DMatrixRMaj U_d = U.getMatrix();
        DMatrixRMaj U_stable = new DMatrixRMaj();
        decomposer.decompose(U_d);
        decomposer.getQ(U_stable, false);

        DMatrixRMaj V_d = V.transpose().getMatrix();
        DMatrixRMaj V_stable = new DMatrixRMaj();
        decomposer.decompose(V_d);
        decomposer.getQ(V_stable, false);

        DMatrixRMaj result = TensorMath.relu(TensorMath.multiply(U_stable, weights1[0]));
        // Apply ReLU after each multiplication with weights1
        for (int i = 1; i < weights1.length; i++) {
            result = TensorMath.relu(TensorMath.multiply(result, weights1[i])); // ReLU after each step
        }
        // Apply ReLU after multiplication with Sigma
        result = TensorMath.relu(TensorMath.multiply(result, S.getMatrix()));
        // Apply ReLU after each multiplication with weights2
        for (int i = 1; i < weights2.length; i++) {
            result = TensorMath.relu(TensorMath.multiply(result, weights2[i])); // ReLU after each step
        }

        // Final multiplication with weightsO and Vh_stable
        result = TensorMath.multiply(weightsO, TensorMath.flatten(TensorMath.multiply(result, V_stable)));
        
        int rx = result.getNumRows();
        int ry = result.getNumCols();
        int w1r = weights1[0].getNumRows();
        int w1c = weights1[0].getNumCols();
        int w2r = weights2[0].getNumRows();
        int w2c = weights2[0].getNumCols();
        int wOr = weightsO.getNumRows();
        int wOc = weightsO.getNumCols();
        assert result.getNumRows() == 1 && result.getNumCols() == 1 : String.format(
            "Invalid output size for mcts evaluation. Expected output of 1x1, got output of %dx$d. Weight matrix shapes are %dx%d %dx%d %dx%d",
                rx, ry, w1r, w1c, w2r, w2c, wOr, wOc);
        // Output will always be a single value
        //System.out.println("SSVDEvaluation::evaluateSSVD::END");
        return result.getData()[0];
    }
    
    @Override
    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        // TODO Auto-generated method stub
        double[][][] obs = TensorMath.intTensorToDoubleTensor(gs.getVectorObservation(minplayer));
        double value = evaluateSSVD(obs);
        return (float) Math.tanh(value);
    }

    @Override
    public float upperBound(GameState gs) {
        // Max of tanh is 1, Min is -1
        return 1.0f;
    }
    
}
