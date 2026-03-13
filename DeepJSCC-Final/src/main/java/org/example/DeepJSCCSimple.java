package org.example;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.*;
import java.util.Map;
public class DeepJSCCSimple {
    // Hyperparameters
    static final int batchSize = 128;
    static final int numEpochs = 5;
    static final double learningRate = 1.5e-4;
    // convert SNR in dB to noise standard deviation ---
    public static double noiseStdFromSNRdB(double snrDb) {
        double snrLinear = Math.pow(10, snrDb / 10.0);
        return Math.sqrt(1.0 / snrLinear);
    }
    static final double channelSNRdB = 10.0; // training SNR (dB)
    static final double beta = 1.0;  // weight for reconstruction
    static final double gamma = 1.0; // weight for classification
    public static void main(String[] args) throws Exception {
        Nd4j.getMemoryManager().setAutoGcWindow(3000);
        // Build network
        ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
                .seed(12345)
                // Optimizer with learning-rate schedule + gradient clipping
                .updater(new Adam.Builder()
                        .learningRateSchedule(new MapSchedule(
                                ScheduleType.EPOCH,
                                Map.of(
                                        0, 1.5e-4,   // Epoch 0–1: base LR
                                        2, 1.0e-4,   // After 2 epochs: slower
                                        4, 7.5e-5    // After 4 epochs: fine-tune
                                )
                        ))
                        .build())
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .weightInit(WeightInit.XAVIER)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutionalFlat(28, 28, 1))
                // Encoder
                .addLayer("enc_c1", new ConvolutionLayer.Builder(4, 4).stride(2, 2)
                        .nOut(32).activation(Activation.LEAKYRELU).build(), "input")
                .addLayer("enc_c2", new ConvolutionLayer.Builder(4, 4).stride(2, 2)
                        .nOut(64).activation(Activation.LEAKYRELU).build(), "enc_c1")
                .addLayer("enc_flat", new DenseLayer.Builder().nOut(256)
                        .activation(Activation.LEAKYRELU).build(), "enc_c2")
                // Feature bottleneck
                .addLayer("features", new DenseLayer.Builder().nOut(200)
                        .activation(Activation.IDENTITY).build(), "enc_flat")
                // Decoder (for reconstruction)
                .addLayer("dec_fc", new DenseLayer.Builder().nOut(256)
                        .activation(Activation.RELU).build(), "features")
                .addLayer("dec_out_flat", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID).nOut(28 * 28).build(), "dec_fc")
                // Classifier head
                .addLayer("clf_fc1", new DenseLayer.Builder().nOut(128)
                        .activation(Activation.RELU).build(), "features")
                .addLayer("classifier", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nOut(10).build(), "clf_fc1")

                .setOutputs("dec_out_flat", "classifier");
        ComputationGraph net = new ComputationGraph(gb.build());  //  FIXED
        net.init();
        net.setListeners(new ScoreIterationListener(200));
        // Data
        DataSetIterator trainIter = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator testIter = new MnistDataSetIterator(100, false, 12345);
        double noiseStd = noiseStdFromSNRdB(channelSNRdB);
        System.out.println("Training DeepJSCC-simple model...");
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            trainIter.reset();
            int batchIdx = 0;
            double snrSum = 0.0;
            int snrCount = 0;
            while (trainIter.hasNext()) {
                DataSet ds = trainIter.next();
                INDArray features = ds.getFeatures();
                INDArray labels = ds.getLabels();
                INDArray convInput = features.reshape(features.size(0), 1, 28, 28);
                //Random SNR between 2 and 20 dB
                double snrDb = 2 + Math.random() * 18;
                double dnoiseStd = noiseStdFromSNRdB(snrDb);
                snrSum += snrDb;
                snrCount++;
                INDArray noisyInput = convInput.add(Nd4j.randn(convInput.shape()).muli(dnoiseStd)).dup();
                INDArray reconTargetFlat = convInput.reshape(convInput.size(0), 28 * 28);

                net.fit(new INDArray[]{noisyInput}, new INDArray[]{reconTargetFlat, labels});

                if (batchIdx % 200 == 0) {
                    Map<String, INDArray> acts = net.feedForward(Collections.singletonMap("input", convInput).isEmpty(), false, false);
                    INDArray encoderFeatures = acts.get("features");
                    double rrProxy = computeRateReductionProxy(encoderFeatures, labels);
                    INDArray[] outputs = net.output(noisyInput);
                    INDArray reconOut = outputs[0];
                    double psnr = estimatePSNR(reconTargetFlat, reconOut.reshape(reconTargetFlat.shape()));

                    System.out.printf("Epoch %d batch %d: proxyΔR=%.4f, PSNR≈%.2f dB%n",
                            epoch + 1, batchIdx, rrProxy, psnr);
                }
                batchIdx++;
            }
            double avgSNR = snrSum / Math.max(1, snrCount);
            System.out.printf("Epoch %d complete. Average SNR ≈ %.2f dB%n", epoch + 1, avgSNR);

            evaluate(net, testIter, noiseStdFromSNRdB(10.0)); // evaluate at fixed test SNR = 10 dB
        }
        System.out.println(" Training complete.");
        visualizeOneExample(net, noiseStd);
        Map<Double, Double> psnrResults = new LinkedHashMap<>();
        Map<Double, Double> accResults = new LinkedHashMap<>();

        double[] snrList = {2, 5, 9, 12, 15, 18};  // test SNRs in dB
        for (double snr : snrList) {
            double currNoiseStd  = noiseStdFromSNRdB(snr);
            double psnrVal = evaluatePSNR(net, testIter, currNoiseStd );
            double accVal = evaluateAccuracy(net, testIter, currNoiseStd );

            psnrResults.put(snr, psnrVal);
            accResults.put(snr, accVal);
        }
        PlotResults.plotPSNRandAccuracy(psnrResults, accResults, "MNIST");
    }
    //Utility functions
    // L2 normalize each feature vector
    static INDArray powerNormalize(INDArray X) {
        INDArray sq = X.mul(X);
        INDArray sums = sq.sum(1); // [batch,1]
        INDArray denom = Transforms.sqrt(sums.add(1e-8), true).reshape(sums.size(0), 1); // FIXED
        return X.divColumnVector(denom);
    }
    // Simple proxy for rate-reduction term
    static double computeRateReductionProxy(INDArray features, INDArray oneHotLabels) {
        int N = (int) features.size(0);
        INDArray meanAll = features.mean(0);
        INDArray centeredAll = features.subRowVector(meanAll);
        INDArray covAll = centeredAll.transpose().mmul(centeredAll).div(N);
        double totalVar = trace(covAll); //  uses helper trace()
        int classes = (int) oneHotLabels.size(1);
        double sumClassVar = 0.0;
        for (int c = 0; c < classes; c++) {
            List<INDArray> classSamples = new ArrayList<>();
            for (int i = 0; i < N; i++) {
                if (oneHotLabels.getDouble(i, c) > 0.5) {
                    classSamples.add(features.getRow(i));
                }
            }
            if (classSamples.isEmpty()) continue;
            INDArray stacked = Nd4j.vstack(classSamples);
            INDArray meanC = stacked.mean(0);
            INDArray centeredC = stacked.subRowVector(meanC);
            INDArray covC = centeredC.transpose().mmul(centeredC).div(stacked.size(0));
            sumClassVar += trace(covC);
        }
        return totalVar - sumClassVar;
    }
    static double estimatePSNR(INDArray clean, INDArray recon) {
        INDArray err = clean.sub(recon);
        double mse = err.mul(err).meanNumber().doubleValue();
        if (mse <= 1e-10) return 100.0;
        return 10.0 * Math.log10(1.0 / mse);
    }
    // Compute trace (sum of diagonal elements)
    static double trace(INDArray matrix) {
        int n = (int) Math.min(matrix.rows(), matrix.columns());
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += matrix.getDouble(i, i);
        }
        return sum;
    }
    static void evaluate(ComputationGraph net, DataSetIterator testIter, double noiseStd) throws Exception {
        testIter.reset();
        int total = 0, correct = 0;
        double psnrSum = 0.0;
        int count = 0;
        while (testIter.hasNext()) {
            DataSet ds = testIter.next();
            INDArray convInput = ds.getFeatures().reshape(ds.getFeatures().size(0), 1, 28, 28);
            INDArray noisyInput = convInput.add(Nd4j.randn(convInput.shape()).muli(noiseStd));
            INDArray[] outs = net.output(noisyInput);
            INDArray reconFlat = outs[0];
            INDArray preds = outs[1];
            INDArray labels = ds.getLabels();
            double psnrBatch = estimatePSNR(convInput.reshape(convInput.size(0), 28 * 28), reconFlat);
            psnrSum += psnrBatch;
            count++;
            INDArray predIdx = preds.argMax(1);
            INDArray labelIdx = labels.argMax(1);
            for (int i = 0; i < predIdx.length(); i++) {
                if (predIdx.getInt(i) == labelIdx.getInt(i)) correct++;
                total++;
            }
        }
        double avgPsnr = psnrSum / Math.max(1, count);
        double acc = 100.0 * correct / (double) total;
        System.out.printf("=== Eval: PSNR≈%.2f dB, Accuracy=%.2f%% (%d samples) ===%n",
                avgPsnr, acc, total);
    }
    static double evaluatePSNR(ComputationGraph net, DataSetIterator testIter, double noiseStd) throws Exception {
        testIter.reset();
        double totalPsnr = 0.0;
        int count = 0;
        while (testIter.hasNext()) {
            var ds = testIter.next();
            var convInput = ds.getFeatures().reshape(ds.getFeatures().size(0), 1, 28, 28);
            var noisyInput = convInput.add(Nd4j.randn(convInput.shape()).muli(noiseStd));
            var outputs = net.output(noisyInput);
            var reconFlat = outputs[0];
            totalPsnr += estimatePSNR(convInput.reshape(convInput.size(0), 28 * 28), reconFlat);
            count++;
        }
        return totalPsnr / count;
    }
    static double evaluateAccuracy(ComputationGraph net, DataSetIterator testIter, double noiseStd) throws Exception {
        testIter.reset();
        int correct = 0, total = 0;
        while (testIter.hasNext()) {
            var ds = testIter.next();
            var convInput = ds.getFeatures().reshape(ds.getFeatures().size(0), 1, 28, 28);
            var noisyInput = convInput.add(Nd4j.randn(convInput.shape()).muli(noiseStd));
            var outputs = net.output(noisyInput);
            var preds = outputs[1];
            var labels = ds.getLabels();
            var predIdx = preds.argMax(1);
            var labelIdx = labels.argMax(1);
            for (int i = 0; i < predIdx.length(); i++) {
                if (predIdx.getInt(i) == labelIdx.getInt(i)) correct++;
                total++;
            }
        }
        return (100.0 * correct) / total;
    }
    // === Visualize one sample (original, noisy, reconstructed) ===
    static void visualizeOneExample(ComputationGraph net, double noiseStd) throws Exception {
        DataSetIterator testIter = new MnistDataSetIterator(1, false, 12345);
        DataSet ds = testIter.next();

        INDArray clean = ds.getFeatures().reshape(1, 1, 28, 28);
        INDArray noisy = clean.add(Nd4j.randn(clean.shape()).muli(noiseStd));
        INDArray[] outputs = net.output(noisy);
        INDArray recon = outputs[0].reshape(28, 28);

        new File("results").mkdirs();
        saveImage(clean.reshape(28, 28), "results/original.png");
        saveImage(noisy.reshape(28, 28), "results/noisy.png");
        saveImage(recon, "results/reconstructed.png");

        System.out.println(" Saved example images: original.png, noisy.png, reconstructed.png in /results/");
    }
    // Utility to save a grayscale 28x28 INDArray as an image
    static void saveImage(INDArray tensor, String filePath) throws IOException {
        INDArray unnormalized = tensor.mul(255.0);
        int height = (int) unnormalized.size(0);
        int width = (int) unnormalized.size(1);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = (int) Math.max(0, Math.min(255, unnormalized.getDouble(y, x)));
                int rgb = (gray << 16) | (gray << 8) | gray;
                image.setRGB(x, y, rgb);
            }
        }
        ImageIO.write(image, "png", new File(filePath));
    }
}
