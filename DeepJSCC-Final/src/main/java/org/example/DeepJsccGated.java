package org.example;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
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
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

public class DeepJsccGated {

    static final int batchSize = 128;
    static final int numEpochs = 5;
    static final double learningRate = 1.5e-4;
    static final Random rng = new Random(12345);

    // Randomize SNR per batch in [2, 15] dB
    static double sampleSNR() {
        return 2.0 + rng.nextDouble() * 13.0;
    }

    static double noiseStdFromSNRdB(double snrDb) {
        double snr = Math.pow(10, snrDb / 10.0);
        return Math.sqrt(1.0 / snr);
    }

    public static void main(String[] args) throws Exception {
        ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
                .seed(12345)
                // --- Optimizer with learning-rate schedule + gradient clipping ---
                .updater(new Adam.Builder()
                        .learningRateSchedule(new MapSchedule(
                                ScheduleType.EPOCH,
                                Map.of(
                                        0, 1.5e-4,   // Epochs 0–1
                                        2, 1.0e-4,   // After 2 epochs
                                        4, 7.5e-5    // After 4 epochs
                                )
                        ))
                        .build())
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .weightInit(WeightInit.XAVIER)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .graphBuilder()
                .addInputs("input", "snr_input") // Add SNR input for gating MLP
                .setInputTypes(InputType.convolutionalFlat(28, 28, 1), InputType.feedForward(1))

                // --- Encoder ---
                .addLayer("enc_c1", new ConvolutionLayer.Builder(4, 4).stride(2, 2)
                        .nOut(32).activation(Activation.LEAKYRELU).build(), "input")
                .addLayer("enc_c2", new ConvolutionLayer.Builder(4, 4).stride(2, 2)
                        .nOut(64).activation(Activation.LEAKYRELU).build(), "enc_c1")
                .addLayer("enc_flat", new DenseLayer.Builder().nOut(256)
                        .activation(Activation.LEAKYRELU).build(), "enc_c2")
                .addLayer("features", new DenseLayer.Builder().nOut(200)
                        .activation(Activation.IDENTITY).build(), "enc_flat")

                // --- Gating Network (takes SNR input, outputs gate vector) ---
                .addLayer("gate_fc1", new DenseLayer.Builder().nOut(64)
                        .activation(Activation.RELU).build(), "snr_input")
                .addLayer("gate_fc2", new DenseLayer.Builder().nOut(200)
                        .activation(Activation.SIGMOID).build(), "gate_fc1")

                // Element-wise multiply gate * features -> "gated_features"
                .addVertex("gated_features", new ElementWiseVertex(ElementWiseVertex.Op.Product),
                        "features", "gate_fc2")

                // --- Decoder ---
                .addLayer("dec_fc", new DenseLayer.Builder().nOut(256)
                        .activation(Activation.RELU).build(), "gated_features")
                .addLayer("dec_out_flat", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID).nOut(28 * 28).build(), "dec_fc")

                // --- Classifier ---
                .addLayer("clf_fc1", new DenseLayer.Builder().nOut(128)
                        .activation(Activation.RELU).build(), "gated_features")
                .addLayer("classifier", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nOut(10).build(), "clf_fc1")

                .setOutputs("dec_out_flat", "classifier");

        ComputationGraph net = new ComputationGraph(gb.build());
        net.init();
        net.setListeners(new ScoreIterationListener(200));

        DataSetIterator trainIter = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator testIter = new MnistDataSetIterator(100, false, 12345);

        System.out.println("Training Adaptive Gated JSCC model...");
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            trainIter.reset();
            int batchIdx = 0;

            while (trainIter.hasNext()) {
                DataSet ds = trainIter.next();
                INDArray features = ds.getFeatures().reshape(ds.getFeatures().size(0), 1, 28, 28);
                INDArray labels = ds.getLabels();

                double snrDb = sampleSNR();
                double noiseStd = noiseStdFromSNRdB(snrDb);

                // Simulate channel
                INDArray noisy = features.add(Nd4j.randn(features.shape()).muli(noiseStd)).dup();

                // SNR input tensor (batch,1)
                INDArray snrTensor = Nd4j.valueArrayOf(features.size(0), 1, snrDb / 15.0); // normalize 0–1

                INDArray reconTargetFlat = features.reshape(features.size(0), 28 * 28);
                net.fit(new INDArray[]{noisy, snrTensor}, new INDArray[]{reconTargetFlat, labels});

                if (batchIdx % 200 == 0) {
                    System.out.printf("Epoch %d batch %d: SNR=%.2f dB%n", epoch + 1, batchIdx, snrDb);
                }
                batchIdx++;
            }

            evaluate(net, testIter);
        }
        System.out.println("✅ Training complete (Adaptive Gated JSCC).");

        // Optional evaluation over SNR range
        Map<Double, Double> psnrResults = new LinkedHashMap<>();
        Map<Double, Double> accResults = new LinkedHashMap<>();
        double[] snrList = {4, 8, 12, 16, 20};

        for (double snr : snrList) {
            double psnrVal = evaluatePSNR_Gated(net, testIter, snr);
            double accVal = evaluateAccuracy_Gated(net, testIter, snr);
            psnrResults.put(snr, psnrVal);
            accResults.put(snr, accVal);
        }

        // Optionally plot results using your existing class
        PlotResultsGated.plotGatedResults(psnrResults, accResults, "MNIST");

        // === Save example images ===
        visualizeOneExample(net, 10.0); // visualize at 10 dB SNR
    }

    // ===================== EVALUATION FUNCTIONS =====================
    static void evaluate(ComputationGraph net, DataSetIterator testIter) throws Exception {
        testIter.reset();
        int total = 0, correct = 0;
        double psnrSum = 0.0;
        int count = 0;
        while (testIter.hasNext()) {
            DataSet ds = testIter.next();
            INDArray input = ds.getFeatures().reshape(ds.getFeatures().size(0), 1, 28, 28);
            INDArray labels = ds.getLabels();

            double snrDb = 10.0; // fixed test SNR
            double noiseStd = noiseStdFromSNRdB(snrDb);
            INDArray noisy = input.add(Nd4j.randn(input.shape()).muli(noiseStd));
            INDArray snrTensor = Nd4j.valueArrayOf(input.size(0), 1, snrDb / 15.0);

            INDArray[] outs = net.output(noisy, snrTensor);
            INDArray recon = outs[0];
            INDArray preds = outs[1];

            double psnr = estimatePSNR(input.reshape(input.size(0), 28 * 28), recon);
            psnrSum += psnr; count++;

            INDArray predIdx = preds.argMax(1);
            INDArray labelIdx = labels.argMax(1);
            for (int i = 0; i < predIdx.length(); i++) {
                if (predIdx.getInt(i) == labelIdx.getInt(i)) correct++;
                total++;
            }
        }
        double avgPSNR = psnrSum / Math.max(count, 1);
        double acc = 100.0 * correct / total;
        System.out.printf("=== Eval: PSNR≈%.2f dB, Accuracy=%.2f%% ===%n", avgPSNR, acc);
    }

    static double evaluatePSNR_Gated(ComputationGraph net, DataSetIterator testIter, double snrDb) throws Exception {
        testIter.reset();
        double psnrSum = 0.0;
        int count = 0;
        double noiseStd = noiseStdFromSNRdB(snrDb);

        while (testIter.hasNext()) {
            DataSet ds = testIter.next();
            INDArray input = ds.getFeatures().reshape(ds.getFeatures().size(0), 1, 28, 28);
            INDArray noisy = input.add(Nd4j.randn(input.shape()).muli(noiseStd));
            INDArray snrTensor = Nd4j.valueArrayOf(input.size(0), 1, snrDb / 15.0);

            INDArray[] outs = net.output(new INDArray[]{noisy, snrTensor});
            INDArray recon = outs[0];

            INDArray cleanFlat = input.reshape(input.size(0), 28 * 28);
            INDArray err = cleanFlat.sub(recon);
            double mse = err.mul(err).meanNumber().doubleValue();
            double psnr = (mse <= 1e-10) ? 100.0 : 10.0 * Math.log10(1.0 / mse);

            psnrSum += psnr;
            count++;
        }
        return psnrSum / Math.max(1, count);
    }

    static double evaluateAccuracy_Gated(ComputationGraph net, DataSetIterator testIter, double snrDb) throws Exception {
        testIter.reset();
        int total = 0, correct = 0;
        double noiseStd = noiseStdFromSNRdB(snrDb);

        while (testIter.hasNext()) {
            DataSet ds = testIter.next();
            INDArray input = ds.getFeatures().reshape(ds.getFeatures().size(0), 1, 28, 28);
            INDArray labels = ds.getLabels();
            INDArray noisy = input.add(Nd4j.randn(input.shape()).muli(noiseStd));
            INDArray snrTensor = Nd4j.valueArrayOf(input.size(0), 1, snrDb / 15.0);

            INDArray[] outs = net.output(new INDArray[]{noisy, snrTensor});
            INDArray preds = outs[1];

            INDArray predIdx = preds.argMax(1);
            INDArray labelIdx = labels.argMax(1);
            for (int i = 0; i < predIdx.length(); i++) {
                if (predIdx.getInt(i) == labelIdx.getInt(i)) correct++;
                total++;
            }
        }
        return 100.0 * correct / total;
    }

    static double estimatePSNR(INDArray clean, INDArray recon) {
        INDArray err = clean.sub(recon);
        double mse = err.mul(err).meanNumber().doubleValue();
        if (mse <= 1e-10) return 100.0;
        return 10.0 * Math.log10(1.0 / mse);
    }

    // ===================== IMAGE VISUALIZATION =====================
    static void saveImage(INDArray tensor, String filePath) throws IOException {
        INDArray img = tensor;
        if (tensor.length() == 28 * 28) img = tensor.reshape(28, 28);
        int height = (int) img.size(0), width = (int) img.size(1);
        BufferedImage buffered = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double val = img.getDouble(y, x);
                int gray = (int) Math.round(Math.max(0.0, Math.min(1.0, val)) * 255.0);
                int rgb = (gray << 16) | (gray << 8) | gray;
                buffered.setRGB(x, y, rgb);
            }
        }
        File out = new File(filePath);
        out.getParentFile().mkdirs();
        ImageIO.write(buffered, "png", out);
        System.out.println("Saved: " + filePath);
    }

    static void visualizeOneExample(ComputationGraph net, double exampleSNRdB) {
        try {
            DataSetIterator testIter = new MnistDataSetIterator(1, false, 12345);
            DataSet ds = testIter.next();
            INDArray clean = ds.getFeatures().reshape(1, 1, 28, 28);
            double noiseStd = noiseStdFromSNRdB(exampleSNRdB);
            INDArray noisy = clean.add(Nd4j.randn(clean.shape()).muli(noiseStd));
            INDArray snrTensor = Nd4j.valueArrayOf(1, 1, exampleSNRdB / 15.0);
            INDArray[] outs = net.output(new INDArray[]{noisy, snrTensor});
            INDArray recon = outs[0].reshape(28, 28);

            saveImage(clean.reshape(28, 28), "results/example_original.png");
            saveImage(noisy.reshape(28, 28), "results/example_noisy.png");
            saveImage(recon, "results/example_reconstructed.png");

            System.out.printf("Visualization done (SNR=%.2f dB) — saved in 'results/' folder.%n", exampleSNRdB);
        } catch (Exception e) {
            System.err.println("Visualization failed: " + e.getMessage());
        }
    }
}
