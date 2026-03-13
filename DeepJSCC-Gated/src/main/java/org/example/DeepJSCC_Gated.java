package org.example;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class DeepJSCC_Gated {

    public static void main(String[] args) throws IOException {
        System.out.println("--- Running Gated Pipeline (Algorithm 2) ---");
        runGatedPipeline();
    }

    public static void saveImage(INDArray tensor, String filePath) throws IOException {
        INDArray unnormalized = tensor.mul(0.5).add(0.5).mul(255);
        int height = (int) unnormalized.shape()[0];
        int width = (int) unnormalized.shape()[1];
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int grayValue = unnormalized.getInt(y, x);
                int rgb = (grayValue << 16) | (grayValue << 8) | grayValue;
                image.setRGB(x, y, rgb);
            }
        }
        File outputFile = new File(filePath);
        ImageIO.write(image, "png", outputFile);
        System.out.println("Saved image to: " + filePath);
    }

    public static void runGatedPipeline() throws IOException {
        int batchSize = 256;
        double gateThreshold = 0.5;

        Random rand = new Random();
        double snrDb = rand.nextDouble() * 20;

        System.out.println("Loading MNIST dataset...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);

        System.out.println("\nBuilding all models...");
        ComputationGraph encoder = Models.createEncoder();
        ComputationGraph decoder = Models.createDecoder();
        ComputationGraph gatedNet = Models.createGatedNet();
        System.out.println("Models built successfully!");

        if (mnistTrain.hasNext()) {
            INDArray originalImages = mnistTrain.next().getFeatures().reshape(batchSize, 1, 28, 28);
            INDArray firstImage = originalImages.slice(0).reshape(1, 1, 28, 28);

            // 1. Encode image
            INDArray encodedFeatures = encoder.outputSingle(firstImage); // Shape: [1, 200, 1, 1]

            // 2. Get mask from Gated Net
            double snrLinear = Math.pow(10, snrDb / 10.0);
            double noiseVariance = 1 / snrLinear;
            INDArray noiseVarInput = Nd4j.create(new double[]{noiseVariance}).reshape(1, 1);
            INDArray mask = gatedNet.outputSingle(noiseVarInput); // Shape: [1, 200]
            INDArray binaryMask = mask.gt(gateThreshold);

            // --- THE FIX IS HERE: Reshape for robust multiplication ---
            // 3. Flatten features from [1, 200, 1, 1] to [1, 200]
            long featureSize = encodedFeatures.size(1);
            INDArray flatEncoded = encodedFeatures.reshape(1, featureSize);

            // 4. Apply the mask to the flattened features
            INDArray flatFiltered = flatEncoded.mul(binaryMask);

            // 5. Reshape the filtered features back to the original 4D shape
            INDArray filteredFeatures = flatFiltered.reshape(encodedFeatures.shape());
            // --- END OF FIX ---

            // 6. Send through the channel
            System.out.println("\nSimulating channel with random SNR = " + String.format("%.2f", snrDb) + " dB...");
            INDArray noisyFeatures = Channel.simulateChannel(filteredFeatures, snrDb);

            // 7. Decode the result
            INDArray reconstructedImage = decoder.outputSingle(noisyFeatures);

            System.out.println("\n--- Saving Images for Gated Pipeline ---");
            new File("results_gated").mkdirs();

            saveImage(firstImage.reshape(28, 28), "results_gated/original.png");
            saveImage(filteredFeatures.reshape(10, 20), "results_gated/filtered_features.png");
            saveImage(reconstructedImage.reshape(28, 28), "results_gated/reconstructed.png");

            System.out.println("\nAlgorithm 2 pipeline finished. Check the 'results_gated' folder.");
        }
    }
}