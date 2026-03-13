package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Channel {

    /**
     * Implements Power Normalization (Equation 5 from the paper) using a robust method.
     */
    private static INDArray powerNormalize(INDArray features) {
        // --- THE FIX: Flatten to 2D to simplify math and avoid broadcasting errors ---
        long batchSize = features.shape()[0];
        long featureSize = features.length() / batchSize; // Total features per sample (200)

        // Reshape from [256, 200, 1, 1] to [256, 200]
        INDArray flatFeatures = features.reshape(batchSize, featureSize);

        double b = featureSize / 2.0;
        double epsilon = 1e-8;

        // Calculate norm for each row (each sample in the batch)
        INDArray norm = flatFeatures.norm2(1); // norm2 along dimension 1

        // Divide each row by its norm. diviColumnVector is for in-place division.
        flatFeatures.diviColumnVector(norm.add(epsilon));

        // Scale by sqrt(b)
        flatFeatures.muli(Math.sqrt(b)); // muli is for in-place multiplication

        // Reshape back to the original 4D shape
        return flatFeatures.reshape(features.shape());
    }

    /**
     * Simulates the AWGN channel.
     */
    public static INDArray simulateChannel(INDArray features, double snrDb) {
        INDArray normalizedFeatures = powerNormalize(features);

        double snrLinear = Math.pow(10, snrDb / 10.0);
        double noiseVariance = 1 / snrLinear;

        INDArray noise = Nd4j.randn(normalizedFeatures.shape()).mul(Math.sqrt(noiseVariance));
        return normalizedFeatures.add(noise);
    }
}