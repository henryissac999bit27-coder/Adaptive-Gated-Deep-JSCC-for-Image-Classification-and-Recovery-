package org.example;

import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;

public class Models {
    private static final long seed = 12345;
    private static final int channels = 1;
    private static final int height = 28;
    private static final int width = 28;
    private static final int encodedDim = 200;

    public static ComputationGraph createEncoder() {
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed).weightInit(WeightInit.XAVIER).updater(new Adam(1.5e-4))
                .graphBuilder()
                .addInputs("inputImage")
                .setInputTypes(InputType.convolutional(height, width, channels))
                .addLayer("e_conv1", new ConvolutionLayer.Builder(4, 4).stride(2, 2).padding(1, 1).nOut(64).build(), "inputImage")
                .addLayer("e_act1", new ActivationLayer(Activation.LEAKYRELU), "e_conv1")
                .addLayer("e_conv2", new ConvolutionLayer.Builder(4, 4).stride(2, 2).padding(1, 1).nOut(128).build(), "e_act1")
                .addLayer("e_bn2", new BatchNormalization(), "e_conv2")
                .addLayer("e_act2", new ActivationLayer(Activation.LEAKYRELU), "e_bn2")
                .addLayer("e_conv3", new ConvolutionLayer.Builder(3, 3).stride(2, 2).padding(1, 1).nOut(256).build(), "e_act2")
                .addLayer("e_bn3", new BatchNormalization(), "e_conv3")
                .addLayer("e_act3", new ActivationLayer(Activation.LEAKYRELU), "e_bn3")
                .addLayer("e_conv4_features", new ConvolutionLayer.Builder(4, 4).stride(1, 1).padding(0, 0).nOut(encodedDim).build(), "e_act3")
                .setOutputs("e_conv4_features")
                .build();

        ComputationGraph model = new ComputationGraph(config);
        model.init();
        return model;// ... Paste the createEncoder() method code from your previous project here ...
    }

    public static ComputationGraph createDecoder() {
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed).weightInit(WeightInit.XAVIER).updater(new Adam(1.5e-4))
                .graphBuilder()
                .addInputs("decoderInput")
                .setInputTypes(InputType.convolutional(1, 1, encodedDim))
                .addLayer("d_deconv1", new Deconvolution2D.Builder(4, 4).stride(1, 1).padding(0, 0).nOut(256).build(), "decoderInput")
                .addLayer("d_bn1", new BatchNormalization(), "d_deconv1")
                .addLayer("d_act1", new ActivationLayer(Activation.RELU), "d_bn1")
                .addLayer("d_deconv2", new Deconvolution2D.Builder(3, 3).stride(2, 2).padding(1, 1).nOut(128).build(), "d_act1")
                .addLayer("d_bn2", new BatchNormalization(), "d_deconv2")
                .addLayer("d_act2", new ActivationLayer(Activation.RELU), "d_bn2")
                .addLayer("d_deconv3", new Deconvolution2D.Builder(4, 4).stride(2, 2).padding(1, 1).nOut(64).build(), "d_act2")
                .addLayer("d_bn3", new BatchNormalization(), "d_deconv3")
                .addLayer("d_act3", new ActivationLayer(Activation.RELU), "d_bn3")
                .addLayer("d_deconv4_output", new Deconvolution2D.Builder(4, 4).stride(2, 2).padding(1, 1).nOut(channels).activation(Activation.TANH).build(), "d_act3")
                .setOutputs("d_deconv4_output")
                .build();

        ComputationGraph model = new ComputationGraph(config);
        model.init();
        return model;// ... Paste the createDecoder() method code from your previous project here ...
    }

    public static ComputationGraph createGatedNet() {
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed).weightInit(WeightInit.XAVIER).updater(new Adam(1.5e-4))
                .graphBuilder()
                .addInputs("noise_variance")
                .setInputTypes(InputType.feedForward(1))
                .addLayer("gate_dense1", new DenseLayer.Builder().nIn(1).nOut(128).activation(Activation.RELU).build(), "noise_variance")
                .addLayer("gate_dense2", new DenseLayer.Builder().nIn(128).nOut(128).activation(Activation.RELU).build(), "gate_dense1")
                .addLayer("gate_output", new DenseLayer.Builder().nIn(128).nOut(encodedDim).activation(Activation.SIGMOID).build(), "gate_dense2")
                .setOutputs("gate_output")
                .build();

        ComputationGraph model = new ComputationGraph(config);
        model.init();
        return model;
    }
}