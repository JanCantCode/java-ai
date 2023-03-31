package org.example;

import org.example.neuralnetwork.*;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double[] trainingInput = new double[]{0.0, 0.4, 0.3, 1.0, 0.9, 0.8, 0.78, 0.1, 0.15, 0.35, 0.6, 0.78, 0.90, 0.58};
        double[] expectedOutput = new double[]{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};

        Node hiddenNode1 = new Node();
        hiddenNode1.setBias(0);
        hiddenNode1.setWeights(new double[]{TrainingHelper.getRandomDouble(-1, 1.0)});

        Node hiddenNode2 = new Node();
        hiddenNode2.setBias(0);
        hiddenNode2.setWeights(new double[]{TrainingHelper.getRandomDouble(-1.0, 1.0)});

        Node hiddenNode3 = new Node();
        hiddenNode3.setBias(0);
        hiddenNode3.setWeights(new double[]{TrainingHelper.getRandomDouble(-1.0, 1.0)});

        Node outputNode = new Node();
        outputNode.setBias(0);
        outputNode.setWeights(new double[]{TrainingHelper.getRandomDouble(-1.0, 1.0), TrainingHelper.getRandomDouble(0.0, 1.0), TrainingHelper.getRandomDouble(0.0, 1.0)});

        Layer hiddenLayer = new Layer(new Node[]{hiddenNode1, hiddenNode2, hiddenNode3}, LayerType.HIDDEN);
        Layer outputLayer = new Layer(new Node[]{outputNode}, LayerType.OUTPUT);

        Network neuralNetwork = new Network();

        neuralNetwork.setLayers(new Layer[]{hiddenLayer, outputLayer});

        double[][] formattedInput = TrainingHelper.toOneNeuronInput(trainingInput);
        double[][] formattedExpected = TrainingHelper.toOneNeuronInput(expectedOutput);
        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{100.0})));
        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{5.0})));
        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{6.0})));

        neuralNetwork.train(formattedInput, formattedExpected, 10, 0.5);

        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{100.0})));
        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{5.0})));
        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{6.0})));

    }
}