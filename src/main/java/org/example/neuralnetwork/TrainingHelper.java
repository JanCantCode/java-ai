package org.example.neuralnetwork;

public class TrainingHelper {
    public static double[][] toOneNeuronInput(double[] input) {
        double[][] result = new double[input.length][1];
        for (int i = 0; i < input.length; i++) {
            result[i][0] = input[i];
        }
        return result;
    }


    public static double getRandomDouble(double min, double max) {
        return min + (max - min) * Math.random();
    }

}
