package org.example.neuralnetwork;

public class MathHelper {
    public static double sigmoid(double input) {
        return 1 / (1+Math.exp(-1*input));
    }
}
