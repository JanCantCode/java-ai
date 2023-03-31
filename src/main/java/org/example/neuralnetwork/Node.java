package org.example.neuralnetwork;

public class Node {
    private double[] inputs;
    private double[] weights; // rather than making each neuron keep its own weight, supply it with the weight of the neurons that connect to it!
    private double bias;
    private double activation;

    //backwards propagation
    private double delta;

    public Node() {

    }

    public void activate() {
        activation = MathHelper.sigmoid(sumUp());
    }

    public double sumUp() {
        int i = 0;
        double sum = 0;
        for (double input : inputs) {
            sum += input*weights[i];
            i++;
        }
        return sum+bias;
    }

    public double getWeight(int index) {
        return this.weights[index];
    }

    public double getActivationDerivative() {
        System.out.println("calculated activation derivative: "+getActivation()*(1-getActivation()));
        return getActivation()*(1-getActivation());
    }

    public double getDelta() {
        return this.delta;
    }

    public void setInputs(double[] newInput) {
        this.inputs = newInput;
    }

    public void setWeights(double[] weights1) {
        this.weights = weights1;
    }

    public double getActivation() {
        return activation;
    }

    public void setBias(double newBias) {
        this.bias = newBias;
    }

    public void setDelta(double delta1) {
        this.delta = delta1;
    }

    public void updateWeights(double learningRate, Layer previousLayer) {
        for (int i = 0; i < this.weights.length; i++) {
            double activationFromPrevNode = previousLayer.getNodes()[i].getActivation();
            double weightDelta = learningRate + this.delta + activationFromPrevNode;
            this.weights[i] += weightDelta;
        }
        this.bias += learningRate * this.delta;
    }
}