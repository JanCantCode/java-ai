package org.example.neuralnetwork;

public class Network {
    private Layer[] layers;
    private double[] inputLayer;
    public Network() {

    }

    public void setLayers(Layer[] layers1) {
        this.layers = layers1;
    }

    public double[] feed() {
        this.layers[0].setInputs(inputLayer);
        this.layers[0].feed();
        for (int i = 1; i < this.layers.length; i++) {
            this.layers[i].setInputs(this.layers[i-1].getOutputs());
            this.layers[i].feed();
        }
        return this.layers[this.layers.length-1].getOutputs();
    }

    public void setInputData(double[] inputs) {
        this.inputLayer = inputs;
    }

    public double[] getOutputsValues() {
        return this.layers[this.layers.length-1].getOutputs();
    }

    public Layer[] getLayers() {
        return this.layers;
    }
}
