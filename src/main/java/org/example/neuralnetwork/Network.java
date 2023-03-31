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

    public double[] predict(double[] input) {
        this.setInputData(input);
        return feed();
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

    public void backwardsPropagation(double[] expected, double learningRate) {
        Layer outPutlayer = this.layers[this.layers.length-1];

        outPutlayer.setOutputError(expected);

        for (int i = this.layers.length-2; i >= 0; i--) {
            Layer currentLayer = this.layers[i];
            Layer nextLayer = this.layers[i+1];

            currentLayer.calculateError(expected, nextLayer);

            for (Node node : currentLayer.getNodes()) {
                node.updateWeights(learningRate, this.layers[i+1]);
            }
        }
    }

    public void train(double[][] inputs, double[][] expectedOutputs, int epochs, double learningRate) {
        for (int i = 0; i < epochs; i++) {
            double error = 0;
            for (int j = 0; j < inputs.length; j++) {

                double[] output = predict(inputs[j]);


                double[] expected = expectedOutputs[j];

                for (int k = 0; k < output.length; k++) {
                    error += Math.pow(expected[k]-output[k], 2);
                }

                backwardsPropagation(expected, learningRate);

                updateWeights(learningRate);

            }
            error /= inputs.length;
            System.out.println("Epoch "+(i+1)+" Error: "+error);
        }
    }

    public void updateWeights(double learningRate) {
        for (int i = this.layers.length-1; i > 0; i--) {
            Layer currentLayer = this.layers[i];
            Layer previousLayer = this.layers[i-1];

            for (Node node : currentLayer.getNodes()) {
                node.updateWeights(learningRate, previousLayer);
            }
        }
    }
}
