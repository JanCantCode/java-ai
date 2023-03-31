package org.example.neuralnetwork;

public class Layer {
    private int neurons;
    private double[] input;
    private final Node[] nodes;
    private double[] output;
    private final LayerType type;

    public Layer(Node[] nodes1, LayerType layerType) {
        this.nodes = nodes1;
        this.type = layerType;
    }

    public void feed() {
        for (Node node : this.nodes) {
            double[] neuronInputs = new double[this.input.length];
            int currentValueIteration = 0;

            for (double inputValue : this.input) {
                neuronInputs[currentValueIteration] = inputValue;
                currentValueIteration++;
            }
            node.setInputs(neuronInputs);
            node.activate();
        }

        double[] outputs = new double[this.nodes.length];
        int currentNode = 0;
        for (Node node : this.nodes) {
            outputs[currentNode] = node.getActivation();
            currentNode++;
        }
        this.output = outputs;
    }

    public void setInputs(double[] inputs) {
        this.input = inputs;
    }

    public double[] getOutputs() {
        return output;
    }


    public void setAllWeights(double weight, int nodesInPreviousLayer) {
        double[] newWeights = new double[nodesInPreviousLayer];

        for (int i = 0; i < nodesInPreviousLayer; i++) {
            newWeights[i] = weight;
        }

        for (Node node : this.nodes) {
            node.setWeights(newWeights);
        }
    }

    public void setAllBiases(double bias) {
        for (Node node : this.nodes) {
            node.setBias(bias);
        }
    }

    public Node[] getNodes() {
        return this.nodes;
    }

    public void calculateError(double[] expected, Layer nextLayer) {
        switch (this.type) {
            case OUTPUT -> {
                int i = 0;
                for (Node node : this.nodes) {
                    double nodeOutput = node.getActivation();

                    double error = (expected[i] - nodeOutput) * node.getActivationDerivative();
                    node.setDelta(error);

                    i++;
                }
            }

            case HIDDEN -> {
                int i = 0;

                for (Node node : this.nodes) {
                    double errSum = 0.0;

                    for (int j = 0; j < nextLayer.nodes.length; j++) {
                        Node nextLayerNode = nextLayer.nodes[j];
                        double nextNodeError = nextLayerNode.getActivation();

                        double weight = nextLayerNode.getWeight(i);

                        errSum += nextNodeError * weight;
                    }

                    double error = errSum * node.getActivation() * (1 - node.getActivation());
                    node.setDelta(error);

                    i++;
                }
            }
        }
    }

    public void setOutputError(double[] expected) {
        int i = 0;

        for (Node node : this.nodes) {
            double output = node.getActivation();
            double error = expected[i] - output;
            node.setDelta(error);
            i++;
        }
    }
}
