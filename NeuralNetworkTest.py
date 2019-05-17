from scipy.special import expit
import numpy


class NeuralNetwork:

    def __init__(self, in_nodes_number, hi_nodes_number, out_nodes_number, learning_rate):
        self.inputNodesNumber = in_nodes_number
        self.hiddenNodesNumber = hi_nodes_number
        self.outputNodesNumber = out_nodes_number
        self.learningRate = learning_rate
        self.weight_inputlayer_hiddenlayer = numpy.random.normal(0.0, pow(self.hiddenNodesNumber, -0.5), (self.hiddenNodesNumber, self.inputNodesNumber))
        self.weight_hiddenlayer_outputlayer = numpy.random.normal(0.0,pow(self.outputNodesNumber, -0.5), (self.outputNodesNumber, self.hiddenNodesNumber))

    def train(self, input_list, target_list):
        input_array = numpy.array(input_list, ndmin=2).T
        target_array = numpy.array(target_list, ndmin=2).T
        hidden_layer_input = numpy.dot(self.weight_inputlayer_hiddenlayer, input_array)
        hidden_layer_output = self.activate_function(hidden_layer_input)
        output_layer_input = numpy.dot(self.weight_hiddenlayer_outputlayer, hidden_layer_output)
        output_layer_output = self.activate_function(output_layer_input)
        output_layer_discrepancy = target_array - output_layer_output
        hidden_layer_discrepancy = numpy.dot(self.weight_hiddenlayer_outputlayer.T, output_layer_discrepancy)
        self.weight_hiddenlayer_outputlayer += self.learningRate * numpy.dot((output_layer_discrepancy * output_layer_output * (1.0 - output_layer_output)), numpy.transpose(hidden_layer_output))
        self.weight_inputlayer_hiddenlayer += self.learningRate * numpy.dot((hidden_layer_discrepancy * hidden_layer_output * (1.0 - hidden_layer_output)), numpy.transpose(input_array))
        pass


    def query(self, input_list):
        input_array = numpy.array(input_list, ndmin=2).T
        hidden_layer_input = numpy.dot(self.weight_inputlayer_hiddenlayer, input_array)
        hidden_layer_output = self.activate_function(hidden_layer_input)
        output_layer_input = numpy.dot(self.weight_hiddenlayer_outputlayer, hidden_layer_output)
        output_layer_output = self.activate_function(output_layer_input)
        return output_layer_output

    def activate_function(self, value):
        return expit(value)


inputNodesNumber = 784
hiddenNodesNumber = 100
outputNodesNumber = 10

learningRate = 0.3

network = NeuralNetwork(inputNodesNumber, hiddenNodesNumber, outputNodesNumber, learningRate)

training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for training_record in training_data_list:
    training_data = training_record.split(',')
    input_array = (numpy.asfarray(training_data[1:]) / 255.0 * 0.99) + 0.01
    target_array = numpy.zeros(outputNodesNumber) + 0.01
    target_array[int(training_data[0])] = 0.99
    network.train(input_array, target_array)
    pass

test_data_file = open("mnist_dataset/mnist_test_10.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []
for test_record in test_data_list:
    test_data = test_record.split(",")
    correct_label = int(test_data[0])
    print(correct_label, "correct label :")
    test_input_array = (numpy.asfarray(test_data[1:]) / 255.0 * 0.99) + 0.01
    test_output_array = network.query(test_input_array)
    output_label = numpy.argmax(test_output_array)
    print(output_label, "neural network label :")
    if output_label == correct_label:
        scorecard.append(1)
        pass
    else:
        scorecard.append(0)
        pass

    pass

scorecard_array = numpy.asarray(scorecard)
print("performance =  ", scorecard_array.sum() / scorecard_array.size)

