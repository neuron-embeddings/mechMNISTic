# MechMNISTic Interpretability

https://mechmnistic.streamlit.app/

## Motivation

Mechanistic interpretability aims to understand precisely how neural networks work by reverse engineering them into human understandable components.
Whilst there have been exciting results understanding specific portions of [vision](https://distill.pub/2020/circuits/) and [language](https://transformer-circuits.pub/) models, there are very few complete accounts of how even very simple models work.

Inspired by a [post](https://transformer-circuits.pub/2024/jan-update/index.html#mnist-sparse) from the interpretability team at Anthropic, I decided to try and build a tool that allows us to understand in detail the behaviour of small MLP networks trained to classify MNIST digits.

The tool provides two methods of exploration - understanding the behaviour of individual neurons, and understanding the behaviour of the model on individual examples.

## Understanding Neurons

You can explore the behaviour of individual neurons in the model. For each neuron, I visualise the input and output weights, and show the dataset examples that most activate the neuron, as well as the parts of the examples to which the neuron is responding. 

To do this, I compute the Hadamard (element-wise) product of a dataset example and the neuron's input weights and visualise the result. This directly shows what the input weights are detecting, and provides a representation that I use to cluster the dataset examples into [feature clusters](https://github.com/alexjfoote/feature-clusters).

Each cluster captures a distinct feature to which the neuron responds. The common pattern is that clusters consist of a particular way of writing a digit, and a particular neuron often responds to a variety of ways of writing a digit.

Applying this tool to an MLP with 1 hidden layer and 64 neurons, I find that the neurons appear to look for patches of light or dark pixels. For example, some neurons will respond to a vertical patch of light pixels down the centre of the image, and might increase the probability of predicting a 1 or a 7 and decrease the probability of predicting a 0. Neurons seem to typically look for 2 or 3 patches, and activate on 2-4 different digits, increasing the probability of those and decreasing the probability of others.

## Understanding Classifications

You can also explore the behaviour of the model on individual examples. You can either draw your own example, or select one from the MNIST test set. 
I show the output logits and probabilities from the model on the example. I then visualise the output weights and the hidden layer activations, as well as the product of the activation vector with each row of the output weights (the weights connected to a particular class). This shows the **logit effect** - how each neuron is affecting the logit for each class.

For each class, I then find the important neurons - the neurons which contributed more than x% to the logit for that class.
I then visualise the input weights of each important neuron, and the feature embedding of the input example. You can then also select a neuron to view it's full neuron exploration.

This provides a rich view and allows you to conduct a detailed analysis of the model's behaviour on a particular example to understand how the model is making its decision, by seeing which neurons are important for the output probability of each class, and the function of each of the important neurons.

## Future Work

In the immediate term, I plan to extend this tool to work with MLPs with >1 hidden layer. Working with multiple hidden layers is more challenging as you have to understand hidden layer <-> hidden layer connections, where you're not grounded in input space or output space on either end.


