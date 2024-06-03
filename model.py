import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import transforms
from scipy.ndimage import center_of_mass, shift


class MLP(nn.Module):
    def __init__(self, layers=1, width=64, inference=False, device="cpu", sparse_dim=None):
        super(MLP, self).__init__()

        dims = [784] + [width] * layers + [10]

        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers + 1)])

        if sparse_dim is not None:
            self.saes = []
            for dim in dims[1:-1]:
                sparse_encoder = nn.Linear(dim, sparse_dim)
                sparse_decoder = nn.Linear(sparse_dim, dim)
                self.saes.append(nn.ModuleList([sparse_encoder, sparse_decoder]))
                self.saes[-1].to(device)

        self.sparse_dim = sparse_dim

        self.inference = inference

        self.to(device)

    def forward(self, x, sparse=False, ablate=False):
        layer_activations = []
        sparse_activations = []
        sparse_reconstructions = []

        if self.inference:
            layer_activations.append(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < len(self.layers) - 1:
                x = F.relu(x)

            final_layer = i == len(self.layers) - 1

            if self.inference or (sparse and not final_layer):
                layer_activations.append(x)

            if sparse and not final_layer:
                sae = self.saes[i]
                sae_activations = F.relu(sae[0](x - sae[1].bias))
                reconstructed_activations = sae[1](sae_activations)

                sparse_activations.append(sae_activations)
                sparse_reconstructions.append(reconstructed_activations)

                if ablate:
                    x = reconstructed_activations

        if not self.inference and not sparse:
            output = F.log_softmax(x, dim=1)
            return output

        output = F.softmax(x, dim=1)

        if not sparse:
            return output, layer_activations

        if self.inference:
            # Interleave the dense and sparse activations when in sparse inference mode
            layer_activations = [layer_activations[0]] + [acts for act_pair in
                                                          zip(layer_activations[1:-1], sparse_activations) for acts in
                                                          act_pair] + [layer_activations[-1]]
            return output, layer_activations

        return output, layer_activations, sparse_activations, sparse_reconstructions

    def activation(self, layer, neuron, x, sparse=False):
        x = x.reshape(x.size()[0], 784)
        output, activations = self.forward(x, sparse=sparse)
        return activations[layer][:, neuron]


def center_image(img):
    if img is None or np.mean(img) == 0.0:
        return img
    com_x, com_y = center_of_mass(img)
    shift_x = img.shape[0] // 2 - com_x
    shift_y = img.shape[1] // 2 - com_y
    centered_img = shift(img, (shift_x, shift_y), mode='constant', cval=0)
    return centered_img


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(torch.flatten)
])

center_transform = transforms.Compose([
    transforms.Lambda(center_image),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(torch.flatten)
])


def virtual_input_weights(weights, layer_2, neuron_2, layer_1=0, min_value=-0.4242, max_value=2.8215, positive_only=False):
    layer_1_weights = weights[layer_1]

    max_tensor = torch.ones_like(layer_1_weights[0]) * max_value
    min_tensor = torch.ones_like(layer_1_weights[0]) * min_value

    positive_activations = layer_1_weights * max_tensor
    negative_activations = layer_1_weights * min_tensor

    max_activations, _ = torch.max(torch.stack([positive_activations, negative_activations]), dim=0)
    max_activations = F.relu(max_activations).t()

    num_middle_layers = layer_2 - layer_1 - 2
    for layer in range(num_middle_layers):
        layer += layer_1 + 1
        max_activations = max_activations @ weights[layer]
        max_activations = F.relu(max_activations)

    activation_map = weights[layer_2 - 1][neuron_2] @ max_activations.t()
    if positive_only:
        activation_map = F.relu(activation_map)
    return activation_map


def virtual_output_weights(weights, layer_1, neurons):
    neuron_1_weights = weights[layer_1 + 1][:, neurons]
    # neuron_1_weights = F.relu(neuron_1_weights)
    layer_2 = len(weights) - 1
    activation_map = neuron_1_weights.t() @ weights[layer_2].t()
    return activation_map


def sparse_virtual_output_weights(weights, biases, neurons):
    # neuron_1_weights = weights[1][neurons, :]
    # max_activation = 0.6
    max_activation = 1
    neuron_1_weights = weights[3][:, neurons].t()
    # neuron_1_weights *= max_activation
    # neuron_1_weights += biases[1][neurons]
    # neuron_1_weights = F.relu(neuron_1_weights)
    activation_map = neuron_1_weights @ weights[2].t()
    # activation_map += biases[2]
    return activation_map


def get_input_weights(weights, layer, neuron):
    if layer == 0:
        return weights[layer][neuron, :]
    return virtual_input_weights(weights, layer + 1, neuron)


def get_output_weights(weights, biases, layer, neurons):
    if not isinstance(neurons, list):
        neurons = [neurons]

    if layer == 0:
        output_weights = weights[-2][:, neurons]
        if len(output_weights.size()) == 1:
            output_weights = output_weights.unsqueeze(0)
        return output_weights.t()
    else:
        output_weights = sparse_virtual_output_weights(weights, biases, neurons)

    if len(output_weights.size()) == 1:
        return output_weights.unsqueeze(0).t()
    return output_weights
