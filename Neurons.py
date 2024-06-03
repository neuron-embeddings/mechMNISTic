import streamlit as st
import json
import matplotlib as mpl
import functools
import torch
from torchvision import datasets

import display
import utils

from model import get_input_weights, get_output_weights, MLP, transform

# ============== CONFIG ==============

st.set_page_config(page_title="MechMNISTic", layout="wide")

models = ["Standard SAE", "Neuron Embedding SAE"]
default_index = 1

model_name = models[default_index]
model_path = f"models/{model_name}"


def load_model(name, config_only=False):
    path = f"models/{name}"

    with open(f"{path}/config.json") as ifh:
        config = json.load(ifh)

    if config_only:
        return config

    model = MLP(layers=config["layers"], width=config["neurons"], inference=True)
    model.load_state_dict(torch.load(f"{path}/model.pt", map_location=torch.device('cpu')))
    model.eval()

    return model, config


config = load_model(model_name, config_only=True)
layers = config["layers"]
neurons = config["neurons"]


def model_weights(model):
    weights = []
    biases = []

    for i, ((name, parameters), (sparse_name, sparse_parameters)) in enumerate(
            zip(model.named_parameters(), model.saes[0].named_parameters())
    ):
        if "weight" not in name:
            biases.append(parameters)
            biases.append(sparse_parameters)
        else:
            weights.append(parameters)
            weights.append(sparse_parameters)

    output_weights = weights[-1]

    return weights, output_weights, biases


def load_sparse(model_name):
    state_dict = torch.load(f"models/{model_name}/model.pt", map_location="cpu")

    model = MLP(layers=1, width=neurons, inference=True, sparse_dim=8 * 64)
    model.load_state_dict(state_dict)

    model.eval()

    sparse_state_dict = torch.load(f"models/{model_name}/sae.pt", map_location="cpu")
    model.saes[0].load_state_dict(sparse_state_dict)

    return model


@st.cache_resource
def load(model_name):
    model_path = f"models/{model_name}"

    with open(f"{model_path}/activation_statistics.json") as ifh:
        statistics = utils.to_dict(json.load(ifh))

    with open(f"{model_path}/clusters.json") as ifh:
        clusters = utils.to_dict(json.load(ifh))

    with open(f"{model_path}/indexing.json") as ifh:
        indexing = json.load(ifh)

    examples = datasets.MNIST(f'data', train=True, download=True, transform=transform)
    test_examples = datasets.MNIST(f'data', train=False, download=True, transform=transform)

    model = load_sparse(model_name)
    weights, output_weights, biases = model_weights(model)

    return statistics, clusters, indexing, examples, test_examples, model, weights, output_weights, biases


statistics, clusters, indexing, examples, test_examples, model, weights, output_weights, biases = load(model_name)

max_clusters = 10

mpl.rcParams['figure.max_open_warning'] = 0

plot = functools.partial(st.pyplot, clear_figure=True)


# ============== FUNCTIONALITY ==============


def show_neuron(layer, neuron, max_clusters=max_clusters, reload=False, plot_summary=True, plot_clusters=True,
                plot_statistic=True):
    global statistics, clusters, indexing, examples, test_examples, model, weights, output_weights

    def summary(plot_statistic=True):
        if layer == 0:
            col_1_header = "Input Weights"
            col_1_caption = "Input Weights visualised as a 28x28 image\n\nEach weight connects to a pixel in the input"
        else:
            col_1_header = "Activation Map"
            col_1_caption = "Each pixel shows the maximum neuron activation it can induce"

        if layer == layers - 1:
            col_3_caption = "Each weight connects to an output class"
        else:
            col_3_caption = "Each element shows how the neuron affects the output class"

        kernel = get_input_weights(weights, layer, neuron)

        neuron_id = f"{layer}_{neuron}"
        if neuron_id not in indexing["neuron_id_to_example_idxs"]:
            st.subheader(f"No activating examples")
            return

        example_idxs = indexing["neuron_id_to_example_idxs"][neuron_id]

        activating_examples = []
        for example_idx in example_idxs:
            example = examples[example_idx][0]
            activating_examples.append(example)

        average_example = display.average_examples(activating_examples)

        output_weight_tensor = get_output_weights(weights, biases, layer, neuron).t()

        rows = [
            (
                [
                    col_1_header, "Average Input", "Logit Effects"
                ],
                st.subheader
            ),
            (
                [
                    col_1_caption,
                    "Average of the inputs that strongly activate the neuron",
                    col_3_caption
                ],
                st.caption
            ),
            (
                [
                    display.kernel(kernel, colorbar=True, centre=True),
                    display.raw(average_example, colorbar=False),
                    display.output_weights(output_weight_tensor, figsize=(2, 2), colorbar_kwargs={"pad": 0.2}),
                ],
                plot
            )
        ]

        if plot_statistic:
            stats_rows = [
                "Activations",
                "Cumulative distribution of neuron activations",
                display.activation_statistic(statistics, layer, neuron)
            ]

            for (content, func), stat_content in zip(rows, stats_rows):
                content.append(stat_content)

        for row in rows:
            width_1 = 0.3
            width_2 = 0.25
            width_3 = 0.2
            cols = st.columns([width_1, width_2, width_3, 1 - width_1 - width_2 - width_3])

            contents, func = row

            for col, content in zip(cols, contents):
                with col:
                    func(content)

    def show_clusters():
        kernel = get_input_weights(weights, layer, neuron)

        max_per_cluster = 12

        cluster_figs, central_idxs, metrics = display.cluster(
            examples, clusters, layer, neuron, max_per_cluster=max_per_cluster, max_clusters=max_clusters
        )

        if len(metrics) > 0:
            st.subheader("Embedding Metrics")
            width = 0.55
            col_width = width / len(metrics)
            metric_cols = st.columns([col_width for _ in metrics] + [1 - width])
            for col, (name, value) in zip(metric_cols, metrics.items()):
                with col:
                    value = round(value, 2) if isinstance(value, float) else value
                    st.metric(name.replace("_", " ").title(), value)

        cluster_width = 0.5
        other_width = (1 - cluster_width) / 3
        for i, (fig, avg_im) in enumerate(zip(cluster_figs, central_idxs)):
            col_1, col_2, col_3, col_4 = st.columns([cluster_width, other_width, other_width, other_width])
            with col_1:
                if i == 0:
                    st.subheader(f"Top {max_clusters} Feature Clusters")
                    st.caption("Ordered by number of elements, clustered by feature similarity")
                plot(fig)
            with col_2:
                if i == 0:
                    st.subheader(f"Weights")
                    st.caption("Learned Input Weights")
                plot(display.kernel(kernel, colorbar=False, cmap="gray", centre=False))
            with col_3:
                if i == 0:
                    st.subheader(f"Cluster Average")
                    st.caption("\u200B")
                plot(display.raw(avg_im, colorbar=False))
            with col_4:
                if i == 0:
                    st.subheader(f"Feature")
                    st.caption("Cluster Average ⊙ Weights")
                plot(display.feature_embedding(avg_im, kernel))

    if reload:
        statistics, clusters, indexing, examples, test_examples, model, weights, output_weights = load(model_name)

    if plot_summary:
        summary(plot_statistic=plot_statistic)

    if plot_clusters:
        show_clusters()


# ============== PAGE ==============

if __name__ == "__main__":
    st.title("Neuron Explorer")
    st.caption("Explore MLP and SAE neurons for a model trained on MNIST")

    col_width = 0.3
    with st.columns([col_width, 1 - col_width])[0]:
        model_name = st.selectbox("Select model", models, index=default_index)

        config = load_model(model_name, config_only=True)

        layers = config["layers"]
        neurons = config["neurons"]

        statistics, clusters, indexing, examples, test_examples, model, weights, output_weights, biases = load(model_name)

    col_1, col_2 = st.columns([0.5, 0.5])
    with col_1:
        layer_name = st.selectbox(
            f"Select Layer", ["Hidden Layer", "SAE"], index=0, help="Choose a layer to view"
        )
        layer = 1 if layer_name == "Hidden Layer" else 2
    neurons = config["neurons"] if layer == 1 else config["SAE_neurons"]
    with col_2:
        neuron = st.number_input(f"Select Neuron (0 to {neurons - 1})", min_value=0, max_value=neurons - 1, value=0,
                                 help="Choose a neuron to view")

    show_neuron(layer - 1, neuron)
