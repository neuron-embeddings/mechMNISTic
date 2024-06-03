import matplotlib.pyplot as plt
import math
import torch
import torch.nn.functional as F


def shift_scale(tensor):
    tensor = tensor.cpu().detach()
    tensor -= torch.min(tensor)
    tensor /= torch.max(tensor)
    return tensor


def activation_statistic(statistics, layer, neuron):
    summary = statistics[(layer, neuron)]
    summary = {k: v for k, v in sorted(summary.items(), key=lambda x: float(x[0]))}

    keys = [float(key) for key in summary.keys()]
    values = list(summary.values())

    cumulative_counts = [sum(values[:i+1]) for i in range(len(values))]
    total = cumulative_counts[-1]

    weighted_values = [count / total for count in cumulative_counts]

    fig = plt.figure(figsize=(4, 3.5))
    plt.plot(keys, weighted_values)
    plt.ylim([0, 1.1])
    plt.xlim([min(keys), max(keys)])

    plt.xlabel("Activation Value")
    plt.ylabel("Cumulative Probability")
    return fig


def max_absolute(tensor):
    return torch.max(torch.abs(tensor)).item()


def output_weights(weight_tensor, xt=None, yt=None, figsize=None, **kwargs):
    if figsize is None:
        figsize = (len(weight_tensor) * 2, 2.1)

    if yt is None:
        yt = [i for i in range(len(weight_tensor))]

    max_val = max_absolute(weight_tensor)
    return raw(
        weight_tensor, square=False, axis=True, figsize=figsize, cmap="coolwarm", vmin=-max_val, vmax=max_val,
        yt=yt, xt=xt, **kwargs
    )


def raw(
        tensor, square=True, axis=False, figsize=None, cmap="gray", vmin=None, vmax=None, colorbar=True, title=None,
        xt=None, yt=None, colorbar_kwargs=None
):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach()

        if len(tensor.size()) == 1:
            tensor = tensor.unsqueeze(0)

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    if square:
        dim = int(math.sqrt(tensor.size()[-1]))
        tensor = tensor.reshape(dim, dim)

    plt.imshow(tensor, cmap=cmap, vmin=vmin, vmax=vmax)
    show_axis = "on" if axis else "off"
    plt.axis(show_axis)

    if xt is not None:
        if isinstance(xt, tuple):
            plt.xticks(xt[0], xt[1])
        else:
            plt.xticks(xt)
    elif tensor.size()[1] == 1:
        plt.xticks([])

    if yt is not None:
        if isinstance(yt, tuple):
            plt.yticks(yt[0], yt[1])
        else:
            plt.yticks(yt)
    elif tensor.size()[0] == 1:
        plt.yticks([])

    if colorbar:
        if colorbar_kwargs is not None:
            plt.colorbar(**colorbar_kwargs)
        else:
            plt.colorbar()

    if title is not None:
        plt.title(title)

    return fig


def digit(index, data):
    element = data[index][0]
    return raw(element)


def grid(imgs, cols=5, title=None, cmaps=None):
    if cmaps is None:
        cmaps = ["gray"] * len(imgs)

    rows = math.ceil(len(imgs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Turn off axis for all subplots
    for ax in axes.flatten():
        ax.axis('off')

    for i, img in enumerate(imgs):
        row = i // cols
        col = cols - (i % cols) - 1

        if rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]

        ax.imshow(img.reshape(28, 28), cmap=cmaps[i])

    plt.tight_layout()
    if title is not None:
        plt.title(title)

    return fig


def kernel(kernel, cmap="coolwarm", centre=True, **kwargs):
    max_element = max_absolute(kernel)
    if centre:
        return raw(kernel, cmap=cmap, vmin=-max_element, vmax=max_element, **kwargs)
    else:
        return raw(kernel, cmap=cmap, **kwargs)


def cluster(examples, clusters, layer, neuron, max_clusters=10, max_per_cluster=4, neuron_clusters=None):
    if neuron_clusters is None:
        neuron_clusters = clusters[(layer, neuron)]

    metrics = neuron_clusters[-1]
    neuron_clusters = neuron_clusters[:-1][0]
    neuron_clusters = sorted(neuron_clusters, key=lambda x: len(x[0]), reverse=True)

    figs = []
    central_idxs = []
    avg_images = []

    for i, (cluster, central) in enumerate(neuron_clusters):
        imgs = []
        all_imgs = []
        for idx, (example_idx, embedding_idx) in cluster:
            all_imgs.append(examples[example_idx][0])
            if len(imgs) < max_per_cluster:
                imgs.append(examples[example_idx][0])

        fig = grid(imgs, cols=3 if len(imgs) <= 3 else max_per_cluster // 2)
        # fig = grid(imgs, cols=max_per_cluster // 2)
        figs.append(fig)
        central_idxs.append(central[1][0])
        avg_images.append(average_examples(all_imgs))

        if max_clusters is not None and i + 1 >= max_clusters:
            break

    return figs, avg_images, metrics


def feature_embedding(example, kernel, centre=False, **kwargs):
    kernel = kernel.detach()

    background = kernel[0].clone()
    kernel -= background
    kernel = F.relu(kernel)

    example = shift_scale(example)
    kernel = shift_scale(kernel)

    feature_embedding = example * kernel
    # max_abs = max_absolute(feature_embedding)
    # return raw(feature_embedding, colorbar=False, vmin=-max_abs, vmax=max_abs, **kwargs)
    return raw(feature_embedding, colorbar=False, **kwargs)


def average_examples(activating_examples):
    activating_examples = torch.stack(activating_examples)
    activating_examples -= torch.min(activating_examples)
    avg_example = torch.mean(activating_examples, dim=0)
    return avg_example

