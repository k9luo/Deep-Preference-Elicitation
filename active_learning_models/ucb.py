import numpy as np


def ucb(model, matrix, ci=1, num_latent_sampling=5):
    prediction = []

    for i in range(num_latent_sampling):
        prediction.append(model.inference(matrix, sampling=True))

    mean = np.mean(prediction, axis=0)
    std = np.std(prediction, axis=0)
    return mean + ci * std
