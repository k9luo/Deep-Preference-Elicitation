from active_learning_models.greedy import greedy
from active_learning_models.thompson_sampling import thompson_sampling
from active_learning_models.ucb import ucb

from recommendation_models.vae import VAE


rec_models = {
    "VAE-CF": VAE,
}

active_models = {
    "Greedy": greedy,
    "ThompsonSampling": thompson_sampling,
    "UCB": ucb,
}
