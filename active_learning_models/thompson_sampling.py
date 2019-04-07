def thompson_sampling(model, matrix, **unused):
    return model.inference(matrix, sampling=True)
