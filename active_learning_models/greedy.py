def greedy(model, matrix, **unused):
    return model.inference(matrix, sampling=False)
