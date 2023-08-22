import numpy as np

WEIGHTS = np.array(
    [
        # INPUT LAYER
        np.array([
            np.array([0.5, 0.2]),
            np.array([0.6, -0.1]),
            np.array([-0.4, -0.3])
        ], dtype=object),

        # HIDDEN LAYERS

        # OUTPUT LAYER
        np.array([
            np.array([0.7, -0.1, 0.2]),
        ], dtype=object)
    ], dtype=object
)

print(WEIGHTS)
# Salvando os pesos em um arquivo
np.save('weights.npy', WEIGHTS)
