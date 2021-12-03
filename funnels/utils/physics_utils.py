import numpy as np

def calculate_mass(four_vector):
    return (four_vector[:, 0] ** 2 - np.sum(four_vector[:, 1:4] ** 2, axis=1)) ** 0.5

def convert_from_cylindrical(four_vector):
    try:
        four_vector = four_vector.to_numpy()
    except Exception as e:
        print(e)
    constPx = four_vector[:, 0] * np.cos(four_vector[:, 2]) 
    constPy = four_vector[:, 0] * np.sin(four_vector[:, 2])
    constPz = four_vector[:, 0] * np.sinh(four_vector[:, 1])

    three_vector = np.column_stack((constPx, constPy, constPz))
    jetP2 = np.sum(three_vector ** 2, axis=1)
    constE = np.sqrt(jetP2 + four_vector[:, 3] ** 2)

    JetinExyz = np.hstack((constE.reshape(-1, 1), three_vector))
    return JetinExyz

def calculate_bmjj(bjet, j1, j2, cylindrical=True):
    if cylindrical:
        bjet = convert_from_cylindrical(bjet)
        j1 = convert_from_cylindrical(j1)
        j2 = convert_from_cylindrical(j2)

    return calculate_mass(bjet + j1 + j2)