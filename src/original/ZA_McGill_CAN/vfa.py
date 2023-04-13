import numpy as np
from numpy.linalg import norm


def despot(signal, alpha, TR):
    # Ref: Deoni, S. C. L., Peters, T. M., & Rutt, B. K. (2005). MRM 53(1), 237–241. https://doi.org/10.1002/mrm.20314
    # Based on Matlab code by Ives Levesque, McGill University
    x = signal / np.tan(alpha)
    y = signal / np.sin(alpha)
    numerator = np.sum(x * y, axis=-1) - np.sum(x, axis=-1) * np.sum(y, axis=-1) / len(
        alpha
    )
    denominator = np.sum(x * x, axis=-1) - np.sum(x, axis=-1) ** 2 / len(alpha)
    slope = numerator / denominator
    intercept = np.mean(y, axis=-1) - slope * np.mean(x, axis=-1)
    M0 = intercept / (1 - slope)
    T1 = -TR / np.log(slope)
    return M0, T1


def novifast(
    signal,
    alpha,
    TR,
    initialvalues=[5000, 1500],
    maxiter=10,
    tol=1e-6,
    doiterative=True,
):
    # Ref: Ramos-Llordén, G., Vegas-Sánchez-Ferrero, G., Björk, M., Vanhevel, F., Parizel, P. M.,
    #      San José Estépar, R., den Dekker, A. J., and Sijbers, J.
    #      NOVIFAST: a fast algorithm for accurate and precise VFA MRI T1 mapping.
    #      IEEE Trans. Med. Imag., early access, doi:10.1109/TMI.2018.2833288
    spatialdims = signal.shape[:-1]
    if not spatialdims:
        spatialdims = [1]
    numvox = np.prod(spatialdims)
    numangles = signal.shape[-1]

    y = signal.reshape(-1, numangles)
    sinfa = np.asarray(np.sin(alpha)).reshape((1, -1))
    sinfa = np.broadcast_to(sinfa, (numvox, numangles))
    cosfa = np.asarray(np.cos(alpha)).reshape((1, -1))
    cosfa = np.broadcast_to(cosfa, (numvox, numangles))

    initialM0, initialT1 = initialvalues
    # solA and solB and c1 and c2 in paper
    solA = np.repeat(initialM0 * (1 * np.exp(-TR / initialT1)), numvox)
    solB = np.repeat(np.exp(-TR / initialT1), numvox)
    k = 0
    done = False
    while not done:
        solB_prev = np.copy(solB)
        solA = np.broadcast_to(np.asarray(solA).reshape((-1, 1)), (numvox, numangles))
        solB = np.broadcast_to(np.asarray(solB).reshape((-1, 1)), (numvox, numangles))
        # Based on equations 24 to 27 in paper
        denominator = 1 - cosfa * solB
        Z = y / denominator
        A = cosfa * Z
        B = sinfa / denominator
        Abar = cosfa * B * solA / denominator
        # Calculate terms in Eq. 28 of paper
        BB = np.sum(B * B, axis=1)
        BA = np.sum(B * A, axis=1)
        BZ = np.sum(B * Z, axis=1)
        AAbar = np.sum(A * Abar, axis=1)
        BAbar = np.sum(B * Abar, axis=1)
        ZAbar = np.sum(Z * Abar, axis=1)

        determinant = BB * AAbar - BAbar * BA
        solA = (BZ * AAbar - ZAbar * BA) / determinant
        solB = (BB * ZAbar - BAbar * BZ) / determinant
        k += 1
        if not doiterative:
            done = True
        else:
            err = norm(solB - solB_prev) / norm(solB)
            if err < tol or k >= maxiter:
                done = True
    M0 = solA / (1 - solB)
    T1 = -TR / np.log(solB)
    M0 = M0.reshape(spatialdims)
    T1 = T1.reshape(spatialdims)
    return M0, T1
