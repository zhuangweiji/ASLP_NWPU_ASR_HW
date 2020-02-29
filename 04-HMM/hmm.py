# Author: Kaituo Xu, Fan Yu
import numpy as np


def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment
    # Initialize
    alpha = np.zeros([T, N])
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][O[0]]
    # Induction
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                alpha[t][i] += alpha[t - 1][j] * A[j][i]
            alpha[t][i] *= B[i][O[t]]
    # end
    for i in range(N):
        prob += alpha[T - 1][i]
    # End Assignment
    return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment
    beta = np.zeros([T, N])
    for i in range(N):
        beta[T - 1][i] = 1  # t = 2
    # Induction
    for t in range(T - 2, -1, -1):  # t = 1, 0
        for i in range(N):
            for j in range(N):
                beta[t][i] += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]
    # end
    for i in range(N):
        prob += pi[i] * B[i][O[0]] * beta[0][i]
    # End Assignment
    return prob


def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, [0] * T
    # Begin Assignment
    # Initialize
    delta = -np.inf * np.ones([T, N])
    for i in range(N):
        delta[0][i] = pi[i] * B[i][O[0]]
    phi = [[0] * T] * N
    # Induction
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                max_tmp = delta[t - 1][j] * A[j][i]
                if max_tmp > delta[t][i]:
                    delta[t][i] = max_tmp
                    phi[t][i] = j + 1
            delta[t][i] *= B[i][O[t]]
    # Termination
    for i in range(N):
        if delta[T - 1][i] > best_prob:
            best_prob = delta[T - 1][i]
            best_path[T - 1] = i + 1  # t = 2
    # Backtrack
    for t in range(T - 2, -1, -1):  # t = 1, 0
        best_path[t] = phi[t + 1][best_path[t + 1] - 1]
    # End Assignment
    return best_prob, best_path


if __name__ == "__main__":
    color2id = {"RED": 0, "WHITE": 1}
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model)
    print(best_prob, best_path)
