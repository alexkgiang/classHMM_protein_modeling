import numpy as np

class HMM:
    def __init__(self, num_states, num_symbols):
        self.num_states = num_states
        self.num_symbols = num_symbols

        # Initialize with small non-zero values for Laplace smoothing
        self.transition_prob = np.ones((num_states, num_states))
        self.emission_prob = np.ones((num_states, num_symbols))
        self.initial_prob = np.ones(num_states)

    def train(self, sequences, labels):
        self.initial_prob = np.zeros(self.num_states)

        for label in labels:
            self.initial_prob[label[0]] += 1
        self.initial_prob /= len(labels)

        for i, label in enumerate(labels):
            for j in range(1, len(label)):
                self.transition_prob[label[j - 1], label[j]] += 1
                self.emission_prob[label[j], sequences[i][j]] += 1

        # Normalize the probabilities
        self.transition_prob /= self.transition_prob.sum(axis=1, keepdims=True)
        self.emission_prob /= self.emission_prob.sum(axis=1, keepdims=True)

    def predict(self, sequence):
        T = len(sequence)
        delta = np.zeros((self.num_states, T))
        psi = np.zeros((self.num_states, T), dtype=int)

        # Initialization
        delta[:, 0] = self.initial_prob * self.emission_prob[:, sequence[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.num_states):
                trans_prob = delta[:, t - 1] * self.transition_prob[:, j]
                delta[j, t] = np.max(trans_prob)
                psi[j, t] = np.argmax(trans_prob)
            delta[:, t] *= self.emission_prob[:, sequence[t]]

        # Termination
        best_path_prob = np.max(delta[:, -1])
        best_path_state = np.argmax(delta[:, -1])

        # Backtrack to find the best path
        best_path = [best_path_state]
        for t in range(T - 1, 0, -1):
            best_path_state = psi[best_path_state, t]
            best_path.insert(0, best_path_state)

        return best_path, best_path_prob

# Example usage
sequences = [
    [0, 1, 2, 3, 4],  # Protein sequence 1
    [3, 2, 1, 0, 3],  # Protein sequence 2
]

# reason for many labels: one part of the protein might form an alpha helix, but another may form a beta-sheet.
labels = [
    [0, 0, 0, 0, 0],  # Labels for sequence 1
    [1, 2, 0, 1, 2],  # Labels for sequence 2
]

# Create and train the HMM
num_states = 3  # Number of states (alpha, beta, coil)
num_symbols = 20  # Number of symbols (amino acids)
hmm = HMM(num_states, num_symbols)
hmm.train(sequences, labels)

# Test with a new sequence
test_sequence = [0, 1, 2, 3, 4]  # An unlabeled sequence
predicted_label, best_path_prob = hmm.predict(test_sequence)
print("Predicted Label:", predicted_label)
print("Best Path Probability:", best_path_prob) #how confident the model is
