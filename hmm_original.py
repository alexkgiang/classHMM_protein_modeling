import numpy as np

class HMM:
    def __init__(self, num_states, num_symbols):
        self.num_states = num_states
        self.num_symbols = num_symbols

        # Initialize with small non-zero values for Laplace smoothing
        self.transition_prob = np.ones((num_states, num_states))
        self.emission_prob = np.ones((num_states, num_symbols))
        self.initial_prob = np.ones(num_states)

    def forward_algorithm(self, sequence):
        """
        The forward algorithm: computes the sum of probabilities over all possible paths through the model
        that could produce the observed sequence. Returns P(s|theta), or the probability
        of observing a sequence s given the model paramters theta in a Hidden Markov Model.
        :return: P(s|theta)
        """
        alpha = np.zeros((self.num_states, len(sequence)))

        # Initialization
        alpha[:, 0] = self.initial_prob * self.emission_prob[:, sequence[0]]

        # Recursion
        for t in range(1, len(sequence)):
            for j in range(self.num_states):
                alpha[j, t] = np.sum(alpha[:, t - 1] * self.transition_prob[:, j]) * self.emission_prob[j, sequence[t]]

        # Termination: sum of the last column
        return np.sum(alpha[:, -1])
    
    def backward_algorithm(self, sequence):
        """
        Compute the backward probabilities for an HMM given an observation sequence.
        :param sequence: The observation sequence
        :return: A matrix of backward probabilities
        """
        T = len(sequence)
        beta = np.zeros((self.num_states, T))

        # Initialization
        beta[:, -1] = 1

        # Iteration
        for t in range(T - 2, -1, -1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    beta[i, t] += self.transition_prob[i, j] * self.emission_prob[j, sequence[t + 1]] * beta[j, t + 1]

        return beta

    
    
    def forward_algorithm_with_labels(self, sequence, labels):
        """
        The forward algorithm with labels: Calculates the probability of a specific sequence s along with
        a specific path of hidden states pi, given the model parameters theta.
        :return: P(pi, s|theta).
        """
        alpha = np.zeros((self.num_states, len(sequence)))

        # Initialization
        alpha[:, 0] = self.initial_prob * self.emission_prob[:, sequence[0]] * (labels[0] == np.arange(self.num_states))

        # Recursion
        for t in range(1, len(sequence)):
            for j in range(self.num_states):
                alpha[j, t] = np.sum(alpha[:, t - 1] * self.transition_prob[:, j]) * self.emission_prob[j, sequence[t]] * (labels[t] == j)

        # Termination
        return np.sum(alpha[:, -1])
    
    def joint_probability(self, labels, sequence):
        """
        Calculate the joint probability P(c, s|θ, Φ)
        :param labels: The sequence of labels (hidden states) c
        :param sequence: The sequence of observations s
        :return: The joint probability P(c, s|θ, Φ)
        """

        # Initialization
        prob = self.initial_prob[labels[0]] * self.emission_prob[labels[0], sequence[0]]

        # Compute the probability for the specific path and sequence
        for t in range(1, len(sequence)):
            state = labels[t]
            prev_state = labels[t - 1]
            observation = sequence[t]
            prob *= self.transition_prob[prev_state, state] * self.emission_prob[state, observation]

        return prob


    def compute_gradients(self, sequences, labels):
        # Initialize gradients
        transition_gradients = np.zeros_like(self.transition_prob)
        emission_gradients = np.zeros_like(self.emission_prob)

        # Iterate over each sequence and its corresponding label sequence
        for sequence, label_sequence in zip(sequences, labels):
            # Compute probabilities and expected counts for this sequence
            alpha = self.forward_algorithm(sequence)
            beta = self.backward_algorithm(sequence)  # Assuming backward algorithm is implemented

            for t in range(1, len(sequence)):
                actual_transition = (label_sequence[t - 1], label_sequence[t])
                actual_emission = (label_sequence[t], sequence[t])

                # Update transition gradients
                for i in range(self.num_states):
                    for j in range(self.num_states):
                        expected_count = alpha[t - 1, i] * self.transition_prob[i, j] * beta[t, j]
                        transition_gradients[i, j] -= (i == actual_transition[0] and j == actual_transition[1]) - expected_count / self.transition_prob[i, j]

                # Update emission gradients
                for state in range(self.num_states):
                    for symbol in range(self.num_symbols):
                        expected_count = alpha[t, state] * beta[t, state]
                        emission_gradients[state, symbol] -= (state == actual_emission[0] and symbol == actual_emission[1]) - expected_count / self.emission_prob[state, symbol]

        return transition_gradients, emission_gradients



    def update_parameters(self, transition_gradients, emission_gradients, learning_rate):
        # Update transition probabilities
        self.transition_prob -= learning_rate * transition_gradients
        self.transition_prob = np.clip(self.transition_prob, 1e-6, np.inf)
        self.transition_prob /= np.sum(self.transition_prob, axis=1, keepdims=True)

        # Update emission probabilities
        self.emission_prob -= learning_rate * emission_gradients
        self.emission_prob = np.clip(self.emission_prob, 1e-6, np.inf)
        self.emission_prob /= np.sum(self.emission_prob, axis=1, keepdims=True)


    def CHMM_train(self, sequences, labels, learning_rate, epochs):
        for epoch in range(epochs):
            # Calculate Lf using the Forward algorithm for each sequence
            Lf = -sum(np.log(self.forward_algorithm(seq)) for seq in sequences)

            # Calculate Lc using the joint_probability function for each sequence and its labels
            Lc = -sum(np.log(self.joint_probability(lab, seq)) for seq, lab in zip(labels, sequences))

            # Compute the objective
            objective = Lc - Lf

            # Compute gradients
            transition_gradients, emission_gradients = self.compute_gradients(sequences, labels)

            # Update parameters
            self.update_parameters(transition_gradients, emission_gradients, learning_rate)

            print(f"Epoch {epoch+1}, Objective: {objective}")

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

    # This the viterbi algorithm
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
    [0, 1, 0],  # Labels for sequence 1
    [1, 2, 0],  # Labels for sequence 2
]

# Create and train the HMM
num_states = 3  # Number of states (alpha, beta, coil)
num_symbols = 20  # Number of symbols (amino acids)
hmm = HMM(num_states, num_symbols)
hmm.train(sequences, labels)
# hmm.CHMM_train(sequences, labels, 0.001, 10)

# Test with a new sequence
test_sequence = [0, 1, 2, 3, 4]  # An unlabeled sequence
predicted_label, best_path_prob = hmm.predict(test_sequence)
print("Predicted Label:", predicted_label)
print("Best Path Probability:", best_path_prob) #how confident the model is
