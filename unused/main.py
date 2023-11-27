import numpy as np
from hmmlearn import hmm

class CHMM:
    def __init__(self, n_states, n_trials):
        self.n_states = n_states
        self.n_trials = n_trials  # This should be the length of the sequence for DNA/RNA
        self.model = hmm.CategoricalHMM(n_components=n_states, n_iter=100, init_params="ste")

    def initialize_parameters(self, start_probability, transition_matrix, emission_probability):
        self.model.startprob_ = start_probability
        self.model.transmat_ = transition_matrix
        self.model.emissionprob_ = emission_probability

    def train(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)

# Example usage:
if __name__ == "__main__":
    n_states = 4 # 4 states of the model
    n_trials = 10  # Assuming we have sequences of length 10

    # Start probabilities for each state
    # start_prob = np.array([0.25, 0.25, 0.25, 0.25])
    
    # # Transition probabilities between states
    # trans_matrix = np.array([
    #     [0.25, 0.25, 0.25, 0.25],
    #     [0.25, 0.25, 0.25, 0.25],
    #     [0.25, 0.25, 0.25, 0.25],
    #     [0.25, 0.25, 0.25, 0.25]
    # ])
    
    # # Emission probabilities of observing each nucleotide from each state
    # emission_prob = np.array([
    #     [1.0, 0.0, 0.0, 0.0],  # High probability of 'A' in state 1
    #     [0.0, 1.0, 0.0, 0.0],  # High probability of 'T' in state 2
    #     [0.0, 0.0, 1.0, 0.0],  # High probability of 'G' in state 3
    #     [0.0, 0.0, 0.0, 1.0],  # High probability of 'C' in state 4
    # ])

    model = CHMM(n_states, n_trials)
    # model.initialize_parameters(start_prob, trans_matrix, emission_prob)

    # Example sequence - this should be a count matrix now, not just the symbols
    # The counts should add up to n_trials for each observation sequence
    # Here we need to encode our DNA sequence as counts of occurrences of each nucleotide
    sequence_data = np.array([[3, 2, 4, 1]])  # Example: 3 'A's, 2 'T's, 4 'G's, and 1 'C' in one window of the sequence

    model.train(sequence_data)

    # Predicting the state sequence for a new sequence data
    new_sequence_data = np.array([[3, 3, 2, 2]])  # New sequence of counts
    predictions = model.predict(new_sequence_data)
    print("Predicted states:", predictions)
