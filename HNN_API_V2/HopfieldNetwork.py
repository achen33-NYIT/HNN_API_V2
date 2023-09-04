import numpy as np
import pandas as pd
import random

class Neuron:
    def __init__(self, state=0, threshold=0):
        self.state = state
        self.threshold = threshold
    
class Neural_Network:
    def __init__(self, num_neurons, states_set):
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.connection_strengths = np.zeros((num_neurons, num_neurons))
        self.states_set = np.array(states_set)
        self.update_connection_strengths()

    def update_neuron_states(self):
        while True:
            neuron_states = np.array([neuron.state for neuron in self.neurons])
            if np.any((self.states_set == neuron_states).all(axis=1)):
                break

            for i, neuron_i in enumerate(self.neurons):
                num = np.dot(self.connection_strengths[i], neuron_states) - self.connection_strengths[i,i] * neuron_states[i]

                v_new = 1 if num >= neuron_i.threshold else 0
                v_old = neuron_i.state
                neuron_i.state = v_new

    def update_neuron_states_random(self):
        k = 0
        input_seq = [neuron.state for neuron in self.neurons]
        while True:
            if k >= len(self.neurons):
                for i, neuron in enumerate(self.neurons):
                    neuron.state = input_seq[i]
                k = 0

            i = random.choice(range(len(self.neurons)))
            neuron_i = self.neurons[i]

            neuron_states = np.array([neuron.state for neuron in self.neurons])
            if np.any((self.states_set == neuron_states).all(axis=1)):
                break

            num = np.dot(self.connection_strengths[i], neuron_states) - self.connection_strengths[i,i] * neuron_states[i]
            v_new = 1 if num >= neuron_i.threshold else 0

            neuron_i.state = v_new
            k += 1

    def update_connection_strengths(self):
        for state in self.states_set:
            state_adjusted = 2 * state - 1
            self.connection_strengths += np.outer(state_adjusted, state_adjusted)
        np.fill_diagonal(self.connection_strengths, 0)  # Set diagonal to 0

    def get_energy(self):
        neuron_states = np.array([neuron.state for neuron in self.neurons])
        energy = -0.5 * np.sum(neuron_states @ self.connection_strengths * neuron_states)
        return energy



class Text_Processor:
    def __init__(self, words):

        self.words = words
        binary_words, num_neurons = self.words_to_binary(self.words)

        # Create a DataFrame to manage words and their binary forms AFTER computing binary words
        self.df = pd.DataFrame({
            'word': words,
            'binary': binary_words
        })

        states_set = [self.bin_str_to_list(bin_word) for bin_word in binary_words]
        self.neural_network = Neural_Network(num_neurons, states_set)

    def word_to_binary(self, word):
        binary_word = ''.join(format(ord(c), '08b') for c in word)
        print(binary_word)
        return binary_word

    def bin_to_word(self, binary_word):
        byte_word = int(binary_word, 2).to_bytes((len(binary_word) + 7) // 8, byteorder='big')
        recovered_word = byte_word.decode()
        return recovered_word

    def words_to_binary(self, words):
        binary_words = [self.word_to_binary(word) for word in words]
        
        # Move the dataframe creation here
        self.df = pd.DataFrame({
            'word': words,
            'binary': binary_words
        })

        # Extracting max_bits using pandas
        max_bits = self.df['binary'].str.len().max()
        return binary_words, max_bits

    def bin_str_to_list(self, bin_str):
        return [int(bit) for bit in bin_str]

    def list_to_bin_str(self, bin_list):
        return(''.join(str(bit) for bit in bin_list))

    def process_sentence(self, sentence):
        pass
    def process_word(self, word):
        in_word = self.word_to_binary(word).zfill(len(self.neural_network.neurons))
        in_word = self.bin_str_to_list(in_word)

        out_words = {}
        for i in range(10):
            for bit, neuron in zip(in_word, self.neural_network.neurons):
                neuron.state = bit

            self.neural_network.update_neuron_states_random()

            out_word = self.list_to_bin_str([neuron.state for neuron in self.neural_network.neurons])
            out_word = self.bin_to_word(out_word)

            if out_word not in out_words:
                out_words[out_word] = 1
            else:
                out_words[out_word] += 1

        # Use pandas for sorting and extracting the most frequent word
        out_words_df = pd.DataFrame(list(out_words.items()), columns=["word", "count"])
        out_word = out_words_df.sort_values(by="count", ascending=False).iloc[0]['word']

        return(word, out_word)


def main():
    #words_list = ['THE','DOG','CAT']
    words_list = ['BEANS', 'APPLE', 'VALID', 'STACK']
        
    txt_processor = Text_Processor(words=words_list)

if __name__ == '__main__':
    main()
