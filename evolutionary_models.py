from collections import Counter
import numpy as np


# Function to calculate transition and transversion probabilities
def calculate_transitions_transversions(seq1, seq2, multiple=True):
    transitions = 0
    transversions = 0
    purines = {'A', 'G'}
    pyrimidines = {'C', 'T'}

    # Iterate through aligned sequences
    for base1, base2 in zip(seq1, seq2):
        if base1 != base2:  # Only consider substitutions
            if (base1 in purines and base2 in purines) or (base1 in pyrimidines and base2 in pyrimidines):
                transitions += 1
            elif base1 in purines and base2 in pyrimidines or base1 in pyrimidines and base2 in purines:
                transversions += 1

    # Returning number of transitions and transversions (multiple)
    if multiple:
        return transitions, transversions

    # Calculate probabilities (pairwise)
    sequence_length = len(seq1)
    p = transitions / sequence_length
    q = transversions / sequence_length
    return p, q


def hky85_parameters(seq1, seq2):
    # Count transitions and transversions
    transitions = 0
    transversions = 0
    total = 0
    purines = {"A", "G"}
    pyrimidines = {"C", "T"}

    for base1, base2 in zip(seq1, seq2):
        if base1 != base2 and base1 != "-" and base2 != "-":
            if (base1 in purines and base2 in purines) or (base1 in pyrimidines and base2 in pyrimidines):
                transitions += 1
            else:
                transversions += 1
        if base1 != "-" and base2 != "-":
            total += 1

    P = transitions / total
    Q = transversions / total

    # Calculate nucleotide frequencies
    all_bases = seq1 + seq2
    counts = Counter(all_bases)
    total_bases = sum(counts.values())
    pi_A = counts["A"] / total_bases
    pi_C = counts["C"] / total_bases
    pi_G = counts["G"] / total_bases
    pi_T = counts["T"] / total_bases

    return P, Q, pi_A, pi_C, pi_G, pi_T


def hky85_distance_score(p, q, pi_a, pi_c, pi_g, pi_t):
    distance_score = (-0.5 * np.log(1 - (p / (1 - ((pi_a + pi_g) * (pi_c + pi_t)))))
                      - 0.25 * np.log(1 - (2 * q)))
    distance_score = np.nan_to_num(distance_score, nan=1000, neginf=1000, posinf=1000)
    return distance_score


def get_consensus(seq1, seq2):
    consensus = []
    for base1, base2 in zip(seq1, seq2):
        if base1 == base2:  # If bases are identical
            consensus.append(base1)
        elif base1 == '-' or base2 == '-':  # Handle gaps
            consensus.append('-')
        else:  # Mismatch
            consensus.append('N')  # Use 'N' for unresolved positions
    return ''.join(consensus)


def main():
    print("This is module for evolutionary models.")


if __name__ == "__main__":
    main()
