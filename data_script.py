def process_protein_data(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    sequence = lines[0].strip()

    label_sequence = ['0'] * len(sequence)

    for line in lines[1:]:
        parts = line.split()
        structure = parts[0]
        if parts[1] == 'strand':
            indices = parts[2].split('-')
        else:
            indices = parts[1].split('-')
        start = int(indices[0]) - 1
        end = int(indices[1]) - 1

        label = '0'
        if structure == 'Beta':
            label = '2'
        elif structure == 'Helix':
            label = '1'
        elif structure == 'Turn':
            label = '3'

        for i in range(start, end + 1):
            label_sequence[i] = label

    label_sequence_str = ''.join(label_sequence)

    with open(output_file, 'a') as file: 
        file.write(sequence + '\n')
        file.write(label_sequence_str + '\n')

input_file = 'one_sequence.txt'
output_file = 'RAT_training_data.txt'
process_protein_data(input_file, output_file)
