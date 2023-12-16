def convert_to_sequence_and_structure(txt_file_path, output_file_path):
    """
    Converts a CSV file with specific columns to a sequence and structure format.
    Args:
    txt_file_path (str): Path to the input CSV file.
    output_file_path (str): Path to save the output file.
    The function reads an input file, extracts sequence and structure information,
    and saves it to an output file where each sequence is followed by its structure mapping.
    """

    def parse_structure(structure_str, length, structure_type):
        """
        Parses the structure string to generate a sequence structure mapping.
        Args:
        structure_str (str): The structure string from the file.
        length (int): Length of the protein sequence.
        structure_type (str): Type of structure ('H' for Helix, 'B' for Beta strand, 'T' for Turn).

        Returns:
        list: A list of characters representing the structure at each position in the sequence.
        """
        structure_map = ['0'] * (length)

        if structure_str:
            parts = structure_str.split('; ')
            for part in parts:
                if ' ' in part:
                    range_str = part.split(' ')[1]
                    range_parts = range_str.split('..')
                    if len(range_parts) == 2:
                        try:
                            start, end = map(int, range_parts)
                            for i in range(start - 1, end):
                                structure_map[i] = structure_type
                        except ValueError:
                            continue

        return structure_map


    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    with open(output_file_path, 'w') as output_file:
        for line in lines[1:]:
            parts = line.strip().split(',')

            length = int(parts[1])
            sequence = parts[6]
            helix_str = parts[3].replace('"', '')
            strand_str = parts[4].replace('"', '')
            turn_str = parts[5].replace('"', '')

            # Generate structure mappings
            helix_map = parse_structure(helix_str, length, 'H')
            strand_map = parse_structure(strand_str, length, 'B')
            turn_map = parse_structure(turn_str, length, 'T')

            # Combine structure mappings
            combined_structure_map = []
            for h, s, t in zip(helix_map, strand_map, turn_map):
                if h == 'H':
                    combined_structure_map.append('1')  # Helix
                elif s == 'B':
                    combined_structure_map.append('2')  # Beta strand
                elif t == 'T':
                    combined_structure_map.append('3')  # Turn
                else:
                    combined_structure_map.append('0')  # No structure

            output_file.write(sequence + '\n')
            output_file.write(''.join(combined_structure_map) + '\n')

    print(f"Converted data saved to {output_file_path}")

convert_to_sequence_and_structure("input_data.csv", "data_ready_for_our_model.txt")
