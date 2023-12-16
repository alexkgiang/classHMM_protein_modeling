# Reproducibility

Data pipeline:

1. Go to the website https://www.uniprot.org/
2. Search up the word "human". This will display all of the protein data.
3. On the left side of the web page, filter by two things: human and 3D structure. For mice, do the same but filter by mice instead of humans. This will produce 8,374 proteins for humans and 478 proteins for mice.
4. Hit the download button at the top. Choose Excel for format, and do not compress the data.
5. We chose the following for the columns, based on what we needed and in the following order: Length, Organism, Helix, Beta strand, Turn, Sequence.
6. Download this file. Once downloaded, convert the file to a csv.
7. Once the files are in csv format, we run a data script (data_script.py), which transforms the data into the format we need. This format is specified in more detail in the paper, but each protein corresponds to two lines in our data. The first one is the sequence, and the second one is a 1:1 mapping from sequence to label.
8. Once the script is run, place the files into the folder with the jupyter notebook. We cut the data down to about 500 sequences per organism. You can find this data included in the repository, titled HUMAN_training_data.txt and RAT_training_data.txt. Note that we only have one file for each organism, and the train/validation splits are handled within the file.
9. In the jupyter notebook, change the file names to correspond with the correct rat and human data files. Then, hit "Run All". Please note that running this file will take about ~40 minutes.
