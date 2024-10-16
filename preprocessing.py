from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class PeptideDataset(Dataset):
    def __init__(self, sequences, intensities):
        self.sequences = sequences
        self.intensities = intensities

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.intensities[idx]


def restructure_data(deepmass_path: Path, hela1_path: Path, hela2_path: Path):
    deepmass_prism_df = pd.read_csv(deepmass_path, delimiter="\t")
    hela_cell_1_df = pd.read_csv(hela1_path, delimiter="\t")
    hela_cell_2_df = pd.read_csv(hela2_path)

    parsed_fragment = {"deepmass": list(), "hela1": list(), "hela2": list()}
    for freg_intensity in deepmass_prism_df["FragmentIntensities"]:
        parsed_intensities = freg_intensity.split(";")
        parsed_fragment["deepmass"].append(";".join([item for item in parsed_intensities]))

    for freg_intensity in hela_cell_1_df["Exp Intensity"]:
        parsed_intensities = freg_intensity.replace("[", "").replace("]", "").split(",")
        parsed_fragment["hela1"].append(";".join([item.strip() for item in parsed_intensities]))

    for freg_intensity in hela_cell_2_df["Exp intensity"]:
        parsed_intensities = freg_intensity.replace("[", "").replace("]", "").split(",")
        parsed_fragment["hela2"].append(";".join([item.strip() for item in parsed_intensities]))

    deepmass_prism_df["parsed_fragment_intensities"] = parsed_fragment["deepmass"]
    hela_cell_1_df["parsed_fragment_intensities"] = parsed_fragment["hela1"]
    hela_cell_2_df["parsed_fragment_intensities"] = parsed_fragment["hela2"]

    new_deepmass_prism_df = deepmass_prism_df[["ModifiedSequence", "parsed_fragment_intensities"]]
    new_deepmass_prism_df.columns = ["peptide_sequence", "fragment_intensity"]

    new_hela_cell_1_df = hela_cell_1_df[["Peptide_sequence", "parsed_fragment_intensities"]]
    new_hela_cell_1_df.columns = ["peptide_sequence", "fragment_intensity"]

    new_hela_cell_2_df = hela_cell_2_df[["Peptide_sequence", "parsed_fragment_intensities"]]
    new_hela_cell_2_df.columns = ["peptide_sequence", "fragment_intensity"]

    return new_deepmass_prism_df, new_hela_cell_1_df, new_hela_cell_2_df


def one_hot_encoding(sequence: str):
    amino_acid_map = {aa: idx for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    return [amino_acid_map[aa] for aa in sequence]


def retrieve_dataset(input_df, max_seq_length: int, max_intensity_length: int, normalize_intens=False):
    sequences = [one_hot_encoding(seq) for seq in input_df["peptide_sequence"]]
    intensities = [
        torch.tensor([float(item) for item in intens.split(";")]) for intens in input_df["fragment_intensity"]
    ]

    max_seq_length = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [0] * (max_seq_length - len(seq)) for seq in sequences]

    if normalize_intens:
        scaler = MinMaxScaler()
        normalized_intensities = []
        for intens in intensities:
            intens = intens.view(-1, 1)  # 2D 형태로 변환
            normalized_intens = scaler.fit_transform(intens).flatten()  # 정규화 후 1D로 다시 변환
            normalized_intensities.append(torch.tensor(normalized_intens, dtype=torch.float))
        intensities = normalized_intensities

    # max_intensity_length = max(len(intensity) for intensity in intensities)
    padded_intensities = [
        torch.cat([intens, torch.zeros(max_intensity_length - len(intens))]) for intens in intensities
    ]

    # Tensor로 변환
    sequence_tensors = torch.tensor(padded_sequences, dtype=torch.long)
    intensity_tensors = torch.stack(padded_intensities)

    return PeptideDataset(sequence_tensors, intensity_tensors)


if __name__ == "__main__":
    import os

    if not os.path.exists("./data/renewal_deepmass.csv"):
        deepmass_data_path = Path("./data/uniprot-filtered-reviewed-human-peptides-ftms-hcd-charge2.tsv")
        hela1_data_path = Path("./data/HeLaCell_1.tsv")
        hela2_data_path = Path("./data/HeLaCell_2.csv")
        new_deepmass_df, new_hela1_df, new_hela2_df = restructure_data(
            deepmass_data_path, hela1_data_path, hela2_data_path
        )

        new_deepmass_df.to_csv("./data/renewal_deepmass.tsv", sep="\t")
        new_hela1_df.to_csv("./data/renewal_hela1.tsv", sep="\t")
        new_hela2_df.to_csv("./data/renewal_hela2.tsv", sep="\t")
    else:
        # new_deepmass_df = pd.read_csv("./data/renewal_deepmass.csv")
        new_hela1_df = pd.read_csv("./data/renewal_hela1.csv", low_memory=False)
        # new_hela2_df = pd.read_csv("./data/renewal_hela2.csv")
    # print(new_hela1_df)
    dataset = retrieve_dataset(new_hela1_df)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        sequences, intensities = batch
        print(sequences)
        print(intensities)
