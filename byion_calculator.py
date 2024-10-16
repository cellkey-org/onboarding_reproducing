"""This module is developed for peptide-spectrum prediction
but it can be used any other purpose 
"""

import re

elements = {
    "C": 12.0,
    "H": 1.007825,
    "H_ion": 1.007276,
    "Acetyl": 42.0106,
    "O": 15.994915,
    "N": 14.003074,
    "S": 31.972071,
    "Na": 22.9897692809,
    "H2O": 18.01056,
    "Y1*": 83.03657,
    "NH3": 17.026549,
    "NH2": 16.01818,
    "e_ion": 0.000549,
    "C13-C12": 1.003355,
    "NH-O": -0.984016,
    "P": 30.97321,
}
modification_mass = {
    "None": 0.0,
    "Acetyl": 42.0106,
    "iTRAQ4plex": 144.102063,
    "iTRAQ8plex": 304.205360,
    "TMT0plex": 224.152478,
    "TMT2plex": 225.155833,
    "TMT6plex": 229.162932,
    "TMT10plex": 229.162932,
    "TMT16plex": 304.207,
}

aminoacid_mass = {
    "G": 57.021464,
    "A": 71.037110,
    "S": 87.032020,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047680,
    "C": 103.009190,
    "I": 113.084060,
    "L": 113.084060,
    "N": 114.042930,
    "D": 115.026940,
    "E": 129.042590,
    "Q": 128.058580,
    "M": 131.040490,
    "H": 137.058910,
    "F": 147.068410,
    "R": 156.101110,
    "Y": 163.063330,
    "W": 186.079310,
    "K": 128.094960,
    "X": 0.000000,
    "B": 114.53494,
    "Z": 128.55059,
    "*": 15.994915,
    "U": 0.000000,
    "#": 57.021464,
    "m": 147.03485,
    "n": 114.042930,
}


def ion_calculator(peptide_sequence: str, ion_type="b") -> tuple[list, list]:
    """
    Calculate ion and mass from the peptide sequence. with consideration of modification
    This is only considered as MSGF search result, which is that sequence always contains modification mass
        ex> +229.163C+57.021HAANPNGR

    Args:
        peptide_sequence (str): peptide sequence

    Returns:
        tuple[list, list]: bion list, bion mass list (same order)
    """
    bions = list()
    bion_mass = list()

    # set base mass
    base_mass = elements.get("H")
    if ion_type == "y":
        base_mass = base_mass * 3

    # pre-calculate aa mass list for the peptide sequence
    sequence_mass = get_sequence_mass(peptide_sequence)

    # calc ion mass
    for index in range(len(peptide_sequence)):
        partial_sequence = peptide_sequence[: index + 1]
        bions.append(partial_sequence)
        bion_mass.append(sum(sequence_mass[: index + 1]))

    return bions, bion_mass


def get_sequence_mass(peptide_sequence: str) -> list:
    """
    Get aa mass list as peptide sequence order

    Args:
        peptide_sequence (str): peptide sequence with modifications

    Returns:
        list: mass list [+229.163, 57.02]
    """
    sequence_mass = list()

    for aa in peptide_sequence:
        sequence_mass.append(aminoacid_mass[aa])

    return sequence_mass
