import re
from pathlib import Path

import pandas as pd
import pymzml
from tqdm import tqdm


def get_scannum_data(psm_df: pd.DataFrame) -> dict:
    """scannum data dalkjfklajdsf

    Args:
        psm_df (pd.DataFrame): datdjflakjdsflkj

    Returns:
        dict: scannum: {peptide, charge, intensity}
    """
    res = dict()

    for row in tqdm(psm_df.itertuples(), desc="### read psm file"):
        row = row._asdict()

        peptide = row["PeptideSequence"]
        scannumber = row["ScanNum"]
        charge = row["OriginalCharge"]

        res[scannumber] = {"peptide": peptide, "charge": charge, "intensities": list()}

    return res


def main(psm: Path, mzml: Path):
    psm_df = pd.read_csv(psm, delimiter="\t")

    scannum_data = get_scannum_data(psm_df)

    for spectrum in tqdm(pymzml.run.Reader(mzml_path), desc="### read mzml file"):
        scannum = spectrum.ID
        intensities = list(spectrum.i)

        if scannum in scannum_data:
            scannum_data[scannum]["intensities"] = intensities

    for scannum, psm_info in scannum_data.items():
        peptide = psm_info["peptide"]
        frag_int = psm_info["intensities"]
        if not frag_int:
            continue

        peptide = peptide.replace("+229.163")

        print(amino_acid_sequence)

    return scannum_data


if __name__ == "__main__":
    psm_path = "./data/01CPTAC_GBM_W_PNNL_20190123_B1S1_f01.psm"
    mzml_path = "./data/01CPTAC_GBM_W_PNNL_20190123_B1S1_f01.mzML.gz"
    scannum_data = main(psm_path, mzml_path)

    # print(scannum_data)
