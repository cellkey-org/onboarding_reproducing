import ion_calculator
import pytest

tolerance = 1e-5  # 소숫점 근사를 위해서


def test_bion_calculator():
    """
    test b, y ion calculator
    b ion = subset of string of peptide with sequential order (3'->5')
    y ion = same as b ion but the seqeunce should be reversed (5'->3')

    goal is getting ion list and mass of each ions
    """

    peptide_sequence = "CDN"
    expected_bion = ["C", "CD", "CDN"]
    expected_yion = ["N", "ND", "NDC"]

    expected_bion_mass = [103.009190, 103.009190 + 115.026943, 103.009190 + 115.026943 + 114.042927]
    expected_yion_mass = [114.042927, 115.026943 + 114.042927, 103.009190 + 115.026943 + 114.042927]

    calculated_bion, calculated_bion_mass = ion_calculator.ion_calculator(peptide_sequence)
    calculated_yion, calculated_yion_mass = ion_calculator.ion_calculator(peptide_sequence[::-1], ion_type="y")

    assert expected_bion == calculated_bion
    assert expected_bion_mass == pytest.approx(calculated_bion_mass, abs=tolerance)

    assert expected_yion == calculated_yion
    assert expected_yion_mass == pytest.approx(calculated_yion_mass, abs=tolerance)


def test_get_sequence_mass():
    peptide_sequence = "CDN"

    expected_sequence_mass = [103.009190, 115.026943, 114.042927]
    calculated_sequence_mass = ion_calculator.get_sequence_mass(peptide_sequence)

    assert expected_sequence_mass == pytest.approx(calculated_sequence_mass, abs=tolerance)
