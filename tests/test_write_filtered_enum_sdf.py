"""Unit tests for ``rdkit_tools.write_filtered_enum_sdf`` and
``rdkit_tools.get_smiles_df_from_enum``."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from python.Rdkit_tools import write_filtered_enum_sdf, get_smiles_df_from_enum


def _make_record(r1_code, r2_code, body):
    """Build a minimal synthetic SDF record (no leading blank title line)."""
    return (
        '%s\nM  END\n'
        '>  <R1_Code>\n%s\n\n'
        '>  <R2_Code>\n%s\n\n'
    ) % (body, r1_code, r2_code)


def _write_sdf(path, records):
    """Concatenate records with the $$$$ delimiter and write to disk."""
    with open(path, 'w') as f:
        for rec in records:
            f.write(rec + '$$$$\n')


def _make_real_format_record(r1_code, r2_code):
    """
    Build a record mimicking what MarvinSketch writes: blank title line,
    software/timestamp line, blank comment line, V2000 counts line, atom
    block, ``M  END``, then SDF tags. Uses a 1-atom carbon mol so RDKit can
    parse it on round-trip.
    """
    return (
        '\n'                              # line 1: blank title
        '  Synthetic 01010100002D\n'      # line 2: software
        '\n'                              # line 3: blank comment
        '  1  0  0  0  0  0            999 V2000\n'
        '    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n'
        'M  END\n'
        '>  <R1_Code>\n%s\n\n'
        '>  <R2_Code>\n%s\n\n'
    ) % (r1_code, r2_code)


class WriteFilteredEnumSdfTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src_dir  = self.tmp.name
        self.out_path = os.path.join(self.src_dir, 'out', 'filtered.sdf')

        records_a = [
            _make_record('EN300-A1', 'EN300-B1', body='molblock_A1B1'),
            _make_record('EN300-A2', 'EN300-B2', body='molblock_A2B2'),
        ]
        records_c = [
            _make_record('EN300-C1', 'EN300-D1', body='molblock_C1D1'),
        ]
        _write_sdf(os.path.join(self.src_dir, 'fileA.sdf'), records_a)
        _write_sdf(os.path.join(self.src_dir, 'fileC.sdf'), records_c)

    def tearDown(self):
        self.tmp.cleanup()

    def _read_out(self):
        with open(self.out_path) as f:
            return f.read()

    def test_basic_single_pick(self):
        """
        Input    : 2-record SDF (A1B1, A2B2); df selects only A1B1 with prob 0.9.
        Expected : output has exactly 1 record (A1B1 only), with the
                   ``predicted_probas`` tag and value 0.9.
        Rationale: happy-path filter — make sure selection actually filters and
                   the prob tag is appended.
        """
        df = pd.DataFrame([
            {'R1_Code': 'EN300-A1', 'R2_Code': 'EN300-B1',
             'filename': 'fileA', 'predicted_probas': 0.9},
        ])
        n = write_filtered_enum_sdf(df, self.src_dir, self.out_path)
        # function must report one record kept
        self.assertEqual(n, 1)

        text = self._read_out()
        # exactly one $$$$ delimiter ⇒ one record on disk
        self.assertEqual(text.count('$$$$'), 1)
        # selected record's molblock must appear
        self.assertIn('molblock_A1B1', text)
        # non-selected record's molblock must NOT appear
        self.assertNotIn('molblock_A2B2', text)
        # the predicted_probas tag header must be present
        self.assertIn('>  <predicted_probas>', text)
        # the actual prob value must appear
        self.assertIn('0.9', text)

    def test_pick_all_records_in_file(self):
        """
        Input    : 2-record source SDF; df selects both records.
        Expected : output preserves both records (count and both molblocks).
        Rationale: verifies we don't accidentally drop records when all are picked.
        """
        df = pd.DataFrame([
            {'R1_Code': 'EN300-A1', 'R2_Code': 'EN300-B1',
             'filename': 'fileA', 'predicted_probas': 0.5},
            {'R1_Code': 'EN300-A2', 'R2_Code': 'EN300-B2',
             'filename': 'fileA', 'predicted_probas': 0.6},
        ])
        n = write_filtered_enum_sdf(df, self.src_dir, self.out_path)
        # both records should be reported as kept
        self.assertEqual(n, 2)

        text = self._read_out()
        # two $$$$ delimiters ⇒ two records on disk
        self.assertEqual(text.count('$$$$'), 2)
        # first selected record present
        self.assertIn('molblock_A1B1', text)
        # second selected record present
        self.assertIn('molblock_A2B2', text)

    def test_pick_across_multiple_files(self):
        """
        Input    : 2 source SDFs (fileA, fileC); df picks 1 record from each.
        Expected : output contains both selected records, drawn from their
                   respective source files.
        Rationale: verifies multi-file iteration and that pairs are matched
                   per-source rather than globally.
        """
        df = pd.DataFrame([
            {'R1_Code': 'EN300-A1', 'R2_Code': 'EN300-B1',
             'filename': 'fileA', 'predicted_probas': 0.7},
            {'R1_Code': 'EN300-C1', 'R2_Code': 'EN300-D1',
             'filename': 'fileC', 'predicted_probas': 0.8},
        ])
        n = write_filtered_enum_sdf(df, self.src_dir, self.out_path)
        # one record from each source ⇒ 2 written
        self.assertEqual(n, 2)

        text = self._read_out()
        # record from fileA present
        self.assertIn('molblock_A1B1', text)
        # record from fileC present
        self.assertIn('molblock_C1D1', text)

    def test_missing_pair_silently_skipped(self):
        """
        Input    : df contains 1 valid (R1, R2) pair plus 1 nonexistent pair,
                   both pointing at the same source file.
        Expected : output contains only the valid record; no exception is raised.
        Rationale: the function should be tolerant of stale or unmatched keys
                   instead of erroring on the first miss.
        """
        df = pd.DataFrame([
            {'R1_Code': 'EN300-A1', 'R2_Code': 'EN300-B1',
             'filename': 'fileA', 'predicted_probas': 0.9},
            {'R1_Code': 'EN300-MISSING', 'R2_Code': 'EN300-NOPE',
             'filename': 'fileA', 'predicted_probas': 0.99},
        ])
        n = write_filtered_enum_sdf(df, self.src_dir, self.out_path)
        # only the valid pair should be reported as kept
        self.assertEqual(n, 1)
        # exactly one $$$$ delimiter on disk
        self.assertEqual(self._read_out().count('$$$$'), 1)

    def test_no_prob_column_does_not_add_tag(self):
        """
        Input    : df has only R1_Code / R2_Code / filename — no prob column.
        Expected : output has the selected record but no <predicted_probas> tag.
        Rationale: the prob column is optional; the function must not invent
                   a tag when the source df lacks one.
        """
        df = pd.DataFrame([
            {'R1_Code': 'EN300-A1', 'R2_Code': 'EN300-B1', 'filename': 'fileA'},
        ])
        n = write_filtered_enum_sdf(df, self.src_dir, self.out_path)
        # one record kept
        self.assertEqual(n, 1)
        # no predicted_probas tag should be present anywhere in the output
        self.assertNotIn('<predicted_probas>', self._read_out())

    def test_record_kept_verbatim(self):
        """
        Input    : df selects A2B2; source record body and tag block known.
        Expected : original molblock + R1/R2 tag block appears byte-for-byte
                   in the output.
        Rationale: the function copies records verbatim — it must not re-render
                   or canonicalize the molblock or reformat the tags.
        """
        df = pd.DataFrame([
            {'R1_Code': 'EN300-A2', 'R2_Code': 'EN300-B2',
             'filename': 'fileA', 'predicted_probas': 0.42},
        ])
        write_filtered_enum_sdf(df, self.src_dir, self.out_path)
        text = self._read_out()
        original_body = (
            'molblock_A2B2\nM  END\n'
            '>  <R1_Code>\nEN300-A2\n\n'
            '>  <R2_Code>\nEN300-B2'
        )
        # the original record body must survive byte-for-byte (no rewriting)
        self.assertIn(original_body, text)

    def test_creates_missing_output_dir(self):
        """
        Input    : out_path inside a nested directory that doesn't exist yet.
        Expected : function creates the parent directories and writes the file.
        Rationale: convenience — caller shouldn't have to ``os.makedirs`` first.
        """
        df = pd.DataFrame([
            {'R1_Code': 'EN300-A1', 'R2_Code': 'EN300-B1',
             'filename': 'fileA', 'predicted_probas': 0.5},
        ])
        nested = os.path.join(self.src_dir, 'a', 'b', 'c', 'out.sdf')
        write_filtered_enum_sdf(df, self.src_dir, nested)
        # nested output file must exist after the call
        self.assertTrue(os.path.exists(nested))


class WriteFilteredEnumSdfRealFormatTests(unittest.TestCase):
    """
    Regression tests reproducing the real MarvinSketch SDF layout (blank
    title line, software line, blank comment line, V2000 molblock) and
    validating that the output can be re-parsed by RDKit.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src_dir  = self.tmp.name
        self.out_path = os.path.join(self.src_dir, 'out', 'filtered.sdf')

        records = [
            _make_real_format_record('EN300-A1', 'EN300-B1'),
            _make_real_format_record('EN300-A2', 'EN300-B2'),
        ]
        with open(os.path.join(self.src_dir, 'fileA.sdf'), 'w') as f:
            for rec in records:
                f.write(rec + '$$$$\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_round_trips_through_rdkit(self):
        """
        Input    : 2-record SDF in MarvinSketch layout (with blank title line).
        Expected : after write_filtered_enum_sdf, RDKit's SDMolSupplier parses
                   every record without returning None, and the SDF tags
                   (R1_Code, R2_Code, predicted_probas) survive cleanly.
        Rationale: regression for the "Cannot convert 'M  ' to unsigned int"
                   bug caused by stripping the molfile's blank title line.
        """
        df = pd.DataFrame([
            {'R1_Code': 'EN300-A1', 'R2_Code': 'EN300-B1',
             'filename': 'fileA', 'predicted_probas': 0.91},
            {'R1_Code': 'EN300-A2', 'R2_Code': 'EN300-B2',
             'filename': 'fileA', 'predicted_probas': 0.42},
        ])
        n = write_filtered_enum_sdf(df, self.src_dir, self.out_path)
        # both records should be reported as kept
        self.assertEqual(n, 2)

        from rdkit import Chem
        mols = list(Chem.SDMolSupplier(self.out_path))
        # RDKit should see 2 records in the output file
        self.assertEqual(len(mols), 2)
        for m in mols:
            # parser must succeed on every record (None == parse failure)
            self.assertIsNotNone(m, 'RDKit failed to parse a written record')

        codes = sorted(
            (m.GetProp('R1_Code'), m.GetProp('R2_Code'), m.GetProp('predicted_probas'))
            for m in mols
        )
        # all three SDF tags must round-trip with their exact values
        self.assertEqual(
            codes,
            [('EN300-A1', 'EN300-B1', '0.91'),
             ('EN300-A2', 'EN300-B2', '0.42')],
        )


class SdfCsvSdfCsvRoundTripTests(unittest.TestCase):
    """
    End-to-end round trip:
        sdf  --get_smiles_df_from_enum-->  df1  (acts as the "csv")
        df1  --write_filtered_enum_sdf-->  sdf'
        sdf' --get_smiles_df_from_enum-->  df2
    The two dataframes must be equal.
    """

    def setUp(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        self.tmp = tempfile.TemporaryDirectory()
        self.src_dir  = self.tmp.name
        self.src_path = os.path.join(self.src_dir, 'src.sdf')
        self.out_path = os.path.join(self.src_dir, 'out', 'roundtrip.sdf')

        # Use real, canonicalizable SMILES so MolToSmiles is deterministic and
        # MolBlock-generated coordinates round-trip cleanly through the SDF.
        seed = [
            ('CCO',         'EN300-A0', 'EN300-B0'),
            ('c1ccccc1N',   'EN300-A1', 'EN300-B1'),
            ('CC(=O)O',     'EN300-A2', 'EN300-B2'),
            ('C1CCCCC1',    'EN300-A3', 'EN300-B3'),
        ]
        with open(self.src_path, 'w') as f:
            for smi, r1, r2 in seed:
                mol = Chem.MolFromSmiles(smi)
                AllChem.Compute2DCoords(mol)
                f.write(Chem.MolToMolBlock(mol))
                f.write('>  <R1_Code>\n%s\n\n' % r1)
                f.write('>  <R2_Code>\n%s\n\n' % r2)
                f.write('$$$$\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_sdf_to_df_to_sdf_to_df_is_idempotent(self):
        """
        Input    : 4-record SDF built from real SMILES via RDKit.
        Expected : sdf -> df1 -> sdf' -> df2 yields df1 equal to df2 by
                   ``pd.testing.assert_frame_equal``.
        Rationale: full pipeline integrity check — guarantees that the two
                   SDF helpers compose without information loss.
        """
        df1 = get_smiles_df_from_enum(self.src_path)
        # the 4 seed records should produce 4 rows in df1
        self.assertEqual(len(df1), 4)

        df_filter = df1.copy()
        df_filter['filename'] = 'src'
        n = write_filtered_enum_sdf(df_filter, self.src_dir, self.out_path)
        # write_filtered_enum_sdf must report keeping every row from df1
        self.assertEqual(n, len(df1))

        df2 = get_smiles_df_from_enum(self.out_path)
        # df1 and df2 must be exactly equal — full round-trip identity
        pd.testing.assert_frame_equal(df1, df2)


class CsvToSdfToDfMatchesCsvTests(unittest.TestCase):
    """
    Pipeline mirror: a CSV with smiles/R1_Code/R2_Code/compound/filename/
    predicted_probas (the format produced in the notebook) is filtered through
    ``write_filtered_enum_sdf`` and then re-extracted via
    ``get_smiles_df_from_enum``. The re-extracted df must match the CSV's
    smiles/R1_Code/R2_Code columns row-for-row.
    """

    def setUp(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        self.tmp = tempfile.TemporaryDirectory()
        self.src_dir   = self.tmp.name
        self.src_fname = 'src'
        self.src_path  = os.path.join(self.src_dir, self.src_fname + '.sdf')
        self.csv_path  = os.path.join(self.src_dir, 'predicted.csv')
        self.out_path  = os.path.join(self.src_dir, 'out', 'filtered.sdf')

        # (smiles, R1_Code, R2_Code, predicted_probas)
        seed = [
            ('CCO',         'EN300-A0', 'EN300-B0', 0.95),
            ('c1ccccc1N',   'EN300-A1', 'EN300-B1', 0.88),
            ('CC(=O)O',     'EN300-A2', 'EN300-B2', 0.72),
            ('C1CCCCC1',    'EN300-A3', 'EN300-B3', 0.61),
        ]

        # 1. Write the source SDF that backs each CSV row.
        with open(self.src_path, 'w') as f:
            for smi, r1, r2, _ in seed:
                mol = Chem.MolFromSmiles(smi)
                AllChem.Compute2DCoords(mol)
                f.write(Chem.MolToMolBlock(mol))
                f.write('>  <R1_Code>\n%s\n\n' % r1)
                f.write('>  <R2_Code>\n%s\n\n' % r2)
                f.write('$$$$\n')

        # 2. Write a CSV mirroring the notebook output, with canonical SMILES
        #    so the comparison after MolToSmiles is exact. Sorted by prob desc
        #    on purpose (the notebook does the same), to verify the test is
        #    order-insensitive.
        rows = []
        for smi, r1, r2, prob in seed:
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            rows.append({
                'smiles':           canonical,
                'R1_Code':          r1,
                'R2_Code':          r2,
                'compound':         '%s_%s' % (r1, r2),
                'filename':         self.src_fname,
                'predicted_probas': prob,
            })
        (pd.DataFrame(rows)
           .sort_values('predicted_probas', ascending=False)
           .to_csv(self.csv_path, index=False))

    def tearDown(self):
        self.tmp.cleanup()

    def test_reopened_sdf_df_matches_original_csv(self):
        """
        Input    : a CSV with smiles/R1_Code/R2_Code/compound/filename/
                   predicted_probas plus the source SDF backing those rows;
                   CSV is sorted by predicted_probas (different order from SDF).
        Expected : feeding the CSV through ``write_filtered_enum_sdf`` and
                   reopening with ``get_smiles_df_from_enum`` yields a df
                   whose [smiles, R1_Code, R2_Code] columns match the CSV's
                   exactly (after a stable sort to factor out record order).
        Rationale: mirrors the notebook's "write filtered SDF, reopen SDF"
                   sanity check — guarantees the CSV→SDF→df pipeline preserves
                   structure-identifying columns.
        """
        csv_df = pd.read_csv(self.csv_path)
        # CSV must carry the columns the notebook pipeline produces
        for col in ('smiles', 'R1_Code', 'R2_Code', 'filename'):
            self.assertIn(col, csv_df.columns)

        n = write_filtered_enum_sdf(csv_df, self.src_dir, self.out_path)
        # write_filtered_enum_sdf must report keeping every CSV row
        self.assertEqual(n, len(csv_df))

        reopened = get_smiles_df_from_enum(self.out_path)
        # reopened df has the same number of rows as the CSV
        self.assertEqual(len(reopened), len(csv_df))

        # compare the structure-identifying columns, sorted for stable order
        cols = ['smiles', 'R1_Code', 'R2_Code']
        sort_keys = ['R1_Code', 'R2_Code']
        csv_subset = csv_df[cols].sort_values(sort_keys).reset_index(drop=True)
        sdf_subset = reopened[cols].sort_values(sort_keys).reset_index(drop=True)
        # the two frames must match row-for-row on smiles/R1_Code/R2_Code
        pd.testing.assert_frame_equal(csv_subset, sdf_subset)


if __name__ == '__main__':
    unittest.main()
