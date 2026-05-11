"""Unit tests for ``stats_tools.check_ML_data``."""
import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from python.Statistics_tools import check_ML_data


def _make_clean_df(n=50, n_features=5, pos_frac=0.5, with_smiles=True):
    """Build a small ML-ready dataframe that should pass every fail check."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'compound': ['c%d' % i for i in range(n)],
        'label':    (np.arange(n) < int(n * pos_frac)).astype(int),
    })
    if with_smiles:
        df['smiles'] = ['SMI%d' % i for i in range(n)]
    for f in range(n_features):
        df['F%d' % f] = rng.normal(size=n)
    return df


def _silent(fn, *args, **kwargs):
    """Run ``fn`` while swallowing its stdout; return (result, captured_text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


class CheckMLDataPassTests(unittest.TestCase):
    """Clean dataframe passes every fail check."""

    def test_clean_df_returns_true(self):
        """
        Input    : 50-row clean dataframe with compound, smiles, label, 5 features.
        Expected : returns True; printed output contains no [FAIL] lines.
        Rationale: positive control — the function must accept a valid input.
        """
        df = _make_clean_df()
        result, out = _silent(check_ML_data, df)
        # clean dataframe must return True
        self.assertTrue(result)
        # no [FAIL] should appear in the diagnostic output
        self.assertNotIn('[FAIL]', out)


class CheckMLDataFailTests(unittest.TestCase):
    """Each fail branch returns False and reports a relevant [FAIL] message."""

    def test_missing_compound_column_fails(self):
        """
        Input    : dataframe without 'compound' column.
        Expected : returns False; output mentions missing 'compound' column.
        Rationale: required-structure check is the first gate.
        """
        df = _make_clean_df().drop(columns='compound')
        result, out = _silent(check_ML_data, df)
        # missing required column must fail
        self.assertFalse(result)
        # diagnostic must mention the missing column name
        self.assertIn("missing required column 'compound'", out)

    def test_missing_label_column_fails(self):
        """
        Input    : dataframe without 'label' column.
        Expected : returns False; output mentions missing 'label' column.
        Rationale: a classifier dataset without labels is unusable.
        """
        df = _make_clean_df().drop(columns='label')
        result, out = _silent(check_ML_data, df)
        # missing label column must fail
        self.assertFalse(result)
        # diagnostic must mention the missing column name
        self.assertIn("missing required column 'label'", out)

    def test_duplicate_compound_fails(self):
        """
        Input    : two rows share the same 'compound' ID.
        Expected : returns False; output reports duplicate 'compound'.
        Rationale: duplicate IDs cause leakage between CV folds.
        """
        df = _make_clean_df()
        df.loc[0, 'compound'] = df.loc[1, 'compound']
        result, out = _silent(check_ML_data, df)
        # duplicates in the ID column must fail
        self.assertFalse(result)
        # diagnostic must call out the duplicate-compound condition
        self.assertIn("duplicate 'compound'", out)

    def test_duplicate_smiles_fails(self):
        """
        Input    : two rows share the same SMILES (different IDs).
        Expected : returns False; output reports duplicate SMILES.
        Rationale: same molecule with different IDs is a silent leak.
        """
        df = _make_clean_df()
        df.loc[0, 'smiles'] = df.loc[1, 'smiles']
        result, out = _silent(check_ML_data, df)
        # SMILES duplicates must fail (silent leak risk)
        self.assertFalse(result)
        # diagnostic must mention duplicate SMILES
        self.assertIn("duplicate SMILES", out)

    def test_label_nan_fails(self):
        """
        Input    : one row has NaN in 'label'.
        Expected : returns False; output reports label has NaN.
        Rationale: classifiers can't handle missing labels.
        """
        df = _make_clean_df().astype({'label': 'float'})
        df.loc[0, 'label'] = np.nan
        result, out = _silent(check_ML_data, df)
        # NaN in label must fail
        self.assertFalse(result)
        # diagnostic must report label NaN count
        self.assertIn("label has 1 NaN", out)

    def test_label_outside_binary_fails(self):
        """
        Input    : one row has label = 2 (outside {0, 1}).
        Expected : returns False; output reports values outside {0, 1}.
        Rationale: this function is binary-classification specific.
        """
        df = _make_clean_df()
        df.loc[0, 'label'] = 2
        result, out = _silent(check_ML_data, df)
        # non-binary label must fail
        self.assertFalse(result)
        # diagnostic must mention the out-of-range label
        self.assertIn("label has values outside", out)

    def test_label_single_class_fails(self):
        """
        Input    : every row has label = 0.
        Expected : returns False; output reports only one class.
        Rationale: a classifier needs both classes represented.
        """
        df = _make_clean_df()
        df['label'] = 0
        result, out = _silent(check_ML_data, df)
        # single-class label must fail
        self.assertFalse(result)
        # diagnostic must explain the cause
        self.assertIn("only one class", out)

    def test_non_numeric_feature_fails(self):
        """
        Input    : feature column F0 contains strings.
        Expected : returns False; output reports non-numeric feature.
        Rationale: sklearn / XGBoost reject object-dtype feature matrices.
        """
        df = _make_clean_df()
        df['F0'] = ['a'] * len(df)
        result, out = _silent(check_ML_data, df)
        # non-numeric features must fail
        self.assertFalse(result)
        # diagnostic must mention non-numeric columns
        self.assertIn("non-numeric feature columns", out)

    def test_nan_in_features_fails(self):
        """
        Input    : one feature cell is NaN.
        Expected : returns False; output reports NaN in features.
        Rationale: NaN in features crashes most estimators on .fit().
        """
        df = _make_clean_df()
        df.loc[0, 'F0'] = np.nan
        result, out = _silent(check_ML_data, df)
        # NaN in features must fail
        self.assertFalse(result)
        # diagnostic must report NaN in features
        self.assertIn("NaN in features", out)

    def test_inf_in_features_fails(self):
        """
        Input    : one feature cell is +inf.
        Expected : returns False; output reports +/-inf in features.
        Rationale: inf propagates through gradients and breaks training.
        """
        df = _make_clean_df()
        df.loc[0, 'F0'] = np.inf
        result, out = _silent(check_ML_data, df)
        # +/-inf in features must fail
        self.assertFalse(result)
        # diagnostic must mention the inf condition
        self.assertIn("+/-inf in features", out)


class CheckMLDataWarnTests(unittest.TestCase):
    """Warning branches don't change the True return value."""

    def test_extreme_class_imbalance_warns_but_passes(self):
        """
        Input    : 200-row df with 1 positive (0.5% positives).
        Expected : returns True; output contains [WARN] about class imbalance.
        Rationale: imbalance is a soft signal — caller should be told but not blocked.
        """
        df = _make_clean_df(n=200, pos_frac=0.005)  # exactly 1 positive
        result, out = _silent(check_ML_data, df)
        # warning must not block the function
        self.assertTrue(result)
        # diagnostic must include the imbalance warning
        self.assertIn("extreme class imbalance", out)

    def test_zero_variance_feature_warns_but_passes(self):
        """
        Input    : feature column F0 is a constant (zero variance).
        Expected : returns True; output contains [WARN] about constant features.
        Rationale: constant features are useless but harmless — warn, don't fail.
        """
        df = _make_clean_df()
        df['F0'] = 0
        result, out = _silent(check_ML_data, df)
        # zero-variance is a warning, function should still return True
        self.assertTrue(result)
        # diagnostic must mention the constant column
        self.assertIn("constant (zero-variance)", out)

    def test_small_dataset_warns_but_passes(self):
        """
        Input    : 10-row dataframe (below the 30-sample warning threshold).
        Expected : returns True; output contains [WARN] about small dataset.
        Rationale: small datasets are a soft caution, not a hard fail.
        """
        df = _make_clean_df(n=10)
        result, out = _silent(check_ML_data, df)
        # small dataset is a warning, function should still return True
        self.assertTrue(result)
        # diagnostic must mention the small-dataset warning
        self.assertIn("very small dataset", out)


if __name__ == '__main__':
    unittest.main()
