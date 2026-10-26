"""Unit tests for ``python/functions.py``.

Coverage tiers:
  - **Tier A (behavioural):** deterministic data-transform / enrichment / stats
    helpers — assert real expected values.
  - **Tier B (smoke):** plot / parallel / file-IO helpers — run under the Agg
    backend (or a tmp dir) and assert they don't raise and return / write the
    right kind of thing (can't assert pixels).
  - **Tier C (skipped):** the two interactive 3D builders and the live
    OpenTargets call — explicit ``@unittest.skip`` so the gaps are visible.

All fixtures are synthetic; any chemistry uses public SMILES / HGNC symbols only
(no project data). File fixtures live in ``tests/data/``.

Run from the project root:

    python -m unittest tests.test_functions -v
"""
import sys
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                       # headless — before any pyplot import
import matplotlib.pyplot as plt

# Project root + Scripts on sys.path so imports work regardless of CWD.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/home/gtamo/Scripts')
DATA = Path(__file__).resolve().parent / 'data'

import python.functions as fn


# ----------------------------------------------------------------------
# Shared fixtures (synthetic; public symbols only)
# ----------------------------------------------------------------------

def _gene2term(rows, collection='C'):
    """rows: list of (gene, term_id, term_name) -> gene2term long frame."""
    return pd.DataFrame(rows, columns=['gene', 'term_id', 'term_name']).assign(collection=collection)


def _func_enrich(rows):
    """rows: list of (compound, function, gsea_NES) -> tidy enrichment frame."""
    return pd.DataFrame(rows, columns=['compound', 'function', 'gsea_NES'])


def _ms_two_tranches():
    """Small MS metadata across two screening tranches (every activity bin present)."""
    return pd.DataFrame({'date': ['2026-06-01'] * 3 + ['2026-06-16'] * 3,
                         'activity': ['Silent', 'Low (2-10)', 'High (>25)',
                                      'Silent', 'Single (1)', 'Medium (11-25)']})


class _MockReg:
    """Stand-in for the ML_Reg module: a fixed-R² CV harness so plate-drop
    orchestration can be tested without the real regressor."""
    def __init__(self, r2=0.5):
        self.r2 = r2

    def run_K_Fold_Xval_Regression(self, ml, **kw):
        return None, pd.DataFrame({'real': [0.0], 'pred': [0.0]})

    def get_reg_metrics_from_preddf(self, df_pred, v=False):
        return {'r2': self.r2}


# ======================================================================
# Tier A — MS / df_raw collapse
# ======================================================================

class TestMSCollapse(unittest.TestCase):

    def test_keep_latest_batch_then_date(self):
        """A screened under batch 1 & 2 -> keep batch 2 (both gene rows); B
        under batch 1 on two dates -> keep the later date. All gene rows of the
        winning (batch,date) survive."""
        df = pd.DataFrame([
            ('A', 1, '2024-01-01', 'g1'), ('A', 1, '2024-01-01', 'g2'),
            ('A', 2, '2024-01-01', 'g1'), ('A', 2, '2024-01-01', 'g2'),
            ('B', 1, '2024-01-01', 'g1'), ('B', 1, '2024-02-01', 'g1'),
        ], columns=['compound', 'batch', 'date', 'genes'])
        out = fn.keep_latest_batch_per_compound(df, verbose=False)
        # A keeps only its highest batch (2), both gene rows
        self.assertEqual(set(out.loc[out['compound'] == 'A', 'batch']), {2})
        # B keeps only its latest date
        self.assertEqual(set(out.loc[out['compound'] == 'B', 'date']), {'2024-02-01'})
        # winning screens: A(batch2)=2 rows + B(latest date)=1 row = 3
        self.assertEqual(len(out), 3)

    def test_collapse_ms_latest_measurement_keeps_value_stamps_earliest_date(self):
        """SRB1 (ndown=3, 2026-06-01) + (ndown=5, 2026-06-16) collapses to one
        row with the LATEST measurement (ndown=5) stamped with the EARLIEST date
        (2026-06-01)."""
        ms = pd.DataFrame({'compound': ['SRB1', 'SRB1', 'SRB2'],
                           'ndown': [3, 5, 1],
                           'date': ['2026-06-01', '2026-06-16', '2026-06-10']})
        out = fn.collapse_ms_latest_measurement(ms, verbose=False)
        row = out[out['compound'] == 'SRB1'].iloc[0]
        # one row per compound
        self.assertEqual(len(out), 2)
        # latest measurement kept
        self.assertEqual(row['ndown'], 5)
        # stamped with earliest date
        self.assertEqual(pd.Timestamp(row['date']), pd.Timestamp('2026-06-01'))


# ======================================================================
# Tier A — proteomics loader (synthetic raw + metadata exports)
# ======================================================================

class TestLoadProteomics(unittest.TestCase):

    def test_serac_mode_dedups_filters_and_joins(self):
        """serac recipe: drop non-SERAC rows, keep the latest-dated row per
        Molecule Name, drop noisy plates, and keep only molecule-batches present
        in the metadata. SRB-0388 (VENDOR) is dropped from MS; SRB-0386 survives
        MS but its only raw rows are on Plate12 (dropped); SRB-0385 keeps its
        2026-04-29 row over the 2026-04-01 one."""
        df_raw, MS = fn.load_proteomics_data(
            str(DATA / 'sample_px_raw_serac.csv'),
            str(DATA / 'sample_px_meta_serac.csv'),
            verbose=False)
        # non-SERAC compound filtered out of the metadata
        self.assertEqual(set(MS['Molecule Name']),
                         {'SRB-0000385', 'SRB-0000386', 'SRB-0000387'})
        # one row per compound after the latest-date dedup
        self.assertEqual(len(MS), 3)
        # SRB-0385 keeps its latest screen date
        kept = MS.loc[MS['Molecule Name'] == 'SRB-0000385',
                      'MSData - Proteomics activities: Date'].iloc[0]
        self.assertEqual(pd.Timestamp(kept), pd.Timestamp('2026-04-29'))
        # raw keeps only 0385 (0386 dropped by Plate12, 0388 not in MS)
        self.assertEqual(set(df_raw['compound']), {'SRB-0000385', 'SRB-0000387'})
        # MoleculeBatchID split into compound + batch
        self.assertEqual(set(df_raw['batch']), {'001'})

    def test_cddvault_mode_renames_and_filters(self):
        """cddvault recipe: rename SMILES->smiles + raw 'unique'->'uniquecontrast',
        auto-detect the 'Batch Molecule-Batch ID' join key, drop Plate15, and keep
        only molecule-batches in the metadata (SRB-0392 absent from metadata is
        dropped by the inner join)."""
        df_raw, MS = fn.load_proteomics_data(
            str(DATA / 'sample_px_raw_cddvault.csv'),
            str(DATA / 'sample_px_meta_cddvault.csv'),
            mode='cddvault', verbose=False)
        # SMILES renamed on the metadata side
        self.assertIn('smiles', MS.columns)
        self.assertNotIn('SMILES', MS.columns)
        # raw 'unique' aligned to the earlier tranche's 'uniquecontrast'
        self.assertIn('uniquecontrast', df_raw.columns)
        self.assertNotIn('unique', df_raw.columns)
        # only 0390 survives (0391 dropped by Plate15, 0392 not in metadata)
        self.assertEqual(set(df_raw['compound']), {'SRB-0000390'})

    def test_bad_mode_raises(self):
        """An unknown mode raises ValueError rather than silently mis-loading."""
        # only 'serac' / 'cddvault' are valid recipes
        with self.assertRaises(ValueError):
            fn.load_proteomics_data(str(DATA / 'sample_px_raw_serac.csv'),
                                    str(DATA / 'sample_px_meta_serac.csv'),
                                    mode='nonsense', verbose=False)


# ======================================================================
# Tier A — FBX tranche loader (synthetic MEASURE / REPORT exports)
# ======================================================================

class TestLoadFbxTranche(unittest.TestCase):

    FBX = DATA / 'fbx'

    def test_main_tranche_crosswalk_and_filters(self):
        """One FBX folder -> (df_raw_fbx, MS_fbx). REPORT maps uniquecontrast ->
        srbnumber -> compound+batch; KO plate dropped (uc5), control (SRB-0099)
        + contaminant (SRB-0002) dropped, QC contrast with no srbnumber (ucQC)
        dropped; MS keeps the max-nr_down row per compound (SRB-0001 -> 8)."""
        dfx, msx = fn.load_fbx_tranche(str(self.FBX / '20260616'),
                                       control_compounds=['SRB-0000099'],
                                       contaminants=['SRB-0000002'], verbose=False)
        # exact target schemas
        self.assertEqual(list(dfx.columns), fn.FBX_DFRAW_COLS)
        self.assertEqual(list(msx.columns), fn.FBX_MS_COLS)
        # only the two clean compounds survive every filter
        self.assertEqual(set(dfx['compound']), {'SRB-0000001', 'SRB-0000005'})
        # uc1(2 genes)+uc2(1)+uc6(1) = 4 per-gene rows; KO/control/contaminant/QC gone
        self.assertEqual(len(dfx), 4)
        # KO plate removed entirely
        self.assertNotIn('PwKO1', set(dfx['MSPlate']))
        # srbnumber split into compound + batch
        self.assertEqual(set(dfx['batch']), {'001'})
        # MS: one row per compound, representative = max nr_down
        self.assertEqual(set(msx['compound']), {'SRB-0000001', 'SRB-0000005'})
        self.assertEqual(msx.loc[msx['compound'] == 'SRB-0000001', 'ndown'].item(), 8)
        # origin tag + parsed tranche date
        self.assertTrue((msx['origin'] == 'MS20260616').all())
        self.assertEqual(pd.Timestamp(msx['date'].iloc[0]), pd.Timestamp('2026-06-16'))

    def test_reexport_suffix_folder_globs_and_dates(self):
        """A '_2' re-export folder (files named *_FBX_MEASURE_02.csv) is globbed
        despite the suffix; origin keeps the full folder name while the date
        parses from the leading YYYYMMDD."""
        dfx, msx = fn.load_fbx_tranche(str(self.FBX / '20260616_2'), verbose=False)
        # the *_02 file was found and loaded
        self.assertEqual(set(dfx['compound']), {'SRB-0000001'})
        # origin keeps the distinct folder name ...
        self.assertTrue((msx['origin'] == 'MS20260616_2').all())
        # ... but the date still parses (leading 8 digits)
        self.assertEqual(pd.Timestamp(msx['date'].iloc[0]), pd.Timestamp('2026-06-16'))


# ======================================================================
# Tier A — Px database (RDS) source + MS derived from df_raw
# ======================================================================

def _rds_tables():
    """The four FBX tables a fake Px database serves, in the real column order.

    uc1 is inserted two times (id 1, then id 5 with a different logfc); uc2 sits in both the main
    and the glues table; uc8 has no date in PX_DATES. uc4 exists only in the archive.
    """
    measure = pd.DataFrame([
        # uc1, gene A: the OLD insert, then the NEW one
        dict(id=1, pg='P1', genes='AAA', uniquecontrast='uc1', logfc=-0.5, pvalue=1e-3, adjpval=0.01, significant=1, plate='Pw01'),
        dict(id=5, pg='P1', genes='AAA', uniquecontrast='uc1', logfc=-2.0, pvalue=1e-9, adjpval=0.01, significant=1, plate='Pw01'),
        dict(id=2, pg='P2', genes='BBB', uniquecontrast='uc2', logfc=-9.9, pvalue=1e-9, adjpval=0.01, significant=1, plate='Pw02'),
        dict(id=3, pg='P3', genes='CCC', uniquecontrast='uc3', logfc=-1.0, pvalue=1e-4, adjpval=0.02, significant=1, plate='Pw03'),
        dict(id=4, pg='P4', genes='DDD', uniquecontrast='uc8', logfc=-1.0, pvalue=1e-4, adjpval=0.02, significant=1, plate='Pw08'),
    ])
    glues = pd.DataFrame([
        dict(id=7, pg='P2', genes='BBB', uniquecontrast='uc2', logfc=-3.0, pvalue=1e-6, adjpval=0.01, significant=1, plate='Pw02'),
    ])
    report = pd.DataFrame([
        dict(id=1, uniquecontrast='uc1', srbnumber='SRB-0000001-001', plate='Pw01', nr_down=4, activity='Low (2-10)'),
        dict(id=2, uniquecontrast='uc2', srbnumber='SRB-0000002-001', plate='Pw02', nr_down=1, activity='Single (1)'),
        dict(id=3, uniquecontrast='uc3', srbnumber='SRB-0000003-001', plate='Pw03', nr_down=30, activity='High (>25)'),
        dict(id=4, uniquecontrast='uc8', srbnumber='SRB-0000008-001', plate='Pw08', nr_down=2, activity='Low (2-10)'),
    ])
    glues_report = pd.DataFrame([
        dict(id=7, uniquecontrast='uc2', srbnumber='SRB-0000002-001', plate='Pw02', nr_down=9, activity='Low (2-10)'),
    ])
    return {'px_fbx_measure': measure, 'px_fbx_glues': glues,
            'px_fbx_report': report, 'px_fbx_glues_report': glues_report}


class TestLoadFbxRds(unittest.TestCase):
    """fn.load_fbx_rds — the Px database as the FBX source, one entry per experiment date."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        pd.DataFrame([
            dict(uniquecontrast='uc1', molecule_batch_id='SRB-0000001-001', plate='Pw01', date='2026-06-16', source='FBX'),
            dict(uniquecontrast='uc2', molecule_batch_id='SRB-0000002-001', plate='Pw02', date='2026-06-16', source='FBX'),
            dict(uniquecontrast='uc3', molecule_batch_id='SRB-0000003-001', plate='Pw03', date='2026-06-30', source='FBX'),
            dict(uniquecontrast='uc4', molecule_batch_id='SRB-0000004-001', plate='Pw04', date='2026-06-30', source='FBX'),
            dict(uniquecontrast='uc9', molecule_batch_id='SRB-0000009-001', plate='Pw09', date='2026-04-29', source='df_raw'),
        ]).to_csv(root / 'PX_dates.csv', index=False)
        # the archive holds uc4 (the database lacks it) and a stale uc1 (the database wins)
        pd.DataFrame([
            dict(id=90, pg='P4', genes='EEE', uniquecontrast='uc4', logfc=-1.5, pvalue=1e-5, adjpval=0.01, significant=1, plate='Pw04'),
            dict(id=91, pg='P1', genes='AAA', uniquecontrast='uc1', logfc=-0.1, pvalue=0.5, adjpval=0.9, significant=0, plate='Pw01'),
        ]).to_parquet(root / 'fbx_measure.parquet', index=False)
        pd.DataFrame([
            dict(id=90, uniquecontrast='uc4', srbnumber='SRB-0000004-001', plate='Pw04', nr_down=7, activity='Low (2-10)'),
            dict(id=91, uniquecontrast='uc1', srbnumber='SRB-0000001-001', plate='Pw01', nr_down=99, activity='High (>25)'),
        ]).to_parquet(root / 'fbx_report.parquet', index=False)
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def _load(self, **kw):
        tables = _rds_tables()
        with unittest.mock.patch.object(fn, 'rds_table', lambda table, **_: tables[table].copy()):
            return fn.load_fbx_rds('fake-host', px_dates=str(self.root / 'PX_dates.csv'),
                                   archive_dir=str(self.root), verbose=False, **kw)

    def test_one_entry_per_date_and_schemas(self):
        """Each experiment date becomes one tranche, in the df_raw / MS schemas the folders give."""
        out = self._load()
        # uc1+uc2 are 2026-06-16, uc3+uc4 are 2026-06-30; uc8 has no date, uc9 is not FBX
        self.assertEqual(sorted(out), ['20260616', '20260630'])
        dfx, msx = out['20260616']
        # the loader returns exactly the target schemas, so a plain concat lines up
        self.assertEqual(list(dfx.columns), fn.FBX_DFRAW_COLS)
        self.assertEqual(list(msx.columns), fn.FBX_MS_COLS)
        # origin + date are stamped from the PX_DATES date, as a folder name would stamp them
        self.assertTrue((msx['origin'] == 'MS20260616').all())
        self.assertEqual(pd.Timestamp(msx['date'].iloc[0]), pd.Timestamp('2026-06-16'))

    def test_newest_insert_and_glues_win(self):
        """A contrast uploaded two times keeps the highest id; a glues copy beats the main copy."""
        dfx, msx = self._load()['20260616']
        # uc1 keeps the id-5 insert, not the id-1 one
        self.assertAlmostEqual(dfx.loc[dfx['uniquecontrast'] == 'uc1', 'logfc'].item(), -2.0)
        # uc2 comes from px_fbx_glues (-3.0), not from px_fbx_measure (-9.9)
        self.assertAlmostEqual(dfx.loc[dfx['uniquecontrast'] == 'uc2', 'logfc'].item(), -3.0)
        # the glues REPORT supplies its nr_down too
        self.assertEqual(msx.loc[msx['compound'] == 'SRB-0000002', 'ndown'].item(), 9)

    def test_archive_fills_only_what_the_database_lacks(self):
        """uc4 comes from the archive; the archive's stale uc1 never overrides the database."""
        dfx16, _ = self._load()['20260616']
        dfx30, _ = self._load()['20260630']
        # uc4 is present although no table serves it
        self.assertIn('uc4', set(dfx30['uniquecontrast']))
        # uc1 keeps the database row (significant), not the archive's non-significant one
        self.assertEqual(dfx16.loc[dfx16['uniquecontrast'] == 'uc1', 'significant'].item(), 1)

    def test_plate_and_compound_filters_apply(self):
        """The MS_ML rules still run on the database rows: plate substrings and control ids."""
        out = self._load(control_compounds=['SRB-0000003'], drop_plate_substr=('Pw01',))
        dfx16, _ = out['20260616']
        dfx30, _ = out['20260630']
        # Pw01 matched the plate-drop rule, so uc1 is gone
        self.assertNotIn('uc1', set(dfx16['uniquecontrast']))
        # SRB-0000003 is a control, so uc3 is gone and only the archived uc4 is left
        self.assertEqual(set(dfx30['compound']), {'SRB-0000004'})


class TestEnsureFbxPlate(unittest.TestCase):

    def test_plate_rebuilt_from_the_contrast_suffix(self):
        """A validation export ships no `plate`; the uniquecontrast suffix carries it."""
        d = pd.DataFrame({'uniquecontrast': ['SRB1_vs_DMSO_complement_Pw144VM', 'x_complement_Pw02']})
        # the text after the last '_complement_' becomes the plate
        self.assertEqual(list(fn.ensure_fbx_plate(d)['plate']), ['Pw144VM', 'Pw02'])

    def test_existing_plate_is_left_alone(self):
        """A frame that already has `plate` is returned unchanged."""
        d = pd.DataFrame({'uniquecontrast': ['a_complement_Pw99'], 'plate': ['Pw01']})
        # the file's own value wins over the name
        self.assertEqual(fn.ensure_fbx_plate(d)['plate'].item(), 'Pw01')


class TestMsFromDfRaw(unittest.TestCase):
    """fn.ms_from_df_raw — the per-compound summary recomputed from df_raw."""

    def _frames(self, pvalue=1e-8):
        """df_raw with a loud and a quiet experiment for C1, a silent C2, and an MS row for a
        compound df_raw never saw (C3)."""
        rows = [dict(compound='SRB-0000001', uniquecontrast='uc_lo', MSPlate='Pw01', pg=f'P{i}',
                     logfc=-2.0, pvalue=pvalue, significant=1) for i in range(3)]
        rows += [dict(compound='SRB-0000001', uniquecontrast='uc_hi', MSPlate='Pw02', pg=f'P{i}',
                      logfc=-1.0, pvalue=1e-4, significant=1) for i in range(8)]
        rows += [dict(compound='SRB-0000002', uniquecontrast='uc_s', MSPlate='Pw03', pg='P1',
                      logfc=0.3, pvalue=0.4, significant=0)]
        MS = pd.DataFrame({'compound': ['SRB-0000001', 'SRB-0000002', 'SRB-0000003'],
                           'ndown': [3.0, 0.0, 40.0], 'origin': 'MS20260616',
                           'activity': ['Low (2-10)', 'Silent', 'High (>25)'],
                           'date': pd.Timestamp('2026-06-16')})
        return MS, pd.DataFrame(rows)

    def test_derives_and_keeps_the_delivered_pair(self):
        """ndown/activity come from df_raw; the delivered pair survives as *_src."""
        out = fn.ms_from_df_raw(*self._frames(), verbose=False)
        r1 = out[out['compound'] == 'SRB-0000001'].iloc[0]
        # the loudest experiment wins: 8 protein groups on uc_hi, not 3 on uc_lo
        self.assertEqual(r1['ndown'], 8)
        self.assertEqual(r1['uniquecontrast'], 'uc_hi')
        self.assertEqual(r1['MSPlate'], 'Pw02')
        # the bin follows the derived ndown, so Low (2-10) stays Low (2-10) at 8
        self.assertEqual(r1['activity'], 'Low (2-10)')
        # the delivered values are untouched
        self.assertEqual(r1['ndown_src'], 3.0)
        self.assertEqual(r1['activity_src'], 'Low (2-10)')
        # ms_score = max over the kept rows of -log10(p) * -logfc = 4 * 1.0
        self.assertAlmostEqual(r1['ms_score'], 4.0)

    def test_silent_and_uncovered_compounds(self):
        """A compound df_raw covers but never hits is Silent; one it never saw keeps its value."""
        out = fn.ms_from_df_raw(*self._frames(), verbose=False).set_index('compound')
        # covered, no significant-down row -> ndown 0, Silent, no representative experiment
        self.assertEqual(out.loc['SRB-0000002', 'ndown'], 0)
        self.assertEqual(out.loc['SRB-0000002', 'activity'], 'Silent')
        self.assertTrue(pd.isna(out.loc['SRB-0000002', 'uniquecontrast']))
        # absent from df_raw -> the delivered number is kept rather than read as 0
        self.assertEqual(out.loc['SRB-0000003', 'ndown'], 40.0)
        self.assertEqual(out.loc['SRB-0000003', 'activity'], 'High (>25)')

    def test_zero_pvalue_is_floored_not_infinite(self):
        """A p-value that underflowed to 0 is floored at the smallest positive p in df_raw."""
        MS, df_raw = self._frames(pvalue=0.0)
        out = fn.ms_from_df_raw(MS, df_raw, verbose=False)
        # every scored compound stays finite (SRB-0000003 has no df_raw row, so its score is NaN)
        self.assertTrue(np.isfinite(out['ms_score'].dropna()).all())
        # ... and uc_hi still wins on count, so the score is unchanged at 4.0
        self.assertAlmostEqual(out.loc[out['compound'] == 'SRB-0000001', 'ms_score'].item(), 4.0)


# ======================================================================
# Tier A — gene-set / enrichment
# ======================================================================

class TestEnrichment(unittest.TestCase):

    def test_gene_category_long_shape(self):
        """{gene: category} -> long frame with gene/collection/term_id/term_name,
        term_id == term_name == the category."""
        out = fn.gene_category_long({'A': 'CatX', 'B': 'CatY'})
        # exact column contract for gsea_preranked / ora_enrichment
        self.assertEqual(list(out.columns), ['gene', 'collection', 'term_id', 'term_name'])
        # collection label applied; term_id mirrors the category
        self.assertTrue((out['collection'] == 'Function').all())
        self.assertEqual(out.loc[out['gene'] == 'A', 'term_name'].item(), 'CatX')

    def test_load_gmt_parses_and_uppercases(self):
        """Parse tests/data/sample.gmt -> long table; genes upper-cased, 2 sets."""
        g = fn.load_gmt({'Hallmark': str(DATA / 'sample.gmt')})
        # one collection, two gene sets
        self.assertEqual(g['collection'].unique().tolist(), ['Hallmark'])
        self.assertEqual(g['term_name'].nunique(), 2)
        # gene symbols upper-cased and a known member present
        self.assertIn('TP53', set(g['gene']))

    def test_load_gmt_missing_raises(self):
        """A glob matching no file raises FileNotFoundError (not silent empty)."""
        # missing pattern must raise, not return an empty frame
        with self.assertRaises(FileNotFoundError):
            fn.load_gmt({'X': str(DATA / 'does_not_exist_*.gmt')})

    def test_categorize_genes_keyword_and_default(self):
        """A gene whose term name hits a category keyword maps to that category;
        a gene with no categorisable term -> 'Other'; an out-of-table gene
        requested via genes= is guaranteed an 'Other' key."""
        g2t = _gene2term([('GX', 't1', 'DNA replication fork'),
                          ('GY', 't2', 'totally unrelated process')])
        out = fn.categorize_genes(g2t, genes={'GX', 'GY', 'GZ'})
        # keyword 'dna replication' -> the DNA replication category
        self.assertEqual(out['GX'], 'DNA replication')
        # no categorisable term -> default
        self.assertEqual(out['GY'], 'Other')
        # gene absent from the table still gets a key (default)
        self.assertEqual(out['GZ'], 'Other')

    def test_ora_enrichment_detects_overrepresented_term(self):
        """A term whose members are concentrated in the query set scores a low
        p; a term with < min_overlap hits is dropped."""
        g2t = _gene2term([('A', 'T1', 'T1'), ('B', 'T1', 'T1'), ('C', 'T1', 'T1'),
                          ('D', 'T1', 'T1'), ('E', 'T1', 'T1'),
                          ('F', 'T2', 'T2'), ('G', 'T2', 'T2'), ('H', 'T2', 'T2')])
        bg = list('ABCDEFGHIJ')
        out = fn.ora_enrichment({'A', 'B', 'C'}, bg, g2t, collections=('C',),
                                min_overlap=2, max_term_size=500)
        # T1 (3/5 hits) is reported; T2 (0 hits) is dropped by min_overlap
        self.assertIn('T1', set(out['term_id']))
        self.assertNotIn('T2', set(out['term_id']))
        # contract columns + correct overlap count k
        self.assertEqual(out.loc[out['term_id'] == 'T1', 'k'].item(), 3)
        for col in ['collection', 'term_id', 'k', 'K', 'n', 'N', 'p', 'fdr', 'overlap_genes']:
            self.assertIn(col, out.columns)

    def test_gsea_preranked_direction_and_sign(self):
        """A gene set concentrated at the TOP of the ranking enriches 'up'
        (NES>0); one at the bottom enriches 'down' (NES<0). Seeded null ->
        deterministic sign."""
        ranks = pd.Series({f'g{i}': float(12 - i) for i in range(12)})   # g0=12 (top) .. g11=1
        g2t = _gene2term([('g0', 'UP', 'UP'), ('g1', 'UP', 'UP'), ('g2', 'UP', 'UP'),
                          ('g9', 'DOWN', 'DOWN'), ('g10', 'DOWN', 'DOWN'), ('g11', 'DOWN', 'DOWN')])
        out = fn.gsea_preranked(ranks, g2t, collections=('C',), min_size=3,
                                max_size=100, n_perm=200, seed=0)
        up = out[out['term_id'] == 'UP'].iloc[0]
        down = out[out['term_id'] == 'DOWN'].iloc[0]
        # top-concentrated set -> up / positive NES
        self.assertEqual(up['direction'], 'up')
        self.assertGreater(up['NES'], 0)
        # bottom-concentrated set -> down / negative NES
        self.assertEqual(down['direction'], 'down')
        self.assertLess(down['NES'], 0)


# ======================================================================
# Tier A — signature / fingerprint / distance
# ======================================================================

class TestSignature(unittest.TestCase):

    def test_mean_logfc_rank_aggregates_and_uppercases(self):
        """Mean per-gene logFC over the chosen compounds only, gene symbols
        upper-cased; compounds outside the set are excluded."""
        df = pd.DataFrame({'compound': ['C1', 'C1', 'C2', 'C2', 'C3'],
                           'genes': ['g1', 'g2', 'g1', 'g2', 'g1'],
                           'logfc': [2.0, 4.0, 4.0, 6.0, 100.0]})
        r = fn.mean_logfc_rank(df, {'C1', 'C2'})
        # mean over C1+C2 per gene, index upper-cased
        self.assertAlmostEqual(r['G1'], 3.0)
        self.assertAlmostEqual(r['G2'], 5.0)
        # C3 (excluded) does not leak into the ranking
        self.assertEqual(len(r), 2)

    def test_select_strong_signature_compounds(self):
        """Keep a compound only if it is BOTH active (activity in bins) AND has a
        peak |NES| >= threshold."""
        fe = _func_enrich([('C1', 'f1', 2.5), ('C1', 'f2', 0.1),    # strong
                           ('C2', 'f1', 0.5), ('C2', 'f2', -0.3),   # weak
                           ('C3', 'f1', 3.0), ('C3', 'f2', 0.0)])   # strong but silent
        ms = pd.DataFrame({'compound': ['C1', 'C2', 'C3'],
                           'activity': ['Low (2-10)', 'High (>25)', 'Silent']})
        out = fn.select_strong_signature_compounds(fe, ms, max_abs_nes=2.0, verbose=False)
        # only C1 is active AND strong
        self.assertEqual(set(out['compound']), {'C1'})

    def test_label_signature_clusters(self):
        """Each cluster is named by its strongest |mean NES| function + arrow;
        compound-function names split on ' / ' and take the first token."""
        means = pd.DataFrame(
            {'Cell cycle / mitosis': [-3.0, 0.2], 'DNA replication': [0.5, 2.5],
             'Transport / vesicle': [1.0, -0.3]}, index=[0, 1])
        labels = fn.label_signature_clusters(means)
        # strongest pole is cell-cycle-down for C0
        self.assertEqual(labels[0], 'C0: Cell cycle ↓')
        # strongest pole is DNA-replication-up for C1
        self.assertEqual(labels[1], 'C1: DNA replication ↑')

    def test_signature_matrix_from_enrichment_pivots_and_fills(self):
        """Pivot tidy enrichment to compound × function; missing cells filled."""
        fe = _func_enrich([('C1', 'f1', 1.0), ('C1', 'f2', 2.0), ('C2', 'f1', 3.0)])
        M = fn.signature_matrix_from_enrichment(fe)
        # compound × function matrix
        self.assertEqual(M.shape, (2, 2))
        # present value preserved
        self.assertEqual(M.loc['C1', 'f2'], 2.0)
        # missing (C2,f2) filled with the neutral 0.0
        self.assertEqual(M.loc['C2', 'f2'], 0.0)

    def test_compound_distance_matrix_cosine(self):
        """Cosine distance: identical rows -> 0, orthogonal -> 1; diagonal NaN
        (exclude_self); square + symmetric."""
        feats = pd.DataFrame({'x': [1.0, 1.0, 0.0], 'y': [0.0, 0.0, 1.0]},
                             index=['A', 'B', 'C'])
        D = fn.compound_distance_matrix(feats, metric='cosine')
        # square, indexed by compound
        self.assertEqual(D.shape, (3, 3))
        # self-distance masked
        self.assertTrue(np.isnan(D.loc['A', 'A']))
        # identical vectors -> ~0 distance
        self.assertAlmostEqual(D.loc['A', 'B'], 0.0, places=6)
        # orthogonal vectors -> ~1 distance
        self.assertAlmostEqual(D.loc['A', 'C'], 1.0, places=6)


# ======================================================================
# Tier A — metrics
# ======================================================================

class TestMetrics(unittest.TestCase):

    def test_per_class_report_perfect_prediction(self):
        """Perfect predictions + perfect probabilities -> Accuracy/F1/ROC_auc=1
        per class; returns one row per class with the metric columns."""
        y = np.array(['a', 'a', 'b', 'b'])
        proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        out = fn.per_class_report(y, y, proba, classes=['a', 'b'])
        # one row per class
        self.assertEqual(len(out), 2)
        # perfect classification -> unit metrics
        self.assertAlmostEqual(out['Accuracy'].iloc[0], 1.0)
        self.assertAlmostEqual(out['F1'].iloc[0], 1.0)
        self.assertAlmostEqual(out['ROC_auc'].iloc[0], 1.0)

    def test_floor_zero_pvalues_noop_when_no_zeros(self):
        """No 0.0 p-values -> early return, nothing floored, no render."""
        meas = pd.DataFrame({'genes': ['g1', 'g2'], 'uniquecontrast': ['u1', 'u1'],
                             'pvalue': [0.01, 0.5], 'significant': [1, 0],
                             'logfc': [-2.0, 0.1], 'plate': ['P1', 'P1']})
        res = fn.floor_zero_pvalues_and_refresh_volcanoes(meas, volcano_dir='/tmp/_unused')
        # nothing to floor -> zero counts and no render stats
        self.assertEqual(res['n_floored'], 0)
        self.assertIsNone(res['stats'])


# ======================================================================
# Tier A — plate-quality CV (small synthetic + mocked harness)
# ======================================================================

class TestPlateCV(unittest.TestCase):

    @staticmethod
    def _plate_data(n=12, genes=('G',), plates=('P1', 'P2'), seed=0):
        rng = np.random.RandomState(seed)
        comps = [f'c{i}' for i in range(n)]
        rows = []
        for g in genes:
            for p in plates:
                for c in comps:
                    rows.append((c, g, p, float(rng.normal())))
        df_raw = pd.DataFrame(rows, columns=['compound', 'genes', 'MSPlate', 'logfc_corrected'])
        MF = pd.DataFrame({'compound': comps,
                           'F1': rng.rand(n), 'F2': rng.rand(n), 'F3': rng.rand(n)})
        return df_raw, MF, comps

    def test_assess_plates_globally_structure(self):
        """LOPO scan returns the matrix / per-plate scores / drop list with the
        documented keys and the evaluated plates as matrix columns."""
        df_raw, MF, _ = self._plate_data()
        res = fn.assess_plates_globally(df_raw, MF, genes=['G'], min_train=3,
                                        min_test=2, n_rf_jobs=1, verbose=False)
        # documented return contract
        self.assertEqual(set(res), {'lopo_matrix', 'plate_scores', 'recommended_drop'})
        # both plates evaluated as columns of the LOPO matrix
        self.assertEqual(set(res['lopo_matrix'].columns), {'P1', 'P2'})
        # recommendation is a (possibly empty) list
        self.assertIsInstance(res['recommended_drop'], list)

    def test_validate_plate_drop_orchestration_with_mock(self):
        """With a fixed-R² mock harness, keep_all == drop so Δ==0; output carries
        the per-gene keep/drop/delta columns."""
        df_raw, MF, _ = self._plate_data()
        out = fn.validate_plate_drop(df_raw, MF, genes=['G'], drop_plates=['P2'],
                                     n_rf_jobs=1, verbose=False, ML_Reg_module=_MockReg(0.5))
        # delta = drop - keep_all; both 0.5 -> 0
        self.assertAlmostEqual(out.loc['G', 'delta'], 0.0)
        # the pivoted result exposes both conditions + counts
        for col in ['keep_all', 'drop', 'n_keep', 'n_drop', 'delta']:
            self.assertIn(col, out.columns)

    def test_cumulative_plate_ablation_with_mock(self):
        """k = 0..len(drop_order) rows per gene; Δ measured vs each gene's k=0
        baseline (0 here since the mock R² is constant)."""
        df_raw, MF, _ = self._plate_data()
        out = fn.cumulative_plate_ablation(df_raw, MF, genes=['G'], drop_order=['P2'],
                                           n_rf_jobs=1, verbose=False, ML_Reg_module=_MockReg(0.5))
        # one row for k=0 and one for k=1
        self.assertEqual(sorted(out['k'].unique()), [0, 1])
        # constant R² -> zero marginal delta
        self.assertTrue((out['delta'].abs() < 1e-9).all())


# ======================================================================
# Tier B — smoke tests (Agg backend / tmp dir)
# ======================================================================

class TestSmoke(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_function_enrichment_all_pipeline(self):
        """End-to-end per-compound ORA+GSEA over coarse categories on tiny
        synthetic data -> tidy compound × function frame with NES columns."""
        rng = np.random.RandomState(0)
        cats = {f'g{i}': ('CatA' if i < 6 else 'CatB') for i in range(12)}
        rows = []
        for comp in ['C1', 'C2']:
            for g, _cat in cats.items():
                rows.append((comp, g, float(rng.normal()), float(rng.uniform(1e-4, 1)),
                             int(rng.rand() < 0.3)))
        df_raw = pd.DataFrame(rows, columns=['compound', 'genes', 'logfc', 'pvalue', 'significant'])
        out = fn.function_enrichment_all(df_raw, cats, n_perm=50, n_jobs=1,
                                         min_overlap=1, verbose=False)
        # tidy output carries the GSEA column and both compounds
        self.assertIn('gsea_NES', out.columns)
        self.assertEqual(set(out['compound']), {'C1', 'C2'})

    def test_plot_function_enrichment_smoke(self):
        """Lollipop of per-function enrichment draws without error."""
        df = pd.DataFrame({'function': ['DNA replication', 'Cell cycle / mitosis', 'Signaling'],
                           'gsea_NES': [2.4, -1.8, 0.3], 'gsea_fdr': [0.01, 0.03, 0.6]})
        fn.plot_function_enrichment(df)
        # a figure with axes was produced
        self.assertTrue(plt.gcf().get_axes())

    def test_plot_activity_composition_bars_smoke(self):
        """Stacked activity-composition bars draw across two tranches."""
        fn.plot_activity_composition_bars(_ms_two_tranches(), annotate=False)
        # bars rendered onto an axes
        self.assertTrue(plt.gcf().get_axes())

    def test_plot_activity_composition_bars_by_group_normalized(self):
        """group_col + normalize: one 100 % bar per group, in the order asked, counts in the summary."""
        ms = _ms_two_tranches().assign(flavor=['Music', 'Music', 'Lipo', 'Lipo', 'Sonic', 'Library'])
        ax, summary = fn.plot_activity_composition_bars(
            ms, group_col='flavor', order=['Sonic', 'Music', 'Lipo', 'Library', 'Absent'],
            normalize=True, annotate=False)
        # a group named in `order` but absent from the frame is skipped, the rest keep that order
        self.assertEqual(list(summary.index), ['Sonic', 'Music', 'Lipo', 'Library'])
        # the summary holds counts, not shares — 6 compounds over the 4 bars
        self.assertEqual(summary['n'].tolist(), [1, 2, 2, 1])
        # every bar is scaled to 100 %, so the y-axis stops just above it
        self.assertAlmostEqual(ax.get_ylim()[1], 112)
        # the x-axis carries the group values, not dates
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()],
                         ['Sonic', 'Music', 'Lipo', 'Library'])

    def test_plot_activity_composition_bars_without_silent(self):
        """show_silent=False: the silent band leaves the stack, so each bar stops at its activity rate."""
        ms = _ms_two_tranches().assign(flavor=['Music'] * 4 + ['Lipo'] * 2)
        ax, summary = fn.plot_activity_composition_bars(
            ms, group_col='flavor', order=['Music', 'Lipo'], normalize=True,
            show_silent=False, annotate=False)
        # the silent band is not drawn, so it is absent from the legend
        self.assertNotIn('Silent', [t.get_text() for t in ax.get_legend().get_texts()])
        # Music holds 2 silent of 4, so its bar stops at 50 %; Lipo holds none and reaches 100 %
        tops = [max(b.get_y() + b.get_height() for b in ax.patches
                    if round(b.get_x() + b.get_width() / 2) == j) for j in (0, 1)]
        self.assertEqual([round(t) for t in tops], [50, 100])
        # the axis top follows the tallest bar drawn
        self.assertAlmostEqual(ax.get_ylim()[1], 112.0)
        # the summary still counts every compound, silent included
        self.assertEqual(summary['n'].tolist(), [4, 2])
        self.assertEqual(summary['Silent'].tolist(), [2, 0])

    def test_plot_activity_rate_by_tranche_smoke(self):
        """Per-tranche activity-rate bar plot draws without error."""
        fn.plot_activity_rate_by_tranche(_ms_two_tranches(), annotate=False)
        # rate bars rendered onto an axes
        self.assertTrue(plt.gcf().get_axes())

    def test_plot_activity_composition_over_time_smoke(self):
        """100%-stacked activity composition area draws without error."""
        fn.plot_activity_composition_over_time(_ms_two_tranches())
        # stacked area rendered onto an axes
        self.assertTrue(plt.gcf().get_axes())

    def test_plot_activity_area_absolute_smoke(self):
        """Absolute (count) stacked activity area draws without error."""
        fn.plot_activity_area_absolute(_ms_two_tranches())
        # area rendered onto an axes
        self.assertTrue(plt.gcf().get_axes())

    def test_plot_volcano_smoke(self):
        """Single-compound volcano draws with one gene highlighted."""
        df = pd.DataFrame({'compound': ['C1'] * 4,
                           'genes': ['TP53', 'MDM2', 'CDK1', 'MYC'],
                           'logfc': [-3.0, 2.0, -0.2, 0.1],
                           'pvalue': [1e-5, 1e-4, 0.5, 0.8]})
        fn.plot_volcano(df, 'C1', 'TP53')
        # volcano rendered onto an axes
        self.assertTrue(plt.gcf().get_axes())

    def test_plot_autoresearch_progress_smoke(self):
        """Progress plot reads the sample JSONL log and returns (fig, ax)."""
        fig, ax = fn.plot_autoresearch_progress(DATA / 'sample_autoresearch.jsonl',
                                                metric_name='pr_auc')
        # returns a populated matplotlib figure/axes pair
        self.assertTrue(ax.get_lines() or ax.collections)

    def test_recompute_volcanoes_writes_image(self):
        """recompute_volcanoes renders the requested (gene, key) volcano to disk
        (exercises plot_volcano_significant + the cache writer)."""
        meas = pd.DataFrame({'genes': ['TP53', 'MDM2', 'CDK1', 'MYC'],
                             'uniquecontrast': ['u1'] * 4,
                             'logfc': [-3.0, -2.5, 0.1, 0.2],
                             'pvalue': [1e-6, 1e-5, 0.4, 0.7],
                             'significant': [1, 1, 0, 0],
                             'plate': ['P1'] * 4})
        with tempfile.TemporaryDirectory() as d:
            stats = fn.recompute_volcanoes(meas, [('TP53', 'u1')], d,
                                           significant=True, size_px=200, n_jobs=1)
            # one volcano image (SVG) produced in the cache dir
            self.assertTrue(list(Path(d).glob('*.svg')) or list(Path(d).glob('*.png')))


# ======================================================================
# Tier C — not unit-tested (explicit, so the gaps are visible)
# ======================================================================

class TestNotUnitTested(unittest.TestCase):

    @unittest.skip("network: live OpenTargets GraphQL API")
    def test_get_opentarget_disease_score(self):
        pass

    @unittest.skip("integration: interactive 3D HTML builder (network + RDKit + many files)")
    def test_plot_target_3d(self):
        pass

    @unittest.skip("integration: interactive 3D HTML builder (network + RDKit + many files)")
    def test_plot_3d_interface(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
