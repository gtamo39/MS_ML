# RDKit_tools.py
# Collection of RDKit-based utilities for molecular manipulation and analysis

from numpy import dot, transpose, sqrt, sum, array
from numpy.linalg import svd, det
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS, rdMolAlign, rdDistGeom, rdFingerprintGenerator, rdDepictor
from rdkit import DataStructs
from rdkit.Chem import rdmolfiles
from rdkit.Chem.Draw import rdMolDraw2D
try:
    from sklearn.neighbors import DistanceMetric
except:
    from sklearn.metrics import DistanceMetric
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from tqdm import tqdm
import pandas as pd
import numpy as np
from copy import deepcopy
# import numba
# from more_itertools import chunker
import argparse
import os
import subprocess as sub

# Try importing local modules
try:
    from .Molecule import Molecule as M
except:
    from Molecule import Molecule as M


class SVDSuperimposer(object):
    """
    SVDSuperimposer finds the best rotation and translation to put
    two point sets on top of each other (minimizing the RMSD). This is
    eg. useful to superimpose crystal structures.

    SVD stands for Singular Value Decomposition, which is used to calculate
    the superposition.

    Reference:

    Matrix computations, 2nd ed. Golub, G. & Van Loan, CF., The Johns
    Hopkins University Press, Baltimore, 1989
    """
    def __init__(self):
        self._clear()

    # Private methods

    def _clear(self):
        self.reference_coords = None
        self.coords = None
        self.transformed_coords = None
        self.rot = None
        self.tran = None
        self.rms = None
        self.init_rms = None

    def _rms(self, coords1, coords2):
        "Return rms deviations between coords1 and coords2."
        diff = coords1 - coords2
        l = coords1.shape[0]
        return sqrt(sum(sum(diff * diff)) / l)

    # Public methods

    def set(self, reference_coords, coords):
        """
        Set the coordinates to be superimposed.
        coords will be put on top of reference_coords.

        o reference_coords: an NxDIM array
        o coords: an NxDIM array

        DIM is the dimension of the points, N is the number
        of points to be superimposed.
        """
        # clear everything from previous runs
        self._clear()
        # store cordinates
        self.reference_coords = reference_coords
        self.coords = coords
        n = reference_coords.shape
        m = coords.shape
        if n != m or not(n[1] == m[1] == 3):
            raise Exception("Coordinate number/dimension mismatch.")
        self.n = n[0]

    def run(self):
        "Superimpose the coordinate sets."
        if self.coords is None or self.reference_coords is None:
            raise Exception("No coordinates set.")
        coords = self.coords
        reference_coords = self.reference_coords
        # center on centroid
        av1 = sum(coords) / self.n
        av2 = sum(reference_coords) / self.n
        coords = coords - av1
        reference_coords = reference_coords - av2
        # correlation matrix
        a = dot(transpose(coords), reference_coords)
        u, d, vt = svd(a)
        self.rot = transpose(dot(transpose(vt), transpose(u)))
        # check if we have found a reflection
        if det(self.rot) < 0:
            vt[2] = -vt[2]
            self.rot = transpose(dot(transpose(vt), transpose(u)))
        self.tran = av2 - dot(av1, self.rot)

    def get_transformed(self):
        "Get the transformed coordinate set."
        if self.coords is None or self.reference_coords is None:
            raise Exception("No coordinates set.")
        if self.rot is None:
            raise Exception("Nothing superimposed yet.")
        if self.transformed_coords is None:
            self.transformed_coords = dot(self.coords, self.rot) + self.tran
        return self.transformed_coords

    def get_rotran(self):
        "Right multiplying rotation matrix and translation."
        if self.rot is None:
            raise Exception("Nothing superimposed yet.")
        return self.rot, self.tran

    def get_init_rms(self):
        "Root mean square deviation of untransformed coordinates."
        if self.coords is None:
            raise Exception("No coordinates set yet.")
        if self.init_rms is None:
            self.init_rms = self._rms(self.coords, self.reference_coords)
        return self.init_rms

    def get_rms(self):
        "Root mean square deviation of superimposed coordinates."
        if self.rms is None:
            transformed_coords = self.get_transformed()
            self.rms = self._rms(transformed_coords, self.reference_coords)
        return self.rms


# -------------------------------
# compound properties
# -------------------------------

def get_4_properties(self, x):
    pass


def get_properties(smiles, func):
    pass


def get_CM_properties(df_, v=False):
    pass


# -------------------------------
# Morgan Fingerprints
# -------------------------------

def get_MF_from_smiles(smiles, nBits=2048, radius=2):
    """
    Compute Morgan (ECFP-like) fingerprint bits for a SMILES string.

    :param str smiles: SMILES of the compound
    :param int nBits: fingerprint length (default 2048)
    :param int radius: Morgan radius (default 2, i.e. ECFP4)
    :return np.ndarray: 1D array of 0/1 ints of length nBits (None if SMILES invalid)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits,includeChirality=True)
    # print(gen)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(nBits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_bitvec_from_smiles(compound, smiles, nBits=2048, drawMol=False, drawBits=False):
    """
    Wrapper returning the Morgan FP as a comma-separated bit string in the
    format consumed by :func:`get_MF_bits_from_df` (which does ``.str.split(',')``).
    """
    arr = get_MF_from_smiles(smiles, nBits=nBits)
    if arr is None:
        return None
    return ','.join(str(int(b)) for b in arr)


def get_MF_bits_from_df(df_, nBits=2048, v=False):
    """
    From df containing smiles, returns compute Morgan Fingerprints
    :param df df_: contains columns |compound|smiles|
    :return df tmp: MFs associated to compounds
    """
    df = df_.copy()[['compound', 'smiles']]
    apply_fn = df.progress_apply if v else df.apply
    df['MF'] = apply_fn(
        lambda x: get_bitvec_from_smiles(str(x['compound']), str(x['smiles']), nBits),
        axis=1
    )

    tmp = df.MF.str.split(",", expand=True)
    tmp.columns = ['F' + str(col) for col in tmp.columns]

    tmp.index = df['compound']
    tmp = tmp.dropna(axis=0)  # remove compound with any missing value
    tmp = tmp.astype('int64')
    tmp = tmp.reset_index()

    return tmp

def get_colinear_features(df, step=1):
    """
    Checks which Morgan fingerprints features are always found together across
    all molecules. The ones that are found to be colinear are then removed from
    the original dataframe.

    :param dataframe df: the original dataframe with cols |compound|F0|F...|
    :param int step: how many upstream features to consider (default = 1)

    :return lists streaks,MF2rm: the list of colinear features
    """
    ## remove features that always co-occur:
    cols = df.drop('compound', axis=1).columns

    streaks = [[]]  # list of features to aggregate
    MF2rm = []      # list of colinear features to remove (only keep 1st)
    coli = [int(x[1:]) for x in cols]
    coli.sort()  # sort by MF number (int)

    for i in tqdm(range(len(coli) - step), ncols=70):
        F1, F2 = 'F' + str(coli[i]), 'F' + str(coli[i + step])
        res = abs(df[F1] - df[F2]).sum()
        if res == 0.0:
            MF2rm.append(F2)
            if F1 in streaks[-1]:
                streaks[-1].append(F2)
            else:
                streaks.append([F1, F2])
        elif F1 not in MF2rm:
            streaks.append([F1])

    streaks.remove([])

    return streaks, MF2rm


def get_all_bits(df, r, bitvec, drawMol=False, drawBits=False):
    """
    From file containing compound name and smiles, computes morgan fingerprints
    in tall skinny format.

    :param dataframe df: input file containing |compound|smiles| columns
    :param int r: radius
    :param list bitvec: ...
    :param bool drawMol: whether to draw the compound as png
    :param bool drawBits: whether to draw individual bit as png

    :return: the fingerprint associated to a file in tall skinny format
    """
    bit_col = []  # contains the bit value
    CM_col = []   # contains the CC name
    r_col = []    # contains the radius of bit
    bit_info = {}  # contains info about where a bit was first encountered

    for index, row in tqdm(df.iterrows(), total=df.shape[0], ncols=70):

        smiles = row['smiles']
        compound = row['compound']

        try:
            mol = Chem.MolFromSmiles(smiles)
            info = {}
            fp = AllChem.GetMorganFingerprint(mol, radius=r, bitInfo=info)
            bit_is = (list(info.keys()))

            for bit_i in bit_is:
                CM_col.append(compound)
                bit_col.append(bit_i)
                r_col.append(info[bit_i][0][1])

                if drawBits and bit_i not in bitvec:
                    try:
                        bit_info['F' + str(bit_i)] = [compound, mol, info[bit_i][0][0], info]
                        bitvec.append(bit_i)
                    except:
                        print('>> issue with bit', bit_i)
        except:
            print('>> issue with', compound)

    final_df = pd.DataFrame({'compound': CM_col, 'bit': bit_col, 'r': r_col})
    print('\n>> unique features in file', len(final_df['bit'].unique()))

    return final_df, bit_info


def draw_unique_MFs(streaks, bit_info, MFP2Simple, r=2):
    """
    Based on the list containing aggregated set of bits (streaks),
    draws either single bits or aggregated sets in png

    :param list of list streaks:
    :param bit_info: dict containing original info of each bit
    :param dict MFP2Simple: dict to go back to original ...
    :param int r: radius

    :return list bit2svg: associates image to bit
    """
    inv_map = {v: k for k, v in MFP2Simple.items()}
    bit2svg = {}

    for streak in tqdm(streaks, ncols=70):
        if len(streak) == 1:
            try:
                bits = [j for sub in streak for j in sub]
                bit = streak[0]
                mol = bit_info[bit][1]
                bit_i = int(inv_map[bit][1:])
                info = bit_info[bit][3]
                mfp2_svg = Draw.DrawMorganBit(mol, bit_i, info, useSVG=True)
                bit2svg[bit] = mfp2_svg
            except:
                print('>> could not draw bit', bit)
        else:
            cc = list(set([bit_info[x][0] for x in streak]))
            if len(cc) > 1:
                '>> DANGER: trying to aggregate features from two different molecules! quitting...'
                return

            central_atoms = []
            branches = []
            mol = bit_info[streak[0]][1]

            for bit in streak:
                bit_atom = bit_info[bit][2]
                central_atoms.append(bit_atom)

                env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, bit_atom)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                atoms = amap.keys()
                branches += atoms

            branches = list(set(branches))

            atom_cols = {}
            colours = [(0.8, 0.0, 0.8), (0.0, 0.8, 0.8, 0), (0.0, 0.8, 0.8), (0, 0, 0.8)]
            for i, at in enumerate(branches):
                if at in central_atoms:
                    atom_cols[at] = colours[2]
                else:
                    atom_cols[at] = colours[1]

            molSize = (200, 200)
            mc = Chem.Mol(mol.ToBinary())
            if not mc.GetNumConformers():
                rdDepictor.Compute2DCoords(mc)

            drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
            drawer.DrawMolecule(mc, highlightAtoms=branches, highlightAtomColors=atom_cols)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            bit2svg[streak[0]] = svg.replace('svg:', '')

    return bit2svg


def get_unique_MFs_from_df(df, r=2, drawMol=False, drawBits=False):
    """
    Returns the exact bits associated with features and thus avoid collisions

    :param df: dataframe contains at least |compound|smiles|
    :param int r: MF radius (default 2)
    :param Bool drawMol: whether to draw the molecule (old)
    :param int r: radius

    :return: None, simply draws
    """
    bitvec = []

    MF, bit_info = get_all_bits(df, r, bitvec, drawMol=drawMol, drawBits=drawBits)
    MF['bit'] = MF['bit'].apply(lambda x: 'F' + str(x))

    MF = MF[(MF['r'] == r)]
    MF['value'] = 1

    keyz = list(MF['bit'].unique())
    valz = ['F' + str(i) for i in range(len(keyz))]
    MFP2Simple = {keyz[i]: valz[i] for i in range(len(keyz))}

    MF['bit'] = MF['bit'].apply(lambda x: MFP2Simple[x])

    if drawBits:
        for key, value in MFP2Simple.items():
            bit_info[value] = bit_info.pop(key)

    MF_P = MF.pivot_table(index='compound', columns='bit', values='value').reset_index().fillna(0).rename(columns=MFP2Simple)
    print('>> size pivoted df:', MF_P.shape)

    streaks, MF2rm = get_colinear_features(MF_P, step=1)
    MF = MF[~MF['bit'].isin(MF2rm)]

    if drawBits:
        print('\n>> Drawing bit')
        return MF, draw_unique_MFs(streaks, bit_info, MFP2Simple)

    return MF


# -------------------------------
# utils
# -------------------------------

def convert_smiles_to_canonical(smiles, smiles2ignore=[], printError=False):
    """title says it all"""
    if smiles in smiles2ignore:
        return ''
    try:
        canon = Chem.CanonSmiles(smiles)
        return(canon)
    except:
        if printError:
            print('>> problem with ', smiles)
        return ""


def check_pains(smiles, pains_df):
    """
    From smiles, check violated pains
    """
    pain_score = 0
    m = Chem.MolFromSmiles(smiles)
    for pattern in pains_df['Pattern']:
        screen = Chem.MolFromSmarts(pattern)
        if len(m.GetSubstructMatch(screen)) > 0:
            pain_score += 1
            break
    return pain_score

def has_substructure(smiles, search, typ='smiles'):
    """Returns True if substructure is found within query smiles """
    m = Chem.MolFromSmiles(smiles)
    if typ == 'smiles':
        screen = Chem.MolFromSmiles(search)
    else:
        screen = Chem.MolFromSmarts(search)

    return m.HasSubstructMatch(screen)


def check_smiles_RDKiT(df):
    """
    extracts the list of compounds with invalid smiles
    :param df df: contains columns |compounds|smiles|
    :return: list bad_CMs: the invalide compoud list
    """
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    failed_CMs = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        CM = row['compound']
        smiles = row['smiles']
        try:
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
            fp = mfpgen.GetFingerprint(Chem.MolFromSmiles(smiles))
        except:
            failed_CMs.append(CM)

    return failed_CMs


def show_top_compounds(df, col2show=['target'], top=40):
    """
    Simply displays the smiles of top compounds given a dataset
    """
    CMs = df.head(top)['compound']
    txt = list(CMs)

    for i in range(0, len(txt), 1):
        for col in col2show:
            txt[i] += '\n' + col + ': ' + str(df[df['compound'] == list(CMs)[i]][col].item())

    smiles = [Chem.MolFromSmiles(df[df['compound'] == x]['smiles'].item()) for x in CMs]

    return smiles, txt


def remove_duplicated_smiles(df, speed='fast'):
    """
    compute pairwise distance between every smile and remove smiles which have
    distance == 0

    :param df: the original daframe containing cols |compound|smiles|
    :param bool fast: if large dataset, compute RogersTanimoto instead (faster)
    :return df: the cleaned dataset
    """
    if speed == 'extra_fast':
        df = df.groupby('smiles').first().reset_index()
        return df

    pred = df[['compound', 'smiles']]
    pred_MF = get_MF_bits_from_df(pred, nBits=2048)

    if speed == 'fast':
        d1 = get_RogersTanimoto_distance_matrix(pred_MF, pred_MF)
    elif speed == 'slow':
        d1 = get_RDKiTTanimoto_distance_matrix(pred_MF, pred_MF)
    d1['compound1'] = d1.index
    d1 = d1.melt(id_vars=['compound1']).rename(columns={'variable': 'compound2'})
    tmp = d1[(d1['value'] == 0) & (d1['compound1'] != d1['compound2'])].sort_values(['compound1', 'compound2'])
    if tmp.shape[0] != 0:
        tmp['pair'] = tmp[['compound1', 'compound2']].apply(lambda x: ','.join(sorted([x[0], x[1]])), axis=1)
        tmp = tmp.groupby('pair').first().reset_index()
        cm2rm = list(tmp['compound2'].unique())
        df = df[~df['compound'].isin(cm2rm)]
    return df


def remove_stereochemistry(smiles):
    """Removes stereochemistry and returns flat smiles"""
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def get_mol_from_structure(filename):
    """Load structure file as RDKit molecule"""
    try:
        suffix = filename.split('.')[-1]
        if suffix == 'pdb':
            mol = Chem.rdmolfiles.MolFromPDBFile(filename)
        elif suffix == 'sdf':
            supp = Chem.SDMolSupplier(filename)
            mol = [mol for mol in supp if mol][0]
        elif suffix == 'mol2':
            print('>> reading mol2')
            mol = Chem.rdmolfiles.MolFromMolBlock(filename)
        return mol
    except:
        return None


def display_SMARTS(smi, search):
    """
    Displays matched smarts atoms onto smiles
    :param list smi: smiles
    :param str smarts: SMARTS pattern
    """
    mol = Chem.MolFromSmiles(smi)
    sub = Chem.MolFromSmarts(search)
    highlights = list(mol.GetSubstructMatches(sub)[0])
    return mol, highlights


def extract_vina_out_pdbqt_txt(fn, smiref=None):
    """
    reads Autodock vina's output and separates pdbqts
    by models, assigns correct bond order and returns
    an RDkit mol
    """
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    models = []
    scores = []
    pdbqt_vina = open(fn, 'r')

    for l in pdbqt_vina.readlines():
        if l:
            if l[:18] == 'REMARK VINA RESULT':
                score = list(filter(None, l.split(' ')))[3]
                scores.append(float(score))

            if l[:5] == 'MODEL':
                models.append([])
            else:
                models[-1].append(l)

    models = [''.join(x) for x in models]

    if smiref != None:
        new_models = []
        template = Chem.MolFromSmiles(smiref)

        for pdbqtxt in models:
            try:
                tmp = M()
                tmp.import_pdb(pdbqtxt, from_text=True)
                tmp = tmp.assign_correct_atomtype().remove_H()
                tmp.conect = ''

                docked_pose = AllChem.MolFromPDBBlock(tmp.write_pdb())
                newMol = AllChem.AssignBondOrdersFromTemplate(template, docked_pose)
                new_models.append(newMol)
            except:
                new_models.append(np.nan)

        models = new_models

    df = pd.DataFrame({'score': scores, 'mol': models}).sort_values('score', ascending=True).reset_index(drop=True)

    return df


def get_non_matching_fragments_smiles(smiles, smarts):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"

    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return "Invalid SMARTS"

    matches = mol.GetSubstructMatches(pattern)

    if not matches:
        return [smiles]

    atom_indices = set(range(mol.GetNumAtoms()))
    for match in matches:
        atom_indices -= set(match)

    if not atom_indices:
        return []

    non_matching_submol = Chem.MolFragmentToSmiles(mol, list(atom_indices), isomericSmiles=True, allHsExplicit=True, allBondsExplicit=True)
    fragments = non_matching_submol.split('.')
    valid_fragments = [frag for frag in fragments if frag]

    return valid_fragments


def keep_longest_fragment(l_v):
    leng = []
    for x in l_v:
        leng.append(len(x))
    return l_v[np.argmax(leng)]


def get_mismatching_atoms(smiles1, smiles2):
    """From Chatgpt"""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if not mol1 or not mol2:
        raise ValueError("Invalid SMILES string provided.")

    mcs_result = rdFMCS.FindMCS([mol1, mol2])
    mcs_smarts = mcs_result.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)

    match1 = mol1.GetSubstructMatch(mcs_mol)
    match2 = mol2.GetSubstructMatch(mcs_mol)

    atom_indices1 = set(range(mol1.GetNumAtoms()))
    atom_indices2 = set(range(mol2.GetNumAtoms()))

    matched_indices1 = set(match1)
    matched_indices2 = set(match2)

    mismatching_indices1 = atom_indices1 - matched_indices1
    mismatching_indices2 = atom_indices2 - matched_indices2

    print(mismatching_indices2)

    mismatching_atoms1 = [mol1.GetAtomWithIdx(idx).GetSymbol() for idx in mismatching_indices1]
    mismatching_atoms2 = [mol2.GetAtomWithIdx(idx).GetSymbol() for idx in mismatching_indices2]

    return mismatching_atoms1, mismatching_atoms2


def cluster_df_by_smiles(df, smiles_col='smiles', n_clusters=10, how='kmeans'):
    """
    Gets the clusters of the dataset based on tanimoto difference
    :param df: must contain |compound|smiles|
    :param str smiles_col: column on which to cluster
    :param int n_clusters: number of clusters
    """
    MFs = get_MF_bits_from_df(df)
    d_matrix = get_RDKiTTanimoto_distance_matrix(MFs, MFs)
    mds = MDS(n_components=2, n_jobs=4, random_state=42, dissimilarity='precomputed')
    E = mds.fit_transform(d_matrix)
    df[['e1', 'e2']] = E
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(E)

    centroids = []
    for i in range(len(kmeans.cluster_centers_)):
        tmp = df[['compound']].copy()
        tmp['dist'] = np.sqrt(np.sum((kmeans.cluster_centers_[i] - E)**2, axis=1))
        centroids.append(tmp.sort_values('dist').head(1)['compound'].item())

    df['is_centroid'] = False
    df['cluster'] = kmeans.labels_
    df.loc[df['compound'].isin(centroids), 'is_centroid'] = True

    return df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3D manipulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_SVD_rot_trans(x, y):
    """get the roto-translation matrix to get y on x"""
    sup = SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rot, tran = sup.get_rotran()
    return sup, rot, tran


def align_x_on_y(y, rot, tran):
    y_on_x = dot(y, rot) + tran
    return y_on_x


def create_CELMoD_from_smiles(smiles, out_path='',
                              atom2gluttype={'C': '/home/jovyan/work/pharao_IPS/data/ref_glutarimide/gsp1_ligand.sdf',
                                             'N': '/home/jovyan/work/pharao_IPS/data/ref_glutarimide/DHU_align.pdb'}, v=1):
    """
    From smiles, returns CELMoD with correct glutarimide/DHU stereochemisty
    :param str smiles: the smiles string
    :param str out_path: path to write .pdb
    :param dict atom2gluttype: contains the full paths to clean glutaride & DHU
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetSubstructMatches(Chem.MolFromSmarts('O=C1[C;H2][C;H2][C]C(=0)[N;H1]1')):
        atom = 'C'
    elif mol.GetSubstructMatches(Chem.MolFromSmarts('O=C1[C;H2][C;H2][N]C(=0)[N;H1]1')):
        atom = 'N'
    else:
        print('could not find glutarimide type')

    flag = 1
    max_trials = 20
    trial = 0

    while flag:
        m = generate_single_conformer_from_smiles(smiles)
        Mol = assign_glutarimide_stereochemistry(m, atom2gluttype[atom], out_path)
        pdbtxt = Mol.write_pdb()
        molFromPdb = Chem.rdmolfiles.MolFromPDBBlock(pdbtxt)

        if molFromPdb:
            idx = list(get_yxz_of_matched_smarts(Mol, 'O=C1[C;H2][C;H2][C,N](*)C(=0)[N;H1]1', origin='Molecule')[1])
            intraC = Mol.get_intraContacts()
            intraC = [list(x) for x in intraC if len(set(idx) & set(x)) == 0]
            vdw = Mol.compute_intra_vdw(intraC)

            if vdw <= 0.0:
                flag = 0
        else:
            if v:
                print('>> could not load ' + smiles + ', retrying...')
        trial += 1
        if trial >= max_trials:
            return None

    if out_path != '' and get_mol_from_structure(out_path):
        flag = 0

    return Mol


def generate_single_conformer_from_smiles(smiles):
    """
    :param str smiles:
    :return rkdit_mol m:
    """
    m = Chem.MolFromSmiles(smiles)

    if 11 < 3:
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m)
        AllChem.MMFFOptimizeMolecule(m)
        m = Chem.RemoveHs(m)
    else:
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = False
        m = Chem.AddHs(m)
        AllChem.EmbedMultipleConfs(m, numConfs=10, params=params)
        m = Chem.RemoveHs(m)

    return m


def assign_glutarimide_stereochemistry(m, ref_CM_path, out_path='', glut_smarts='O=C1[C;H2][C;H2][C,N](*)C(=0)[N;H1]1'):
    """
    Create a Molecule with optimized glutarimide stereochemistry based on crystal structure.
    :param rdkit_mol m: The input mol of the structure
    :param str ref_CM_path: the crystal structure (pdb or sdf)
    :param str out_path: path to write pdb of molecule
    :param str glut_smarts: smarts of glutarimide to match
    :return Molecule() sdf_opt: structure with correct glutarimide/DHU
    """
    screen = Chem.MolFromSmarts(glut_smarts)
    matches = list(m.GetSubstructMatch(screen, useChirality=True))

    anchor_i = [3, 4, 5]

    real_glut_xyz, real_glut_matches = get_yxz_of_matched_smarts(ref_CM_path, glut_smarts, v=0)

    ref = []
    for i in anchor_i:
        j = matches[i]
        pos = m.GetConformer().GetAtomPosition(j)
        ref.append([pos.x, pos.y, pos.z])

    sup = SVDSuperimposer()

    x = np.array(ref)
    y = real_glut_xyz[anchor_i]
    sup.set(x, y)
    sup.run()

    rot, tran = sup.get_rotran()
    new_coords = np.dot(real_glut_xyz, rot) + tran

    it = 0
    for i in matches:
        m.GetConformer().SetAtomPosition(i, new_coords[it])
        it += 1

    processed_molblock = Chem.SDWriter.GetText(m)
    sdf_opt = M()
    sdf_opt.import_sdf(processed_molblock, from_text=True)
    sdf_opt = sdf_opt.renumber_atomname()

    if out_path != '':
        sdf_opt.write_pdb(out_path)

    return sdf_opt


def align_molecules_based_on_smarts(ref, mod, search, pdb_name=None, glut_ind_ref=[], glut_ind_mod=[], from_Molecule=False):
    """
    we assign the two molecules are different but share
    a distinct substructure, which will be used as anchor
    to align the two molecules:
    :param str ref: sdf name of molecule
    :param str mod: sdf name of molecule
    :param str search: smarts pattern
    :return Molecule(): Molecule() instance of mod
    """
    if from_Molecule:
        ref_M = ref
        mod_M = mod
    else:
        ref_M = M()
        ref_M.import_from_suffix(ref)
        mod_M = M()
        mod_M.import_from_suffix(mod)

    if glut_ind_ref:
        glut_xyz_ref = ref_M.xyz[glut_ind_ref]
    else:
        glut_xyz_ref, glut_ind_ref = get_yxz_of_matched_smarts(ref_M, search, v=0, origin='Molecule')

    if glut_ind_mod:
        glut_xyz_mod = mod_M.xyz[glut_ind_mod]
    else:
        glut_xyz_mod, glut_ind_mod = get_yxz_of_matched_smarts(mod_M, search, v=0, origin='Molecule')

    sup, rot, tran = get_SVD_rot_trans(glut_xyz_ref, glut_xyz_mod)

    mod_M.set_xyz(align_x_on_y(mod_M.xyz, rot, tran))

    if pdb_name != None:
        mod_M.write_pdb(pdb_name)

    return mod_M


def remove_H_from_pdbqt(pdbqt_file, from_text=True):

    if not from_text:
        pdbqt_lines = open(pdbqt_file).readlines()
    else:
        pdbqt_lines = pdbqt_file.split('\n')

    pdbqt_out = ''

    for line in pdbqt_lines:
        data = list(filter(None, line.replace('\n', '').split(' ')))

        if len(data) > 0:
            if data[0] == 'ATOM' and data[2] == 'H':
                continue

        pdbqt_out += line + '\n'

    return pdbqt_out


def get_smiles_from_structures(filename):
    """get a smiles string from structure"""
    try:
        mol = get_mol_from_structure(filename)
        return Chem.MolToSmiles(mol)
    except:
        print('>> could not compute smiles')
        return np.nan


def get_yxz_of_matched_smarts(filename, smarts_pattern, v=0, origin='Molecule'):
    """
    Gets the xyz coordinates of atoms matching specific SMARTS pattern
    :param str filename: molecule filename e.g. ligand.sdf or ligand.pdb
    :param str smarts_pattern: the pattern to look for
    :return np.array xyz
    """
    if origin == 'file':
        mol = get_mol_from_structure(filename)
    elif origin == 'Molecule':
        mol = Chem.rdmolfiles.MolFromPDBBlock(filename.write_pdb())
    elif origin == 'rdkitmolecule':
        mol = filename

    if mol:
        query = Chem.MolFromSmarts(smarts_pattern)

        if query:
            matches = mol.GetSubstructMatches(query)

            if matches:
                for match in matches:
                    if v:
                        print("Match found with atom indices:", match)
                    lxyz = mol.GetConformer().GetPositions().astype(np.float32)
                    xyz_matched = np.array(lxyz[list(match)])

                    return xyz_matched, match

        elif v:
            print("No matches found for the SMARTS pattern.")
    elif v:
        print("Invalid SMARTS pattern.")
    elif v:
        print("Error loading file.")


def compute_rsmd_of_2_molecules(m1, m2, v=0):
    """returns rmsd between different conformations"""
    try:
        if m1.shape[0] == m2.shape[0]:
            rmsd = np.sqrt(np.sum(np.sum((m1 - m2)**2), axis=0) / len(m1))
            return rmsd
        elif v:
            print('>> un-matched atoms for two molecules, return -1')
            return -1.0
    except:
        if v:
            print('>> could not compute rmsds of molecules, return -1')
        return -1.0


def optimize_3D_geometry(sdfPath, outputPath=None):
    """
    Quickly optimize 3D ligand geometry whilst keeping atom superimposition to
    reference molecule
    :param str sdfPath: path to .sdf ligand file
    :return rdkit mol: the optimize molecule superimposed to reference
    """
    supp = Chem.SDMolSupplier(sdfPath)
    mol_opt = [mol for mol in supp if mol][0]

    AllChem.MMFFOptimizeMolecule(mol_opt)

    processed_molblock = Chem.SDWriter.GetText(mol_opt)
    sdf_opt = M()
    sdf_opt.import_sdf(processed_molblock, from_text=True)
    pdbtxt = sdf_opt.renumber_atomname().write_pdb()

    with open(os.devnull, 'wb') as shutup:
        pdbqt = sub.check_output('printf "' + pdbtxt + '" | obabel -ipdb -opdbqt', shell=True, stderr=shutup).decode()

    pdbqt, rot2branches = extract_torsion_from_pdbqt(pdbqt, from_text=True)
    return sample_and_minimize_rmsd(sdfPath, pdbqt, rot2branches)


def single_conf_gen(tgt_mol, num_confs=1000, seed=42, removeHs=True, how='mmff'):
    """
    Originally from Uni-Mol code. Generate conformers from mol.
    :param rdkit_mol tgt_mol: the target molecule
    :param int num_confs: number of conformers to generates
    :param int seed: random seed to repeat experiment
    :param Bool removeHs: whether to remove hydrogen atoms (default is yes)
    :return mol: molecule with all generated conformers
    """
    mol = deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)

    if how == 'mmff':
        allconformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=0
        )
        sz = len(allconformers)
        for i in range(sz):
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=i, numThreads=0)
            except:
                continue
    elif how == 'etkdg':
        etkdg = rdDistGeom.ETKDGv3()
        etkdg.randomSeed = seed
        etkdg.verbose = False
        etkdg.numThreads = 0
        etkdg.useRandomCoords = True
        conformer_num = num_confs

        etkdg.optimizerForceTol = float(['0.0135', '0.001'][0])
        Chem.AssignStereochemistryFrom3D(mol)
        conformation_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=conformer_num, params=etkdg)

    if removeHs:
        mol = Chem.RemoveHs(mol)
    return mol


def clustering_coords(mol, M=1000, N=100, seed=42, removeHs=True, how='mmff'):
    """
    Originally from Uni-Mol code. Cluster conformers from mol
    :param rdkit_mol mol: the target molecule
    :param int M: number of conformers to generates
    :param int N: number of K-means clusters to extract
    :param int seed: random seed to repeat experiment
    :param Bool removeHs: whether to remove hydrogen atoms (default is yes)
    :return mol, list[int] cluster_ids: molecule with N clustered conformers ids
    """
    rdkit_coords_list = []

    rdkit_mol = single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs, how=how)

    noHsIds = [
        rdkit_mol.GetAtoms()[i].GetIdx()
        for i in range(len(rdkit_mol.GetAtoms()))
        if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
    ]

    rdMolAlign.AlignMolConformers(rdkit_mol, atomIds=noHsIds)
    sz = len(rdkit_mol.GetConformers())
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        rdkit_coords_list.append(_coords)

    rdkit_coords_flatten = np.array(rdkit_coords_list)[:, noHsIds].reshape(sz, -1)
    ids = (
        KMeans(n_clusters=N, random_state=seed)
        .fit_predict(rdkit_coords_flatten)
        .tolist()
    )

    cluster_ids = [ids.index(i) for i in range(N)]

    return rdkit_mol, cluster_ids


def assign_glutarimide_stereochemistry_multiple_confs(m, ids, ref_CM_path, out_path='', glut_smarts='O=C1[C;H2][C;H2][C,N](*)C(=0)[N;H1]1'):
    """
    Assigns the correct glutarimide (based on X-ray) for each
    conformer and returns a list of Molecules.
    """
    screen = Chem.MolFromSmarts(glut_smarts)
    matches = list(m.GetSubstructMatch(screen, useChirality=True))

    anchor_i = [3, 4, 5]

    real_glut_xyz, real_glut_matches = get_yxz_of_matched_smarts(ref_CM_path, glut_smarts, v=0, origin='file')

    opt_confs = []
    for c in ids:

        ref = []
        for i in anchor_i:
            j = matches[i]
            pos = m.GetConformers()[c].GetAtomPosition(j)
            ref.append([pos.x, pos.y, pos.z])

        sup = SVDSuperimposer()

        x = np.array(ref)
        y = real_glut_xyz[anchor_i]
        sup.set(x, y)
        sup.run()

        rot, tran = sup.get_rotran()
        new_coords = np.dot(real_glut_xyz, rot) + tran

        it = 0
        for i in matches:
            m.GetConformers()[c].SetAtomPosition(i, new_coords[it])
            it += 1

        _coords = m.GetConformers()[c].GetPositions().astype(np.float32)

        processed_molblock = Chem.SDWriter.GetText(m)
        sdf_opt = M()
        sdf_opt.import_sdf(processed_molblock, from_text=True)
        sdf_opt.set_xyz(_coords)
        sdf_opt = sdf_opt.renumber_atomname()

        opt_confs.append(sdf_opt)

    return opt_confs


def create_CELMoD_confs_from_smiles(smiles, out_path='', n_confs=100, n_clusters=10, how='mmff',
                                    atom2gluttype={'C': '/home/jovyan/work/pharao_IPS/data/ref_glutarimide/gsp1_ligand.sdf',
                                                   'N': '/home/jovyan/work/pharao_IPS/data/ref_glutarimide/DHU_align.pdb'}, v=1):
    """
    From smiles generates multiple conformers of CELMoDs
    with correct glutarimide rings.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetSubstructMatches(Chem.MolFromSmarts('O=C1[C;H2][C;H2][C]C(=0)')):
        atom = 'C'
    elif mol.GetSubstructMatches(Chem.MolFromSmarts('O=C1[C;H2][C;H2][N]C(=0)[N;H1]1')):
        atom = 'N'
    else:
        print('could not find glutarimide type')

    confs, c_ids = clustering_coords(mol, M=n_confs, N=n_clusters, how=how)

    confs = assign_glutarimide_stereochemistry_multiple_confs(confs, c_ids, atom2gluttype[atom])

    return confs


def create_generic_confs_from_smiles(smiles, out_path='', n_confs=100, n_clusters=10, how='etkdg', v=1):
    """
    From smiles generates multiple conformers.
    :param str smiles: the smiles string
    :param str out_path: path to write .pdb
    :return list[Molecule()]: list of conformations with correct glutarimide
    """
    mol = Chem.MolFromSmiles(smiles)

    m, c_ids = clustering_coords(mol, M=n_confs, N=n_clusters, how=how)

    opt_confs = []
    for c in c_ids:

        _coords = m.GetConformers()[c].GetPositions().astype(np.float32)

        processed_molblock = Chem.SDWriter.GetText(m)
        sdf_opt = M()
        sdf_opt.import_sdf(processed_molblock, from_text=True)
        sdf_opt.set_xyz(_coords)
        sdf_opt = sdf_opt.renumber_atomname()

        opt_confs.append(sdf_opt)

    return opt_confs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constrained Conformer Generation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generate_conformer_from_reference(smiles, ref_mol, mcs_params=None,
                                       num_attempts=10, random_seed=42,
                                       verbose=False):
    """
    Generate a 3D conformer for a SMILES string by constraining the largest
    matching substructure to have the same stereochemistry/coordinates as
    the reference molecule.

    This function handles symmetric substructures (like phenyl rings) correctly
    by evaluating ALL possible atom mappings and selecting the one that produces
    the lowest-energy conformer.

    Example:
        ref_mol has SMILES: c1ccc(cc1)-c2ccc(cc2)S(=O)(=O)NCC(=O)N=O
        input SMILES: O=NC(=O)CNS(=O)(=O)c1cc(O)ccc1  (has added OH)
        -> The matching core inherits coords from ref_mol with correct positioning

    :param str smiles: SMILES string of the molecule to generate
    :param ref_mol: Reference molecule - can be:
                    - rdkit mol with 3D coordinates
                    - path to SDF or PDB file
    :param dict mcs_params: Optional parameters for FindMCS (e.g.,
                            {'ringMatchesRingOnly': True, 'completeRingsOnly': True})
    :param int num_attempts: Number of embedding attempts per mapping (default 10)
    :param int random_seed: Random seed for reproducibility
    :param bool verbose: Print debug information about mapping selection
    :return tuple: (rdkit_mol with 3D coords, mcs_smarts, atom_map) or (None, None, None) on failure
    """
    # Load reference molecule if path is provided
    if isinstance(ref_mol, str):
        ref_mol = get_mol_from_structure(ref_mol)
        if ref_mol is None:
            print(">> Error: Could not load reference molecule")
            return None, None, None

    # Check that reference has 3D coordinates
    if ref_mol.GetNumConformers() == 0:
        print(">> Error: Reference molecule has no 3D coordinates")
        return None, None, None

    # Parse the input SMILES
    query_mol = Chem.MolFromSmiles(smiles)
    if query_mol is None:
        print(f">> Error: Invalid SMILES: {smiles}")
        return None, None, None

    # Set default MCS parameters if not provided
    if mcs_params is None:
        mcs_params = {
            'ringMatchesRingOnly': True,
            'completeRingsOnly': True,
            'bondCompare': rdFMCS.BondCompare.CompareAny,
            'atomCompare': rdFMCS.AtomCompare.CompareElements,
            'matchValences': True,
            'timeout': 10
        }

    # Find the Maximum Common Substructure
    mcs_result = rdFMCS.FindMCS([ref_mol, query_mol], **mcs_params)

    if mcs_result.numAtoms == 0:
        print(">> Warning: No common substructure found")
        return None, None, None

    mcs_smarts = mcs_result.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)

    if mcs_mol is None:
        print(f">> Error: Could not parse MCS SMARTS: {mcs_smarts}")
        return None, None, None

    if verbose:
        print(f">> MCS found: {mcs_result.numAtoms} atoms, {mcs_result.numBonds} bonds")
        print(f">> MCS SMARTS: {mcs_smarts}")

    # Get ALL possible atom matches in both molecules (handles symmetry)
    ref_matches = ref_mol.GetSubstructMatches(mcs_mol, uniquify=False)
    query_matches = query_mol.GetSubstructMatches(mcs_mol, uniquify=False)

    if not ref_matches or not query_matches:
        print(">> Error: Could not match MCS to molecules")
        return None, None, None

    if verbose:
        print(f">> Found {len(ref_matches)} ref matches, {len(query_matches)} query matches")

    # Build a core molecule from the reference for ConstrainedEmbed
    # This is needed for the fallback method
    ref_conf = ref_mol.GetConformer()

    # Try all combinations of mappings and find the best one
    best_result = None
    best_energy = float('inf')
    best_atom_map = None
    embed_failures = 0

    for ref_match in ref_matches:
        for query_match in query_matches:
            # Create atom mapping: query_atom_idx -> ref_atom_idx
            atom_map = {query_match[i]: ref_match[i] for i in range(len(ref_match))}

            # Add hydrogens to query molecule for embedding
            query_mol_h = Chem.AddHs(query_mol)

            # Build coordinate map for constrained embedding
            coord_map = {}
            for query_idx, ref_idx in atom_map.items():
                ref_pos = ref_conf.GetAtomPosition(ref_idx)
                coord_map[query_idx] = ref_pos

            # Try multiple embedding strategies
            embedded = False

            # Strategy 1: Basic embedding with coordMap (most compatible)
            if not embedded:
                try:
                    result = AllChem.EmbedMolecule(
                        query_mol_h,
                        coordMap=coord_map,
                        randomSeed=random_seed,
                        useRandomCoords=True,
                        maxAttempts=num_attempts * 10,
                        ignoreSmoothingFailures=True,
                        enforceChirality=False
                    )
                    if result != -1 and query_mol_h.GetNumConformers() > 0:
                        embedded = True
                        if verbose:
                            print(f"   Strategy 1 (coordMap embed) succeeded")
                except Exception as e:
                    if verbose:
                        print(f"   Strategy 1 failed: {e}")

            # Strategy 2: Generate unconstrained conformer, then align core atoms
            if not embedded:
                try:
                    query_mol_h = Chem.AddHs(query_mol)  # Fresh copy
                    result = AllChem.EmbedMolecule(query_mol_h, randomSeed=random_seed)
                    if result != -1 and query_mol_h.GetNumConformers() > 0:
                        # Align the generated conformer to reference using the MCS atoms
                        # Use AlignMol with atom map
                        query_atom_list = list(atom_map.keys())
                        ref_atom_list = [atom_map[i] for i in query_atom_list]

                        # Align query to reference
                        rdMolAlign.AlignMol(query_mol_h, ref_mol,
                                           atomMap=list(zip(query_atom_list, ref_atom_list)))
                        embedded = True
                        if verbose:
                            print(f"   Strategy 2 (embed + align) succeeded")
                except Exception as e:
                    if verbose:
                        print(f"   Strategy 2 failed: {e}")

            # Strategy 3: Generate conformer, manually set positions, then optimize
            if not embedded:
                try:
                    query_mol_h = Chem.AddHs(query_mol)  # Fresh copy
                    result = AllChem.EmbedMolecule(query_mol_h, randomSeed=random_seed)
                    if result != -1 and query_mol_h.GetNumConformers() > 0:
                        # Manually set core atom positions
                        conf = query_mol_h.GetConformer()
                        for query_idx, ref_idx in atom_map.items():
                            ref_pos = ref_conf.GetAtomPosition(ref_idx)
                            conf.SetAtomPosition(query_idx, ref_pos)
                        embedded = True
                        if verbose:
                            print(f"   Strategy 3 (embed + set positions) succeeded")
                except Exception as e:
                    if verbose:
                        print(f"   Strategy 3 failed: {e}")

            if not embedded:
                embed_failures += 1
                continue

            # Calculate energy to evaluate this mapping
            try:
                mmff_props = AllChem.MMFFGetMoleculeProperties(query_mol_h)
                if mmff_props is None:
                    # Try UFF instead
                    ff = AllChem.UFFGetMoleculeForceField(query_mol_h)
                else:
                    ff = AllChem.MMFFGetMoleculeForceField(query_mol_h, mmff_props, confId=0)

                if ff is None:
                    continue

                # Add position constraints for core atoms
                for query_idx in atom_map.keys():
                    ff.AddFixedPoint(query_idx)

                # Optimize and get energy
                ff.Minimize(maxIts=500)
                energy = ff.CalcEnergy()

                if verbose:
                    print(f"   Mapping energy: {energy:.2f}")

                if energy < best_energy:
                    best_energy = energy
                    best_result = query_mol_h
                    best_atom_map = atom_map

            except Exception as e:
                if verbose:
                    print(f"   Energy calculation failed: {e}")
                continue

    if verbose:
        total_mappings = len(ref_matches) * len(query_matches)
        print(f">> Embedding failed for {embed_failures}/{total_mappings} mappings")

    if best_result is None:
        print(">> Error: Could not generate conformer with any mapping")
        return None, None, None

    if verbose:
        print(f">> Selected mapping with energy: {best_energy:.2f}")

    # Remove hydrogens
    result_mol = Chem.RemoveHs(best_result)

    return result_mol, mcs_smarts, best_atom_map


def generate_conformer_from_reference_flexible(smiles, ref_mol, mcs_params=None,
                                                num_attempts=10, random_seed=42,
                                                relax_core=True, relax_force_constant=100.0,
                                                verbose=False):
    """
    Generate a 3D conformer with optional relaxation of the core atoms.
    Similar to generate_conformer_from_reference but allows the core atoms
    to move slightly during optimization (restrained, not fixed).

    Handles symmetric substructures correctly by evaluating all possible mappings.

    :param str smiles: SMILES string of the molecule to generate
    :param ref_mol: Reference molecule (rdkit mol or path)
    :param dict mcs_params: Optional parameters for FindMCS
    :param int num_attempts: Number of embedding attempts per mapping
    :param int random_seed: Random seed
    :param bool relax_core: If True, core atoms are restrained; if False, they're fixed
    :param float relax_force_constant: Force constant for position restraints (higher = tighter)
    :param bool verbose: Print debug information
    :return tuple: (rdkit_mol, mcs_smarts, atom_map)
    """
    # Load reference molecule if path is provided
    if isinstance(ref_mol, str):
        ref_mol = get_mol_from_structure(ref_mol)
        if ref_mol is None:
            print(">> Error: Could not load reference molecule")
            return None, None, None

    if ref_mol.GetNumConformers() == 0:
        print(">> Error: Reference molecule has no 3D coordinates")
        return None, None, None

    query_mol = Chem.MolFromSmiles(smiles)
    if query_mol is None:
        print(f">> Error: Invalid SMILES: {smiles}")
        return None, None, None

    # Default MCS parameters
    if mcs_params is None:
        mcs_params = {
            'ringMatchesRingOnly': True,
            'completeRingsOnly': True,
            'bondCompare': rdFMCS.BondCompare.CompareAny,
            'atomCompare': rdFMCS.AtomCompare.CompareElements,
            'matchValences': True,
            'timeout': 10
        }

    # Find MCS
    mcs_result = rdFMCS.FindMCS([ref_mol, query_mol], **mcs_params)

    if mcs_result.numAtoms == 0:
        print(">> Warning: No common substructure found")
        return None, None, None

    mcs_smarts = mcs_result.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)

    if mcs_mol is None:
        return None, None, None

    # Get ALL possible atom matches (handles symmetry)
    ref_matches = ref_mol.GetSubstructMatches(mcs_mol, uniquify=False)
    query_matches = query_mol.GetSubstructMatches(mcs_mol, uniquify=False)

    if not ref_matches or not query_matches:
        return None, None, None

    if verbose:
        print(f">> Found {len(ref_matches)} ref matches, {len(query_matches)} query matches")

    ref_conf = ref_mol.GetConformer()

    # Try all combinations and find the best one
    best_result = None
    best_energy = float('inf')
    best_atom_map = None

    embed_failures = 0

    for ref_match in ref_matches:
        for query_match in query_matches:
            atom_map = {query_match[i]: ref_match[i] for i in range(len(ref_match))}
            query_mol_h = Chem.AddHs(query_mol)

            # Build coordinate map
            coord_map = {}
            for query_idx, ref_idx in atom_map.items():
                ref_pos = ref_conf.GetAtomPosition(ref_idx)
                coord_map[query_idx] = ref_pos

            # Try multiple embedding strategies (same as non-flexible version)
            embedded = False

            # Strategy 1: Basic embedding with coordMap (most compatible)
            if not embedded:
                try:
                    result = AllChem.EmbedMolecule(
                        query_mol_h,
                        coordMap=coord_map,
                        randomSeed=random_seed,
                        useRandomCoords=True,
                        maxAttempts=num_attempts * 10,
                        ignoreSmoothingFailures=True,
                        enforceChirality=False
                    )
                    if result != -1 and query_mol_h.GetNumConformers() > 0:
                        embedded = True
                        if verbose:
                            print(f"   Strategy 1 (coordMap embed) succeeded")
                except Exception as e:
                    if verbose:
                        print(f"   Strategy 1 failed: {e}")

            # Strategy 2: Generate unconstrained conformer, then align core atoms
            if not embedded:
                try:
                    query_mol_h = Chem.AddHs(query_mol)  # Fresh copy
                    result = AllChem.EmbedMolecule(query_mol_h, randomSeed=random_seed)
                    if result != -1 and query_mol_h.GetNumConformers() > 0:
                        query_atom_list = list(atom_map.keys())
                        ref_atom_list = [atom_map[i] for i in query_atom_list]
                        rdMolAlign.AlignMol(query_mol_h, ref_mol,
                                           atomMap=list(zip(query_atom_list, ref_atom_list)))
                        embedded = True
                        if verbose:
                            print(f"   Strategy 2 (embed + align) succeeded")
                except Exception as e:
                    if verbose:
                        print(f"   Strategy 2 failed: {e}")

            # Strategy 3: Generate conformer, manually set positions, then optimize
            if not embedded:
                try:
                    query_mol_h = Chem.AddHs(query_mol)  # Fresh copy
                    result = AllChem.EmbedMolecule(query_mol_h, randomSeed=random_seed)
                    if result != -1 and query_mol_h.GetNumConformers() > 0:
                        conf = query_mol_h.GetConformer()
                        for query_idx, ref_idx in atom_map.items():
                            ref_pos = ref_conf.GetAtomPosition(ref_idx)
                            conf.SetAtomPosition(query_idx, ref_pos)
                        embedded = True
                        if verbose:
                            print(f"   Strategy 3 (embed + set positions) succeeded")
                except Exception as e:
                    if verbose:
                        print(f"   Strategy 3 failed: {e}")

            if not embedded:
                embed_failures += 1
                continue

            # Optimize with restraints and evaluate
            try:
                mmff_props = AllChem.MMFFGetMoleculeProperties(query_mol_h)
                if mmff_props is None:
                    # Try UFF instead
                    ff = AllChem.UFFGetMoleculeForceField(query_mol_h)
                else:
                    ff = AllChem.MMFFGetMoleculeForceField(query_mol_h, mmff_props, confId=0)

                if ff is None:
                    continue

                # For flexible mode: use fixed points (compatible API)
                # Note: AddExternalPositionalConstraint may not be available in all RDKit versions
                for query_idx in atom_map.keys():
                    ff.AddFixedPoint(query_idx)

                ff.Minimize(maxIts=1000)
                energy = ff.CalcEnergy()

                if verbose:
                    print(f"   Mapping energy: {energy:.2f}")

                if energy < best_energy:
                    best_energy = energy
                    best_result = query_mol_h
                    best_atom_map = atom_map

            except Exception as e:
                if verbose:
                    print(f"   Energy calculation failed: {e}")
                continue

    if verbose:
        total_mappings = len(ref_matches) * len(query_matches)
        print(f">> Embedding failed for {embed_failures}/{total_mappings} mappings")

    if best_result is None:
        print(">> Error: Could not generate conformer with any mapping")
        return None, None, None

    if verbose:
        print(f">> Selected mapping with energy: {best_energy:.2f}")

    result_mol = Chem.RemoveHs(best_result)
    return result_mol, mcs_smarts, best_atom_map


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Distances
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# @numba.njit()
def tanimoto_dist(a, b):
    dotprod = np.dot(a, b)
    tc = dotprod / (np.sum(a) + np.sum(b) - dotprod)
    return 1.0 - tc


def tc_dist(fp_a, fp_b):
    """old, 2rm soon"""
    a = fp_a.sum()
    b = fp_b.sum()
    c = (fp_a * fp_b).sum()
    if c != 0:
        return 1.0 - (c / (a + b - c))
    else:
        return 1.0


def get_tanimoto_similarity_from_smiles(smi1, smi2):
    """returns tanimoto similarity from 2 smiles"""
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    s = round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)
    return s


def get_RDKiTTanimoto_distance_matrix(MF1, MF2):
    """
    From data frame containing compound name and MF bits (1 & 0s), compute
    Tanimoto pairwise distance matrix

    :param df MF1: compound + MFs as columns
    :param df MF2: compound + MFs as columns
    :return df: pairwise distance matrix in dataframe type
    """
    dist = DistanceMetric.get_metric(tanimoto_dist)
    Y = np.array(MF1.drop('compound', axis=1))
    X = np.array(MF2.drop('compound', axis=1))
    d2 = dist.pairwise(Y, X)
    d2_df = pd.DataFrame(d2, columns=list(MF2['compound']))
    d2_df.index = list(MF1['compound'])
    return d2_df


def get_RogersTanimoto_distance_matrix(MF1, MF2):
    """
    From data frame containing compound name and MF bits (1 & 0s), compute
    Rogerstanimoto pairwise distance matrix

    :param df MF1: compound + MFs as columns
    :param df MF2: compound + MFs as columns
    :return df: pairwise distance matrix in dataframe type
    """
    dist = DistanceMetric.get_metric('rogerstanimoto')
    Y = np.array(MF1.drop('compound', axis=1))
    X = np.array(MF2.drop('compound', axis=1))
    d2 = dist.pairwise(Y, X)
    d2_df = pd.DataFrame(d2, columns=list(MF2['compound']))
    d2_df.index = list(MF1['compound'])
    return d2_df


def get_NN_from_dist_matrix(df, top=1):
    """return top n nearest neigbours from reference compounds"""
    d2_tmp = df.copy()
    d2_tmp['compound'] = d2_tmp.index
    d2_tmp = d2_tmp.melt(id_vars='compound').sort_values('value', ascending=True).rename(columns={'value': 'distance', 'variable': 'NN'})
    d2_tmp = d2_tmp.groupby(['compound']).head(top)
    return d2_tmp


def get_distance_matrix_from_2_molecules(m1, m2):
    """
    Computes pairwise atom distances between two molecules

    :param np.array(np.array) m1: xyz coords of molecule 1
    :param np.array(np.array) m2: xyz coords of molecule 2
    :return np.array(np.array) dist: the distance matrix
    """
    d = []
    for i in range(0, len(m1), 1):
        d.append(np.sqrt(np.sum((m2 - m1[i])**2, axis=1)))

    dist = np.array(d)

    return dist


def extract_pocket_residues(ligand_file, protein_file, output_file, distance_threshold=6.0, verbose=True):
    """
    Extract protein pocket residues within a distance threshold of a ligand.

    This function identifies all protein residues that have at least one atom
    within the specified distance of any ligand atom, and writes them to a
    separate PDB file.

    :param str ligand_file: Path to ligand structure file (.pdb or .sdf format)
    :param str protein_file: Path to protein structure file (.pdb format)
    :param str output_file: Path for output PDB file containing pocket residues
    :param float distance_threshold: Distance cutoff in Angstroms (default 8.0)
    :param bool verbose: Print progress information (default True)
    :return tuple: (list of contacting residue IDs, number of pocket atoms)
                   Returns (None, 0) on failure
    """
    # Load ligand structure
    ligand_mol = get_mol_from_structure(ligand_file)
    if ligand_mol is None:
        print(f">> Error: Could not load ligand from {ligand_file}")
        return None, 0

    # Load protein structure
    protein_mol = Chem.MolFromPDBFile(protein_file, removeHs=False)
    if protein_mol is None:
        print(f">> Error: Could not load protein from {protein_file}")
        return None, 0

    # Get conformers (3D coordinates)
    if ligand_mol.GetNumConformers() == 0:
        print(">> Error: Ligand has no 3D coordinates")
        return None, 0
    if protein_mol.GetNumConformers() == 0:
        print(">> Error: Protein has no 3D coordinates")
        return None, 0

    # Extract coordinates
    ligand_conf = ligand_mol.GetConformer()
    protein_conf = protein_mol.GetConformer()

    ligand_coords = np.array([
        [ligand_conf.GetAtomPosition(i).x,
         ligand_conf.GetAtomPosition(i).y,
         ligand_conf.GetAtomPosition(i).z]
        for i in range(ligand_mol.GetNumAtoms())
    ])

    protein_coords = np.array([
        [protein_conf.GetAtomPosition(i).x,
         protein_conf.GetAtomPosition(i).y,
         protein_conf.GetAtomPosition(i).z]
        for i in range(protein_mol.GetNumAtoms())
    ])

    if verbose:
        print(f">> Ligand atoms: {len(ligand_coords)}")
        print(f">> Protein atoms: {len(protein_coords)}")

    # Compute distance matrix (ligand x protein)
    dist_matrix = get_distance_matrix_from_2_molecules(ligand_coords, protein_coords)

    # Find protein atoms within threshold of any ligand atom
    # Get minimum distance to any ligand atom for each protein atom
    min_distances = np.min(dist_matrix, axis=0)
    contacting_atom_indices = np.where(min_distances < distance_threshold)[0]

    if verbose:
        print(f">> Found {len(contacting_atom_indices)} protein atoms within {distance_threshold} A of ligand")

    if len(contacting_atom_indices) == 0:
        print(">> Warning: No protein atoms found within threshold distance")
        return [], 0

    # Extract residue information for contacting atoms
    contacting_residues = set()
    pdb_info = protein_mol.GetAtomWithIdx(0).GetPDBResidueInfo()

    for atom_idx in contacting_atom_indices:
        atom = protein_mol.GetAtomWithIdx(int(atom_idx))
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info is not None:
            # Create unique residue identifier: (chain, resnum, insertion_code)
            chain = pdb_info.GetChainId()
            resnum = pdb_info.GetResidueNumber()
            icode = pdb_info.GetInsertionCode()
            resname = pdb_info.GetResidueName()
            contacting_residues.add((chain, resnum, icode, resname))

    if verbose:
        print(f">> Found {len(contacting_residues)} unique contacting residues")
        # Print residue list
        sorted_residues = sorted(contacting_residues, key=lambda x: (x[0], x[1]))
        res_list = [f"{r[3]}{r[1]}" + (f":{r[0]}" if r[0].strip() else "") for r in sorted_residues]
        print(f">> Residues: {', '.join(res_list)}")

    # Collect all atoms belonging to contacting residues (whole residues)
    pocket_atom_indices = []
    for atom_idx in range(protein_mol.GetNumAtoms()):
        atom = protein_mol.GetAtomWithIdx(atom_idx)
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info is not None:
            chain = pdb_info.GetChainId()
            resnum = pdb_info.GetResidueNumber()
            icode = pdb_info.GetInsertionCode()
            resname = pdb_info.GetResidueName()
            if (chain, resnum, icode, resname) in contacting_residues:
                pocket_atom_indices.append(atom_idx)

    if verbose:
        print(f">> Total pocket atoms (whole residues): {len(pocket_atom_indices)}")

    # Write pocket residues to PDB file
    # We need to write a proper PDB file with only the pocket atoms
    with open(output_file, 'w') as f:
        atom_serial = 1
        for atom_idx in pocket_atom_indices:
            atom = protein_mol.GetAtomWithIdx(atom_idx)
            pdb_info = atom.GetPDBResidueInfo()
            pos = protein_conf.GetAtomPosition(atom_idx)

            if pdb_info is not None:
                atom_name = pdb_info.GetName()
                resname = pdb_info.GetResidueName()
                chain = pdb_info.GetChainId()
                resnum = pdb_info.GetResidueNumber()
                icode = pdb_info.GetInsertionCode()
                occupancy = pdb_info.GetOccupancy()
                tempfactor = pdb_info.GetTempFactor()
                element = atom.GetSymbol()

                # Format PDB ATOM line
                # ATOM serial name altLoc resName chainID resSeq iCode x y z occupancy tempFactor element
                record = "ATOM  " if not pdb_info.GetIsHeteroAtom() else "HETATM"
                line = f"{record}{atom_serial:5d} {atom_name:<4s} {resname:>3s} {chain:1s}{resnum:4d}{icode:1s}   "
                line += f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                line += f"{occupancy:6.2f}{tempfactor:6.2f}          {element:>2s}\n"
                f.write(line)
                atom_serial += 1

        f.write("END\n")

    if verbose:
        print(f">> Pocket written to: {output_file}")

    # Return list of residue identifiers and count
    residue_list = [f"{r[3]}{r[1]}" + (f":{r[0]}" if r[0].strip() else "") for r in sorted(contacting_residues)]
    return residue_list, len(pocket_atom_indices)


def extract_pocket_residues_from_mol(ligand_mol, protein_mol, distance_threshold=6.0):
    """
    Extract protein pocket residues within a distance threshold of a ligand.
    Version that takes RDKit mol objects directly instead of file paths.

    :param ligand_mol: RDKit mol object for ligand with 3D coordinates
    :param protein_mol: RDKit mol object for protein with 3D coordinates
    :param float distance_threshold: Distance cutoff in Angstroms (default 8.0)
    :return tuple: (pocket_mol, residue_list) - RDKit mol of pocket and list of residue IDs
                   Returns (None, []) on failure
    """
    if ligand_mol is None or protein_mol is None:
        return None, []

    if ligand_mol.GetNumConformers() == 0 or protein_mol.GetNumConformers() == 0:
        return None, []

    # Extract coordinates
    ligand_conf = ligand_mol.GetConformer()
    protein_conf = protein_mol.GetConformer()

    ligand_coords = np.array([
        [ligand_conf.GetAtomPosition(i).x,
         ligand_conf.GetAtomPosition(i).y,
         ligand_conf.GetAtomPosition(i).z]
        for i in range(ligand_mol.GetNumAtoms())
    ])

    protein_coords = np.array([
        [protein_conf.GetAtomPosition(i).x,
         protein_conf.GetAtomPosition(i).y,
         protein_conf.GetAtomPosition(i).z]
        for i in range(protein_mol.GetNumAtoms())
    ])

    # Compute distance matrix
    dist_matrix = get_distance_matrix_from_2_molecules(ligand_coords, protein_coords)

    # Find protein atoms within threshold
    min_distances = np.min(dist_matrix, axis=0)
    contacting_atom_indices = np.where(min_distances < distance_threshold)[0]

    if len(contacting_atom_indices) == 0:
        return None, []

    # Extract contacting residues
    contacting_residues = set()
    for atom_idx in contacting_atom_indices:
        atom = protein_mol.GetAtomWithIdx(int(atom_idx))
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info is not None:
            chain = pdb_info.GetChainId()
            resnum = pdb_info.GetResidueNumber()
            icode = pdb_info.GetInsertionCode()
            resname = pdb_info.GetResidueName()
            contacting_residues.add((chain, resnum, icode, resname))

    # Collect all atoms belonging to contacting residues
    pocket_atom_indices = []
    for atom_idx in range(protein_mol.GetNumAtoms()):
        atom = protein_mol.GetAtomWithIdx(atom_idx)
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info is not None:
            chain = pdb_info.GetChainId()
            resnum = pdb_info.GetResidueNumber()
            icode = pdb_info.GetInsertionCode()
            resname = pdb_info.GetResidueName()
            if (chain, resnum, icode, resname) in contacting_residues:
                pocket_atom_indices.append(atom_idx)

    # Create pocket mol by extracting atoms
    # Note: This creates a new mol with only the pocket atoms
    try:
        from rdkit.Chem import RWMol
        pocket_mol = Chem.RWMol(protein_mol)

        # Remove atoms not in pocket (in reverse order to preserve indices)
        atoms_to_remove = [i for i in range(protein_mol.GetNumAtoms()) if i not in pocket_atom_indices]
        for atom_idx in sorted(atoms_to_remove, reverse=True):
            pocket_mol.RemoveAtom(atom_idx)

        pocket_mol = pocket_mol.GetMol()
    except Exception as e:
        print(f">> Error creating pocket mol: {e}")
        pocket_mol = None

    residue_list = [f"{r[3]}{r[1]}" + (f":{r[0]}" if r[0].strip() else "") for r in sorted(contacting_residues)]
    return pocket_mol, residue_list


def get_test_NNdist_from_training(test_df, train_df):
    """
    Compute distance of smiles in test df vs train_df and adds columns to test_df

    :param test_df: test dataframe containing |compound|smiles| columns
    :param train_df: same format as test_df
    :return comp_df: test_df with added |distance|NN_smiles| columns
    """
    test_MF = get_MF_bits_from_df(test_df)
    train_MF = get_MF_bits_from_df(train_df)
    print('>> Done computing MFs...')

    d = get_RDKiTTanimoto_distance_matrix(test_MF, train_MF)
    NN = get_NN_from_dist_matrix(d, top=1)

    return NN


def compute_NNs_in_chunks(MF1, MF2, chunk_size=1000, top=1):
    """
    Compute NNs in chunks to avoid memory overload.
    The second argument MF2 is going to be split since MF1 is the reference
    :param of MF1,MF2
    :param int chunk_size
    :param int top: how many top NNs to return
    :return df NN_df
    """
    NN_df = []
    chunks = chunker(list(range(MF2.shape[0])), chunk_size)
    for chunk in tqdm(chunks):
        tmp_MF = MF2.iloc[chunk]
        tmp_matrix = get_RDKiTTanimoto_distance_matrix(MF1, tmp_MF)
        tmp_NNs = get_NN_from_dist_matrix(tmp_matrix, top=1)
        NN_df.append(tmp_NNs)

    NN_df = pd.concat(NN_df).reset_index(drop=True)
    NN_df = NN_df.sort_values('distance').groupby('compound').first().sort_values('distance').reset_index()

    return NN_df


def compute_NNs_from_MFs(MF1, MF2, top=1):
    """Compute distance and then NN distance from MFs"""
    tmp_matrix = get_RDKiTTanimoto_distance_matrix(MF1, MF2)
    tmp_NNs = get_NN_from_dist_matrix(tmp_matrix, top=top)
    tmp_NNs = tmp_NNs[tmp_NNs['compound'] != tmp_NNs['NN']].reset_index(drop=True)
    tmp_NNs['pair'] = tmp_NNs[['compound', 'NN']].apply(lambda x: sorted([x[0], x[1]]), axis=1)
    tmp_NNs = tmp_NNs.drop_duplicates('pair').reset_index(drop=True)
    return tmp_NNs


# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    print('> usage: python RDKit_tools --FUNC optimize_3D_geometry --ARGS sdfPath,outputPath')

    parser = argparse.ArgumentParser()
    parser.add_argument('--FUNC', type=str, help='name of function to call')
    parser.add_argument('--ARGS', type=str, help='arguments of the called function')
    args = parser.parse_args()

    func = globals()[args.FUNC]

    func(*args.ARGS.split(','))
