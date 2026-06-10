"""Build local GO / pathway collections + a per-protein function table for the
Px screen gene universe, for annotating volcano plots (cell-signature analysis).

Everything is built from PUBLIC reference files (GO, Reactome, UniProt features
already exported locally) — no project data leaves the machine and no enrichment
web service is called.

Inputs (all local):
  - output/cell_signature/raw/go-basic.obo                 GO DAG (id, name, namespace, is_a)
  - output/cell_signature/raw/goa_human.gaf.gz             human gene-product -> GO annotations
  - output/cell_signature/raw/UniProt2Reactome_All_Levels.txt   UniProt -> Reactome pathway
  - data/cell_signature/maxquantAnnot.txt                  UniProt feature table (all-organism)
  - data/MS/Px_genes.csv                                   the gene universe (~12k symbols)

Outputs (output/cell_signature/):
  Step 1 — GO + pathway collections
    - go_terms.parquet         term_id, name, namespace (BP/MF/CC) + Reactome
    - gene2term.parquet        long: gene, collection, term_id, term_name   (GO propagated to ancestors)
    - {go_bp,go_mf,go_cc,reactome}.gmt   gene-set files for enrichment (term_id<TAB>name<TAB>genes...)
    - human_gene_uniprot.parquet   gene <-> UniProt accession (from the human GAF)
  Step 0 — per-protein function
    - protein_function.parquet   one row per Px gene: uniprot, protein_name, pfam,
                                 transmembrane/signal_peptide/dna_binding/... flags, domains

Run:  python python/build_cell_signature_annotations.py [--step0] [--step1]
      (no flag = both)
"""
import argparse
import csv
import gzip
import os
import sys

import pandas as pd
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'output/cell_signature/raw')
OUT = os.path.join(ROOT, 'output/cell_signature')
OBO = os.path.join(RAW, 'go-basic.obo')
GAF = os.path.join(RAW, 'goa_human.gaf.gz')
REACTOME = os.path.join(RAW, 'UniProt2Reactome_All_Levels.txt')
MAXQUANT = os.path.join(ROOT, 'data/cell_signature/maxquantAnnot.txt')
PX_GENES = os.path.join(ROOT, 'data/MS/Px_genes.csv')

NS_SHORT = {'biological_process': 'GO_BP',
            'molecular_function': 'GO_MF',
            'cellular_component': 'GO_CC'}

csv.field_size_limit(1 << 24)   # maxquant feature cells are large


# ----------------------------------------------------------------------------
# GO OBO
# ----------------------------------------------------------------------------
def parse_obo(path):
    """Return (name{id}, namespace{id}, child->parent DiGraph) for non-obsolete terms."""
    name, ns = {}, {}
    edges = []           # (child, parent) via is_a
    alt = {}             # alt_id -> primary id
    cur, obsolete = {}, False

    def flush():
        gid = cur.get('id')
        if gid and not obsolete:
            name[gid] = cur.get('name', gid)
            ns[gid] = cur.get('namespace', '')
            for p in cur.get('is_a', []):
                edges.append((gid, p))
            for a in cur.get('alt_id', []):
                alt[a] = gid

    with open(path) as f:
        in_term = False
        for line in f:
            line = line.rstrip('\n')
            if line == '[Term]':
                if in_term:
                    flush()
                cur, obsolete, in_term = {}, False, True
                continue
            if line.startswith('['):      # other stanza type ([Typedef] etc.)
                if in_term:
                    flush()
                in_term = False
                continue
            if not in_term or ':' not in line:
                continue
            key, _, val = line.partition(': ')
            if key == 'id':
                cur['id'] = val
            elif key == 'name':
                cur['name'] = val
            elif key == 'namespace':
                cur['namespace'] = val
            elif key == 'is_a':
                cur.setdefault('is_a', []).append(val.split(' ! ')[0].strip())
            elif key == 'alt_id':
                cur.setdefault('alt_id', []).append(val.strip())
            elif key == 'is_obsolete' and val.strip() == 'true':
                obsolete = True
        if in_term:
            flush()

    G = nx.DiGraph()
    G.add_nodes_from(name)
    G.add_edges_from((c, p) for c, p in edges if c in name and p in name)
    print(f'  OBO: {len(name):,} terms, {G.number_of_edges():,} is_a edges, {len(alt):,} alt_ids')
    return name, ns, G, alt


def parse_gaf(path):
    """human GAF -> (gene2direct_go{gene:set(go)}, gene2uniprot rows, human_acc set)."""
    gene2go, rows, human_acc = {}, [], set()
    opn = gzip.open if path.endswith('.gz') else open
    with opn(path, 'rt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            c = line.rstrip('\n').split('\t')
            if len(c) < 9:
                continue
            acc, sym, qual, go_id = c[1], c[2], c[3], c[4]
            if 'NOT' in qual or not sym or not go_id.startswith('GO:'):
                continue
            gene2go.setdefault(sym, set()).add(go_id)
            human_acc.add(acc)
            rows.append((sym, acc))
    g2u = pd.DataFrame(rows, columns=['gene', 'uniprot']).drop_duplicates()
    print(f'  GAF: {len(gene2go):,} genes annotated, {len(human_acc):,} human accessions')
    return gene2go, g2u, human_acc


def build_go_collections(px_genes):
    name, ns, G, alt = parse_obo(OBO)
    gene2go, g2u, human_acc = parse_gaf(GAF)

    # memoised ancestor sets (is_a closure): a gene annotated to T is also annotated
    # to every ancestor of T.
    anc_cache = {}

    def ancestors(t):
        if t in anc_cache:
            return anc_cache[t]
        a = nx.descendants(G, t) if t in G else set()
        anc_cache[t] = a
        return a

    long_rows = []   # gene, collection, term_id, term_name
    for gene, direct in gene2go.items():
        terms = set()
        for t in direct:
            t = alt.get(t, t)
            if t not in name:
                continue
            terms.add(t)
            terms |= ancestors(t)
        for t in terms:
            coll = NS_SHORT.get(ns.get(t, ''), None)
            if coll:
                long_rows.append((gene, coll, t, name[t]))

    # Reactome (filter to human) -> gene via the GAF accession->symbol map
    acc2sym = {}
    for g, a in g2u.itertuples(index=False):
        acc2sym.setdefault(a, g)
    rea_terms = {}
    with open(REACTOME) as f:
        for line in f:
            c = line.rstrip('\n').split('\t')
            if len(c) < 6 or c[5] != 'Homo sapiens':
                continue
            acc, pid, _url, pname = c[0], c[1], c[2], c[3]
            sym = acc2sym.get(acc) or acc2sym.get(acc.split('-')[0])
            if not sym:
                continue
            long_rows.append((sym, 'Reactome', pid, pname))
            rea_terms[pid] = pname

    gene2term = pd.DataFrame(long_rows, columns=['gene', 'collection', 'term_id', 'term_name']).drop_duplicates()
    # restrict to the Px universe (keep the full human map too? -> Px only per request)
    gene2term_px = gene2term[gene2term['gene'].isin(px_genes)].reset_index(drop=True)

    # term table
    terms_go = pd.DataFrame([(t, name[t], NS_SHORT[ns[t]]) for t in name if ns.get(t) in NS_SHORT],
                            columns=['term_id', 'name', 'collection'])
    terms_rea = pd.DataFrame([(p, n, 'Reactome') for p, n in rea_terms.items()],
                             columns=['term_id', 'name', 'collection'])
    terms = pd.concat([terms_go, terms_rea], ignore_index=True)

    os.makedirs(OUT, exist_ok=True)
    gene2term_px.to_parquet(os.path.join(OUT, 'gene2term.parquet'), index=False)
    terms.to_parquet(os.path.join(OUT, 'go_terms.parquet'), index=False)
    g2u.to_parquet(os.path.join(OUT, 'human_gene_uniprot.parquet'), index=False)

    # GMT files (restricted to Px genes) for enrichment tools
    for coll, fname in [('GO_BP', 'go_bp.gmt'), ('GO_MF', 'go_mf.gmt'),
                        ('GO_CC', 'go_cc.gmt'), ('Reactome', 'reactome.gmt')]:
        sub = gene2term_px[gene2term_px['collection'] == coll]
        with open(os.path.join(OUT, fname), 'w') as fh:
            for (tid, tname), grp in sub.groupby(['term_id', 'term_name']):
                genes = sorted(set(grp['gene']))
                if genes:
                    fh.write(f'{tid}\t{tname}\t' + '\t'.join(genes) + '\n')

    cov = gene2term_px['gene'].nunique()
    print(f'  gene2term: {len(gene2term_px):,} (gene,term) rows; {cov:,}/{len(px_genes):,} Px genes annotated')
    for coll in ['GO_BP', 'GO_MF', 'GO_CC', 'Reactome']:
        s = gene2term_px[gene2term_px['collection'] == coll]
        print(f'    {coll:9}: {s["term_id"].nunique():,} terms, {s["gene"].nunique():,} genes')
    return human_acc


# ----------------------------------------------------------------------------
# Step 0 — per-protein function (maxquantAnnot, human only, Px universe)
# ----------------------------------------------------------------------------
# header indices (verified): 0 Uniprot 1 Gene name 2 Protein name 3 Pfam 5 Reviewed
#   6 active site 7 binding site 14 dna-binding region 15 domain
#   27 nucleotide phosphate-binding region 35 signal peptide
#   41 transmembrane region 44 zinc finger region
COL = dict(uniprot=0, gene=1, protein_name=2, pfam=3, reviewed=5, active=6,
           binding=7, dna=14, domain=15, ntp=27, signal=35, tm=41, zinc=44)


def build_protein_function(px_genes):
    # Join on gene name (covers ~11.8k Px genes); the maxquant accessions don't
    # match the GAF set so we can't filter by them. `Reviewed` is 'T'/'F'
    # (Swiss-Prot); prefer reviewed + most-populated when a symbol maps to several
    # proteins (picks the human canonical for the lab's human-centric FASTA).
    def score(row):
        rev = 1 if row.get('reviewed') else 0
        filled = sum(bool(row.get(k)) for k in ('protein_name', 'pfam', 'domains'))
        return (rev, filled)

    best = {}   # gene -> (score, dict)
    with open(MAXQUANT) as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for c in r:
            if len(c) <= COL['zinc']:
                continue
            gene = c[COL['gene']]
            if gene not in px_genes:                  # universe filter (gene name)
                continue
            row = {
                'gene': gene, 'uniprot': c[COL['uniprot']],
                'protein_name': c[COL['protein_name']],
                'pfam': c[COL['pfam']][:200],
                'reviewed': c[COL['reviewed']].strip() == 'T',
                'transmembrane': bool(c[COL['tm']].strip()),
                'signal_peptide': bool(c[COL['signal']].strip()),
                'dna_binding': bool(c[COL['dna']].strip()),
                'nucleotide_binding': bool(c[COL['ntp']].strip()),
                'zinc_finger': bool(c[COL['zinc']].strip()),
                'catalytic': bool(c[COL['active']].strip()),
                'domains': c[COL['domain']][:250],
            }
            s = score(row)
            if gene not in best or s > best[gene][0]:
                best[gene] = (s, row)

    func = pd.DataFrame([v[1] for v in best.values()])
    func = func[['gene', 'uniprot', 'protein_name', 'pfam', 'reviewed',
                 'transmembrane', 'signal_peptide', 'dna_binding',
                 'nucleotide_binding', 'zinc_finger', 'catalytic', 'domains']]
    func.to_parquet(os.path.join(OUT, 'protein_function.parquet'), index=False)
    print(f'  protein_function: {len(func):,}/{len(px_genes):,} Px genes annotated '
          f'({func["protein_name"].astype(bool).sum():,} with a protein name)')
    return func


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step0', action='store_true', help='only the protein-function table')
    ap.add_argument('--step1', action='store_true', help='only the GO/pathway collections')
    a = ap.parse_args()
    do0 = a.step0 or not (a.step0 or a.step1)
    do1 = a.step1 or not (a.step0 or a.step1)

    px = set(pd.read_csv(PX_GENES)['genes'].dropna().astype(str))
    print(f'Px universe: {len(px):,} genes')

    if do1:
        print('Step 1 — GO + pathway collections')
        build_go_collections(px)
    if do0:
        print('Step 0 — per-protein function table')
        build_protein_function(px)
    print('done ->', OUT)


if __name__ == '__main__':
    main()
