import pandas as pd
import pyranges as pr
import pickle

gtf = pr.read_gtf("../gencode.v39.annotation.gtf")

genes_gr = gtf[gtf.Feature == "gene"]
genes_df = genes_gr.df[["Chromosome", "Start", "End", "Strand", "gene_name"]].copy()
genes_df.columns = ["chrom", "start", "end", "strand", "gene_name"]
genes_df = genes_df.dropna(subset=["gene_name"])
gene_agg = genes_df.groupby("gene_name").agg({
    "chrom": "first",
    "start": "min",
    "end": "max",
    "strand": "first"
}).reset_index()

genes_dict = gene_agg.set_index("gene_name").to_dict(orient="index")

exons_gr = gtf[gtf.Feature == "exon"]
exons_df = exons_gr.df[["Chromosome", "Start", "End", "Strand", "gene_name"]].copy()
exons_df.columns = ["chrom", "start", "end", "strand", "gene_name"]
exons_df = exons_df.dropna(subset=["gene_name"])
gene_exons = {}
for _, row in exons_df.iterrows():
    gene = row["gene_name"]
    if gene not in gene_exons:
        gene_exons[gene] = []
    gene_exons[gene].append((row["chrom"], row["start"], row["end"], row["strand"]))

with open("../data/gene_exons.pkl", "wb") as f:
    pickle.dump(gene_exons, f)
print("ready gene_exons.pkl")

def add_gene_info(df, genes_dict):
    df["gene_chrom"] = df["gene"].map(lambda g: genes_dict.get(g, {}).get("chrom", None))
    df["gene_start"] = df["gene"].map(lambda g: genes_dict.get(g, {}).get("start", None))
    df["gene_end"]   = df["gene"].map(lambda g: genes_dict.get(g, {}).get("end", None))
    df["gene_strand"] = df["gene"].map(lambda g: genes_dict.get(g, {}).get("strand", None))
    initial_len = len(df)
    df = df.dropna(subset=["gene_chrom", "gene_start", "gene_end", "gene_strand"])
    return df

train_df = pd.read_csv("../data/train_variants.csv", sep=",")
val_df   = pd.read_csv("../data/val_variants.csv", sep=",")

train_df = add_gene_info(train_df, genes_dict)
val_df   = add_gene_info(val_df, genes_dict)

train_df.to_csv("../data/train_with_genes.csv", sep=",", index=False)
val_df.to_csv("../data/val_with_genes.csv", sep=",", index=False)
print("end")
