#!/usr/bin/env python3
#
# ############################################################################
# Copyright (c) 2020 LRGASP consortium
# Prepares reference data in needed format and infers artifical novel isoforms
# ############################################################################

from BCBio import GFF
import pandas as pd
from Bio import SeqIO
import subprocess
import logging
import numpy
import random

logger = logging.getLogger('LRGASP')


def combine_references(reference_list, output_prefix):  # in 'data' insert your path to file with genomes and annotations
    df = pd.read_csv(reference_list, sep='\t', header=None)
    df.columns = ['genome', 'annotation']
    genome_id_list_without_duplicates = set()
    new_id_map = {}
    all_genes = []

    genomes_path = output_prefix + '.genomes.fasta'
    annotations_path = output_prefix + '.annotations.gtf'
    transcripts_path = output_prefix + '.transcripts.fasta'
    logger.info("Combining annotations and genomes")
    open(genomes_path, 'w').close()
    open(annotations_path, 'w').close()
    with open(genomes_path, 'a') as genomes_output, open(annotations_path, 'a') as annotations_output:
        for genome in range(len(df['genome'])):
            genome_path = df['genome'].iloc[genome]
            genome_id_list = []
            fasta_records = []
            for genome_rec in SeqIO.parse(genome_path, 'fasta'):
                genome_id_list.append(genome_rec.id)
                fasta_records.append(genome_rec)
            for i in range(len(genome_id_list)):
                if genome_id_list[i] in genome_id_list_without_duplicates:
                    new_id = '{}_{}'.format(genome_id_list[i], i)  # adds an iteration number in the end of duplicate id
                    genome_id_list_without_duplicates.add(new_id)
                    new_id_map[genome_id_list[i]] = new_id  # creates of dict with replaced id
                    genome_rec.id = new_id  # replaces id in genome_id_list and in fasta_records with new one
                if genome_id_list[i] not in genome_id_list_without_duplicates:
                    genome_id_list_without_duplicates.add(genome_id_list[i])
            SeqIO.write(fasta_records, genomes_output, 'fasta')

            annotation_path = df['annotation'].iloc[genome]
            annotation_id_list = []
            gff_records = []
            all_genes.append([])
            for annotation_rec in GFF.parse(annotation_path):
                annotation_id_list.append(annotation_rec.id)
                for feature in annotation_rec.features:
                    if feature.type == 'gene':
                        if 'ID' in feature.qualifiers:
                            all_genes[-1].append(feature.qualifiers['ID'][0])
                        elif 'gene_id' in feature.qualifiers:
                            all_genes[-1].append(feature.qualifiers['gene_id'][0])

                gff_records.append(annotation_rec)
            for i in range(len(annotation_id_list)):
                if annotation_id_list[i] in new_id_map:  # checks if id in annotation is in dict with replaced id
                    annotation_id_list[i] = new_id_map[annotation_id_list[i]]
                    annotation_rec.id = annotation_id_list[i]
            GFF.write(gff_records, annotations_output)

    with open(annotations_path, "r") as annotation_file:
        lines = annotation_file.readlines()

    with open(annotations_path, "w") as annotation_file:
        for number, line in enumerate(lines):
            if number not in [0, 1, 2]:
                annotation_file.write(line)
                
    extract_transcripts(genomes_path, annotations_path, transcripts_path)
    logger.info("Done.")
    return all_genes


def extract_transcripts(genome_path, annotation_path, transcripts_path):
    logger.info("Extracting transcripts")
    res = subprocess.run(['gffread', '-w', transcripts_path, '-g', genome_path, annotation_path])
    if res.returncode != 0:
        logger.error("gffread failed, contact developers for support.")
        return


def read_gene_counts(reference_list):
    df = pd.read_csv(reference_list, sep='\t', header=None)
    df.columns = ['genome', 'annotation']


def generate_meta_counts(all_genes, output_prefix):
    abundances = generate_abundances(len(all_genes))
    gene_coutns = []
    for j, gene_names in enumerate(all_genes):
        counts = generate_gene_counts(len(gene_names))
        for i, gene_id in enumerate(gene_names):
            gene_coutns.append((gene_id, counts[i] * abundances[j]))

    scale_factor = 1000000.0 / sum(map(lambda x:x[1], gene_coutns))
    with open(output_prefix + ".abundances.tsv", "w") as outf:
        for a in abundances:
            outf.write("%.8f\n" % a)
    with open(output_prefix + ".counts.tsv", "w") as outf:
        for gene_info in gene_coutns:
            outf.write("%s\t%.8f\n" % (gene_info[0], gene_info[1] * scale_factor))


def expression_func(x):
    #  2680.8823734890475, 227926.4856264025,
    #  1.0, 1.83150924535486
    a = -2251.914904831915
    b = 5.8732587449434076e-241
    c = 563.3187067205479
    d = 0.0006505959181508507
    return a + b * numpy.exp(c / x ** d)


def generate_gene_counts(total_genes, type=1):
    if type == 0:
        nums = numpy.arange(1, total_genes + 1, 1)
    else:
        nums = numpy.array([random.random() * total_genes / 4 for i in range(total_genes + 1)])
    tpms = []
    for i in range(total_genes):
        tpms.append(0)
    tpms = expression_func(nums)
    # fig, ax = plt.subplots()
    # ax.plot(nums, tpms, label="tpms")
    tpms = list(map(lambda x: 0.0 if x < 0 else x, tpms))
    return tpms


def generate_abundances(genome_count, type=1):
    if type == 0:
        return numpy.arange(10 / (genome_count + 1), 10, 10 / (genome_count + 1))

    if type == 1:
        a = -1239
        b = 942
        c = 1.5
        d = 0.4
    elif type == 2:
        a = -54798
        b = 63
        c = 6
        d = 0.00005
    elif type == 3:
        a = -56
        b = 29
        c = 1
        d = 0.16
    elif type == 4:
        a = -0.2
        b = 0.01
        c = 10
        d = 0.13
    else:
        a = -50
        b = 30
        c = 1
        d = 0.15

    x = numpy.array([random.random() * 10 for i in range(genome_count + 1)])

    coverage_list = []
    for i in x:
        coverage = a + b * numpy.exp(c / i ** d)
        coverage_list.append(coverage)

    return coverage_list
