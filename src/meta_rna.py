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


logger = logging.getLogger('LRGASP')


def combine_references(reference_list, output_prefix):  # in 'data' insert your path to file with genomes and annotations
    df = pd.read_csv(reference_list, sep='\t', header=None)
    df.columns = ['genome', 'annotation']
    genome_id_list_without_duplicates = set()
    new_id_map = {}

    genomes_path = output_prefix + '.genomes.fasta'
    annotations_path = output_prefix + '.annotations.gtf'
    transcripts_path = output_prefix + '.transcripts.fasta'
    logger.info("Combining annotations and genomes")
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
            for annotation_rec in GFF.parse(annotation_path):
                annotation_id_list.append(annotation_rec.id)
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


def extract_transcripts(genome_path, annotation_path, transcripts_path):
    logger.info("Extracting transcripts")
    res = subprocess.run(['gffread', '-w', transcripts_path, '-g', genome_path, annotation_path])
    if res.returncode != 0:
        logger.error("gffread failed, contact developers for support.")
        return


def generate_meta_counts(reference_list, output_prefix):
    df = pd.read_csv(reference_list, sep='\t', header=None)
    df.columns = ['genome', 'annotation']


def generate_gene_counts():
    pass


def generate_abundances():
    pass