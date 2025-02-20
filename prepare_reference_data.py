#!/usr/bin/env python3
#
# ############################################################################
# Copyright (c) 2020 LRGASP consortium
# Prepares reference data in needed format and infers artifical novel isoforms
# ############################################################################

import sys
import os
from traceback import print_exc
import logging
import shutil
import random
import gffutils
from Bio import SeqIO
import argparse
from collections import defaultdict
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger('LRGASP')

POLYA_LEN = 100


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--output", "-o", help="output prefix")
    parser.add_argument("--reference_annotation", "-a", help="reference annotation (GTF/.db)", type=str)
    parser.add_argument("--reference_transcripts", "-t", help="reference transcripts in FASTA format", type=str)
    parser.add_argument("--reference_genome", "-g", help="reference genome in FASTA format", type=str)
    parser.add_argument("--sqanti_prefix", "-q", help="path to SQANTI output "
                                                      "(_classification.txt and _corrected.gtf are needed)", type=str)
    parser.add_argument("--n_random_isoforms", "-n", help="insert this number of random novel isoforms into the annotation",
                        type=int)
    parser.add_argument("--isoform_list", "-l", help="insert only novel isoforms from a given file", type=str)
    parser.add_argument("--seed", "-s", help="randomizer seed [11]", default=11, type=int)

    parser.add_argument("--no_polya", help="do not insert poly-A tails", action="store_true", default=False)

    args = parser.parse_args(args, namespace)

    if not check_params(args):
        parser.print_usage()
        exit(-1)
    args.polya = not args.no_polya
    return args


def check_params(args):
    if args.n_random_isoforms is not None and args.isoform_list is not None:
        logger.warning("Both --n_random_isoforms and --isoform_list are provided, only ones from the list will be used")
    return args.reference_annotation is not None and args.reference_transcripts is not None


def replace_gene_isoform_id(l, new_gene_id, new_transcript_id):
    gene_id_pos = l.find("gene_id")
    if gene_id_pos != -1:
        end_pos = l.find(";", gene_id_pos)
        l = l[:gene_id_pos + len("gene_id")] + ' "' + new_gene_id + '"' + l[end_pos:]

    t_id_pos = l.find("transcript_id")
    if t_id_pos != -1:
        end_pos = l.find(";", t_id_pos)
        l = l[:t_id_pos + len("transcript_id")] + ' "' + new_transcript_id + '"' + l[end_pos:]

    return l


def modify_isofrom_id(t_id):
    return t_id.replace("_", "")


def select_sqanti_isoforms(args):
    total_transcripts = 0
    # isofrom_id -> gene_ind
    novel_isoforms = {}
    logger.info("Loading SQANTI output")
    for l in open(args.sqanti_prefix + "_classification.txt"):
        total_transcripts += 1
        tokens = l.strip().split()
        isoform_id = tokens[0]
        isoform_type = tokens[5]
        gene_id = tokens[6]
        is_canonical = tokens[16] == 'canonical'

        if is_canonical and isoform_type in ['novel_not_in_catalog', 'novel_in_catalog']:
            novel_isoforms[isoform_id] = gene_id
    logger.info("Total isoforms read: %d, nic and nnic selected: %d" % (total_transcripts-1, len(novel_isoforms)))

    # selecting isoforms
    logger.info("Selecting novel isoforms")
    if args.isoform_list is not None:
        selected_isoform_set = set()
        for l in open(args.isoform_list):
            tokens = l.strip().split()
            if l.startswith("#") or not tokens:
                continue
            isoform_id = tokens[0]
            if isoform_id in novel_isoforms:
                selected_isoform_set.add(isoform_id)
            else:
                logger.warning("Specified isoform id %s is not found among canonical novel isoforms in SQANTI output" % isoform_id)
    elif args.n_random_isoforms is not None and args.n_random_isoforms < len(novel_isoforms):
        selected_isoform_set = set()
        all_novel_isoforms = list(novel_isoforms.keys())
        while len(selected_isoform_set) < args.n_random_isoforms:
            isoform_to_add = random.choice(all_novel_isoforms)
            selected_isoform_set.add(isoform_to_add)
    else:
        selected_isoform_set = set(novel_isoforms.keys())
    logger.info("Total isoform selected: %d" % len(selected_isoform_set))

    novel_isoform_file = args.output + ".novel_isoforms.tsv"
    novel_isoform_handler = open(novel_isoform_file, "w")
    for t_id in sorted(selected_isoform_set):
        novel_isoform_handler.write(modify_isofrom_id(t_id) + "\n")
    novel_isoform_handler.close()

    # gene_id -> [(isoform_id, GTF lines)]
    novel_annotation = defaultdict(list)
    logger.info("Loading SQANTI annotation file")
    current_transcript_entry = ""
    current_transcrtip_id = ""

    for l in open(args.sqanti_prefix + "_corrected.gtf"):
        tokens = l.strip().split()
        enrty_type = tokens[2]

        if enrty_type == 'transcript':
            if current_transcrtip_id:
                # previous transcript was selected, adding to the annotation and clearing current entries
                gene_id = novel_isoforms[current_transcrtip_id]
                novel_annotation[gene_id].append((current_transcrtip_id, current_transcript_entry))
                current_transcript_entry = ""
                current_transcrtip_id = ""

            isoform_id = tokens[9].replace('"','').replace(';','')
            if isoform_id in selected_isoform_set:
                current_transcript_entry = replace_gene_isoform_id(l, novel_isoforms[isoform_id], modify_isofrom_id(isoform_id))
                current_transcrtip_id = isoform_id

        elif enrty_type == 'exon' and current_transcrtip_id:
            current_transcript_entry += replace_gene_isoform_id(l, novel_isoforms[current_transcrtip_id], modify_isofrom_id(current_transcrtip_id))

    if current_transcrtip_id:
        # last transcript was selected, adding to the annotation
        gene_id = novel_isoforms[current_transcrtip_id]
        novel_annotation[gene_id].append((current_transcrtip_id, current_transcript_entry))

    return novel_annotation


def load_gene_db(args):
    if args.reference_annotation.endswith(".db"):
        logger.info("Loading annotation from database")
        return gffutils.FeatureDB(args.reference_annotation, keep_order=True)

    logger.info("Converting gene annotation file to .db format (takes a while)...")
    db_file = args.output + ".reference_annotation.db"
    gffutils.create_db(args.reference_annotation, db_file, force=True, keep_order=True, merge_strategy='merge',
                       sort_attribute_values=True, disable_infer_transcripts=True, disable_infer_genes=True)
    logger.info("Gene database written to " + db_file)
    logger.info("Provide this database next time to avoid excessive conversion")

    logger.info("Loading annotation from database")
    return gffutils.FeatureDB(db_file, keep_order=True)


def insert_novel_transcripts_to_gtf(args, genedb, novel_annotation):
    extended_annotation_file = args.output + ".annotation.gtf"
    logger.info("Saving extended gene annotation (takes a while)...")
    added_transcripts = set()
    with open(extended_annotation_file, "w") as f:
        for record in genedb.all_features():
            f.write(str(record) + '\n')
            if record.featuretype == 'gene' and record.id in novel_annotation:
                # add novel isoforms
                for t in novel_annotation[record.id]:
                    logger.debug("Appending %s to the annotation" % t[0])
                    added_transcripts.add(t[0])
                    f.write(t[1])
    logger.info("Extended annotation saved to %s" % extended_annotation_file)
    for g in novel_annotation:
        for t in novel_annotation[g]:
            if t[0] not in added_transcripts:
                logger.warning("Gene %s, transcripts %s are missing" % (g, t[0]))


def parse_transcript(gtf_fragment):
    lines = gtf_fragment.split('\n')
    tokens = lines[0].split()
    assert tokens[2] == 'transcript'

    chr_id = tokens[0]
    strand = tokens[6]

    exons = []
    for l in lines[1:]:
        if not l:
            continue
        tokens = l.split()
        exons.append((int(tokens[3]), int(tokens[4])))

    return chr_id, strand, sorted(exons)


def extract_transcript_from_fasta(chr_seq, strand, exons, polya=True):
    transcript = ""
    for e in exons:
        transcript += str(chr_seq[e[0]-1:e[1]].seq)

    if strand == '-':
        transcript = str(Seq(transcript).reverse_complement()).upper()

    if polya:
        transcript += "A" * POLYA_LEN
    return transcript


def insert_novel_transcripts_to_fasta(args, novel_annotation):
    logger.info("Loading reference genome from " + args.reference_genome)
    genome_records = SeqIO.index(args.reference_genome, "fasta")
    # chrid -> [(t_id, strand, exons)]
    genomic_regions = defaultdict(list)
    for gene_id in novel_annotation:
        for t in novel_annotation[gene_id]:
            chr_id, strand, exons = parse_transcript(t[1])
            genomic_regions[chr_id].append((t[0], strand, exons))

    logger.info("Extracting transcript sequences")
    new_transcripts = []
    for chr_id in genomic_regions:
        chr_seq = genome_records[chr_id]
        for transcript_tuple in genomic_regions[chr_id]:
            transcript_seq = extract_transcript_from_fasta(chr_seq, transcript_tuple[1], transcript_tuple[2], args.polya)
            record = SeqRecord(Seq(transcript_seq), id=modify_isofrom_id(transcript_tuple[0]), description="novel")
            new_transcripts.append(record)

    for record in SeqIO.parse(args.reference_transcripts, "fasta"):
        if args.polya:
            record.seq = Seq(str(record.seq) + "A" * POLYA_LEN)
        else:
            record.seq = Seq(str(record.seq))
        new_transcripts.append(record)

    SeqIO.write(new_transcripts, args.output + ".transcripts.fasta", 'fasta')
    logger.info("Extended transcript sequences save to %s" % (args.output + ".transcripts.fasta")) 


def set_logger(logger_instance):
    logger_instance.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger_instance.addHandler(ch)


def run_pipeline(args):
    logger.info(" === LRGASP reference data preparation started === ")
    random.seed(args.seed)
    if args.sqanti_prefix is not None:
        novel_annotation = select_sqanti_isoforms(args)
    else:
        novel_annotation = defaultdict(list)
    gene_db = load_gene_db(args)
    insert_novel_transcripts_to_gtf(args, gene_db, novel_annotation)
    insert_novel_transcripts_to_fasta(args, novel_annotation)
    shutil.copyfile(args.reference_genome, args.output + ".genome.fasta")
    logger.info("Reference genomes save to %s" % (args.output + ".genome.fasta")) 
    logger.info(" === LRGASP reference data preparation finished === ")


def main(args):
    args = parse_args(args)
    set_logger(logger)
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    run_pipeline(args)


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    try:
        main(sys.argv[1:])
    except SystemExit:
        raise
    except:
        print_exc()
        sys.exit(-1)
