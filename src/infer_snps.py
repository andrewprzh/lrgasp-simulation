#!/usr/bin/env python3
import copy
import os.path

from Bio import SeqIO
import numpy as np
import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import vcf
from time import perf_counter
import sys
import logging
import copy


logger = logging.getLogger('LRGASP')


VCF_HEADER = '##fileformat=VCFv4.2\n' \
             '##FILTER=<ID=PASS,Description="All filters passed">\n' \
             '##FILTER=<ID=LowQual,Description="Low quality variant">\n' \
             '##FILTER=<ID=RefCall,Description="Reference call">\n' \
             '##INFO=<ID=P,Number=0,Type=Flag,Description="Result from pileup calling">\n' \
             '##INFO=<ID=F,Number=0,Type=Flag,Description="Result from full-alignment calling">\n' \
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n' \
             '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n' \
             '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n' \
             '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Read depth for each allele">\n' \
             '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled genotype likelihoods rounded to the closest integer">\n' \
             '##FORMAT=<ID=AF,Number=1,Type=Float,Description="Estimated allele frequency in the range of [0,1]">"'


NUCL_ALPHABET = {'A', 'T', 'G', 'C'}


def insert_mutations(args, genome_file, out_fasta):
    # start = perf_counter()
    logger.info("Inserting mutations into " + genome_file)
    cont_dic = {}
    header_vcf_fname = args.output + '.header.vcf'
    header_vcf = open(header_vcf_fname, 'w')
    last_line = '#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE'
    for rec in SeqIO.parse(genome_file, 'fasta'):
        cont_dic[rec.id] = len(rec.seq)
    # print(cont_dic)

    header_vcf.write(VCF_HEADER)
    for key in cont_dic:
        header_vcf.write('##contig=<ID={},length={}>'.format(key, cont_dic[key]) + '\n')
    header_vcf.write(last_line + '\n')
    header_vcf.close()
    stream = open(args.output + '.inudced_snps.vcf', 'w')
    temp = vcf.Reader(open(header_vcf_fname))
    writer = vcf.Writer(stream, template=temp)

    records = []
    n_cont = 0
    snp_count = 0
    mut_rate = args.mutation_rate
    for rec in SeqIO.parse(genome_file, 'fasta'):
        logger.info("Processing chromosome " + rec.id)
        n_mut = round(len(rec.seq) * mut_rate)
        n_cont += 1
        positions = np.random.randint(0, len(rec.seq), size=n_mut)
        positions = np.sort(np.array(list(set(positions))))
        input_dna = rec.seq
        contig = rec.id
        descr = rec.description
        dna_arr = np.array(list(input_dna))

        for i in positions:
            if dna_arr[i] not in NUCL_ALPHABET:
                continue
            snp_count += 1
            nucls = copy.copy(NUCL_ALPHABET)
            nucls.remove(dna_arr[i])
            subs = list(nucls)[random.randint(0, 2)]
            alt_rec = vcf.model._Substitution(subs)
            record = vcf.model._Record(contig, i, '.', dna_arr[i], [alt_rec],
                                       100, 'PASS', 0, 'GT:GQ:DP:AF', '0/1:6:4:0.2500')
            writer.write_record(record)
            dna_arr[i] = subs

        dna_str = ''.join(list(dna_arr))
        # print(records)
        seq_rec = SeqRecord(Seq(dna_str), id=contig, name=contig, description=descr + ' synthetic mutant')
        # print(seq_rec)
        records.append(seq_rec)
        print('Contig {} has been processed'.format(contig))
        # print(records)

    writer.close()
    SeqIO.write(records, out_fasta, "fasta")
    end_time = perf_counter()
    logger.info('Mutated %d chromosomes at %.5f rate, total mutations inserted: %d' % (n_cont, args.mutation_rate, snp_count))
    logger.info('Written to ' + out_fasta)