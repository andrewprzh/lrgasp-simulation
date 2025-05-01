#!/usr/bin/env python3
"""
@author: Chen Yang & Saber HafezQorani
This script generates simulated Oxford Nanopore 2D reads (genomic and transcriptomic - cDNA/directRNA).
"""
# NOTE: this file was modified by the LRGASP consortium comparing to the original source code

from __future__ import print_function
from __future__ import with_statement

import multiprocessing as mp
from subprocess import call
from textwrap import dedent
import sys
import os
import HTSeq
import pysam
import random
import re
import copy
import argparse
import joblib
from time import strftime
from urllib.request import Request, urlopen
from gzip import GzipFile
import numpy as np
import scipy.stats

if sys.version_info[0] < 3:
    from string import maketrans
    trantab = maketrans("T", "U")
else:
    trantab = str.maketrans("T", "U")

try:
    from six.moves import xrange
except ImportError:
    pass
import mixed_model as mm
import norm_distr as nd
import math
from Bio import SeqIO

PYTHON_VERSION = sys.version_info
VERSION = "3.0.0"
PRORAM = "NanoSim"
AUTHOR = "Chen Yang, Saber Hafezqorani, Theodora Lo (UBC & BCGSC)"
CONTACT = "cheny@bcgsc.ca; shafezqorani@bcgsc.ca"

BASES = ['A', 'T', 'C', 'G']


def select_ref_transcript(input_dict):
    length = 0
    #print(input_dict)
    while True:
        p = random.random()
        for key, val in input_dict.items():
            if key[0] <= p < key[1]:
                length = val[1]
                break
        if length != 0:
            break
    return val[0], length


def select_ref_transcript_fast(cumulative_tpm, sorted_ids):
    p = random.random()
    s = len(cumulative_tpm) - 1
    ind = s // 2
    current_step = s // 2
    while not (p >= cumulative_tpm[ind] and p < cumulative_tpm[ind+1]):
        current_step = max(1, current_step // 2)
        if p < cumulative_tpm[ind]:
            ind -= current_step
        else:
            ind += current_step
    return sorted_ids[ind]


def list_to_range(input_list, min_l):
    l = [min_l]
    l.extend(input_list)
    output_list = []
    for i in xrange(0, len(l) - 1):
        r = (l[i], l[i + 1])
        output_list.append(r)
    return output_list


def make_cdf(dict_exp, dict_len):
    sum_exp = 0
    list_value = []
    for item in dict_exp:
        if item in dict_len:
            sum_exp += dict_exp[item]
    for item in dict_exp:
        if item in dict_len:
            value = dict_exp[item] / float(sum_exp)
            list_value.append((item, value))

    sorted_value_list = sorted(list_value, key=lambda x: x[1])
    sorted_only_values = [x[1] for x in sorted_value_list]
    list_cdf = np.cumsum(sorted_only_values)
    ranged_cdf_list = list_to_range(list_cdf, 0)

    ecdf_dict = {}
    for i in xrange(len(ranged_cdf_list)):
        cdf_range = ranged_cdf_list[i]
        ecdf_dict[cdf_range] = (sorted_value_list[i][0], dict_len[sorted_value_list[i][0]])
        # ecdf_dict[cdf_range] = dict_len[sorted_value_list[i][0]]

    return ecdf_dict


def make_cdf_fast(dict_exp, dict_len):
    sum_exp = 0
    list_value = []
    for item in dict_exp:
        if item in dict_len:
            sum_exp += dict_exp[item]
    current_val = 0
    for item in dict_exp:
        if item in dict_len:
            list_value.append((item, current_val))
            current_val = dict_exp[item] / float(sum_exp)

    sorted_value_list = sorted(list_value, key=lambda x: x[1])
    sorted_only_values = [x[1] for x in sorted_value_list]
    list_cdf = list(np.cumsum(sorted_only_values))
    list_cdf.append(1.00001)
    sorted_ids = [x[0] for x in sorted_value_list] + [""]
    return list_cdf, sorted_ids


def ref_len_from_structure(input):
    l = 0
    for item in input:
        if item[0] == "exon":
            l += item[-2]
    return l


def select_nearest_kde2d(sampled_2d_lengths, ref_len_total):
    fc = sampled_2d_lengths[:, 0]
    idx = np.abs(fc - ref_len_total).argmin()
    return int(sampled_2d_lengths[idx][1])


def update_structure(ref_trx_structure, IR_markov_model):
    count = 0
    for item in ref_trx_structure:
        if item[0] == "intron":
            count += 1

    list_states = []
    flag_ir = False
    prev_state = "start"
    for i in range(0, count):
        p = random.random()
        for key in IR_markov_model[prev_state]:
            if key[0] <= p < key[1]:
                flag = IR_markov_model[prev_state][key]
                if flag == "IR":
                    flag_ir = True
                list_states.append(flag)
                prev_state = flag
                break

    if flag_ir:
        ref_trx_structure_temp = copy.deepcopy(ref_trx_structure)
        j = -1
        for i in xrange(0, len(ref_trx_structure_temp)):
            if ref_trx_structure_temp[i][0] == "intron":
                j += 1
                if list_states[j] == "IR":
                    ref_trx_structure_temp[i] = ("retained_intron",) + ref_trx_structure_temp[i][1:]
    else:
        ref_trx_structure_temp = ref_trx_structure

    return flag_ir, ref_trx_structure_temp


def extract_read_pos(length, ref_len, ref_trx_structure, polya, buffer=10):
    # The aim is to create a genomic interval object
    # example: iv = HTSeq.GenomicInterval( "chr3", 123203, 127245, "+" )
    # buffer: if the extracted read is within 10 base to the reference 3' end, it's considered as reaching to the end

    # find the length before the first retained intron
    len_before = 0
    for item in ref_trx_structure:
        if item[0] == "exon":
            len_before += item[4]
        elif item[0] == "retained_intron":
            break

    # TODO change the random into something truer
    start_pos = random.randint(0, min(ref_len - length, len_before))  # make sure the retained_intron is included

    list_intervals = []
    ir_list = []
    for item in ref_trx_structure:
        if length == 0:
            break
        chrom = item[1]
        if item[0] in ["exon", "retained_intron"]:
            if start_pos < item[4]:
                start = start_pos + item[2]
                if start + length <= item[3]:
                    end = start + length
                else:
                    end = item[3]
                length -= end - start
                start_pos = 0
                iv = HTSeq.GenomicInterval(chrom, start, end, item[5])
                list_intervals.append(iv)
                if item[0] == "retained_intron":
                    ir_list.append((start, end))
            else:
                start_pos -= item[4]

    if polya and end + buffer >= ref_trx_structure[-1][3]:
        retain_polya = True
    else:
        retain_polya = False

    return list_intervals, retain_polya, ir_list


def read_ecdf(profile):
    # We need to count the number of zeros. If it's over 10 zeros, l_len/l_ratio need to be changed to higher.
    # Because it's almost impossible that the ratio is much lower than the lowest historical value.
    header = profile.readline()
    header_info = header.strip().split()
    ecdf_dict = {}
    lanes = len(header_info[1:])

    for i in header_info[1:]:
        boundaries = i.split('-')
        ecdf_dict[(int(boundaries[0])), int(boundaries[1])] = {}

    ecdf_key = sorted(ecdf_dict.keys())
    l_prob = [0.0] * lanes
    l_ratio = [0.0] * lanes

    for line in profile:
        new = line.strip().split('\t')
        ratio = [float(x) for x in new[0].split('-')]
        prob = [float(x) for x in new[1:]]
        for i in xrange(lanes):
            if prob[i] == l_prob[i]:
                continue
            else:
                if l_prob[i] != 0:
                    ecdf_dict[ecdf_key[i]][(l_prob[i], prob[i])] = (l_ratio[i], ratio[1])
                else:
                    ecdf_dict[ecdf_key[i]][(l_prob[i], prob[i])] \
                        = (max(l_ratio[i], ratio[1] - 10 * (ratio[1] - ratio[0])), ratio[1])
                l_ratio[i] = ratio[1]
                l_prob[i] = prob[i]

    for i in xrange(0, len(ecdf_key)):
        last_key = sorted(ecdf_dict[ecdf_key[i]].keys())[-1]
        last_value = ecdf_dict[ecdf_key[i]][last_key]
        ecdf_dict[ecdf_key[i]][last_key] = (last_value[0], ratio[1])

    return ecdf_dict


def get_length_kde(kde, num, log=False, flatten=True):
    tmp_list = kde.sample(num)
    if log:
        tmp_list = np.power(10, tmp_list) - 1
    length_list = tmp_list.flatten()
    if not flatten:
        return tmp_list
    else:
        return length_list


def read_profile(ref_g, number_list, model_prefix, per, mode, strandness, ref_t=None, dna_type=None, abun=None,
                 polya=None, exp=None, model_ir=False, chimeric=False, aligned_only=False):
    # Note var number_list (list) used to be number (int)
    global number_aligned_l, number_unaligned_l, number_segment_list
    global match_ht_list, error_par, trans_error_pr, match_markov_model
    global kde_aligned, kde_ht, kde_ht_ratio, kde_unaligned, kde_aligned_2d
    global seq_dict, seq_len, max_chrom
    global strandness_rate

    if mode == "genome":
        global genome_len
        ref = ref_g
    elif mode == "metagenome":
        global multi_dict_abun, dict_dna_type
        ref = {}
        with open(ref_g, 'r') as genome_list:
            list = genome_list.readlines()
            for genome in list:
                fields = genome.split("\t")
                species = fields[0]
                genome = fields[1].strip("\n")
                ref[species] = genome
    else:
        global dict_exp, list_cdf, sorted_ids ,seq_len
        ref = ref_t

    if strandness is None:
        with open(model_prefix + "_strandness_rate", 'r') as strand_profile:
            strandness_rate = float(strand_profile.readline().split("\t")[1])
    else:
        strandness_rate = strandness

    sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read in reference \n")
    sys.stdout.flush()
    seq_dict = {}
    seq_len = {}
    dict_dna_type = {}

    # Read in the reference genome/transcriptome/metagenome
    if mode == "metagenome":
        max_chrom = {}
        for species in ref.keys():
            fq_path = ref[species]
            seq_dict[species] = {}
            seq_len[species] = {}
            dict_dna_type[species] = {}
            max_chrom[species] = 0

            if fq_path.startswith(("ftp", "http")):
                http_addr = fq_path.replace("ftp://", "http://")
                dir_name = fq_path.strip().split('/')[-1]
                http_complete = http_addr.strip() + '/' + dir_name + '_genomic.fna.gz'
                url_req = Request(http_complete)
                url_req.add_header('Accept-Encoding', 'gzip')
                response = urlopen(url_req)
                with GzipFile(fileobj=response) as f:
                    for line in f:
                        line = str(line, 'utf-8').strip()
                        if line[0] == '>':
                            info = re.split(r'[_\s]\s*', line)
                            chr_name = "-".join(info[1:])
                            seq_dict[species][chr_name.split(".")[0]] = ''
                            seq_len[species][chr_name.split(".")[0]] = 0
                            dict_dna_type[species][chr_name.split(".")[0]] = "linear"  # linear as default
                        else:
                            seq_dict[species][chr_name.split(".")[0]] += line
                            seq_len[species][chr_name.split(".")[0]] += len(line)
                            if seq_len[species][chr_name.split(".")[0]] > max_chrom[species]:
                                max_chrom[species] = seq_len[species][chr_name.split(".")[0]]
            else:
                for record in SeqIO.parse(fq_path, 'fasta'):
                    seqS = str(record.seq)
                    seq_dict[species][record.id] = seqS
                    seq_len[species][record.id] = len(seqS)
                    dict_dna_type[species][record.id] = "linear"  # linear as default
                    if len(seqS) > max_chrom[species]:
                        max_chrom[species] = len(seqS)

        with open(dna_type, 'r') as dna_type_list:
            for line in dna_type_list.readlines():
                fields = line.split("\t")
                species = fields[0]
                chr_name = fields[1].partition(" ")[0]
                type = fields[2].strip("\n")

                if species not in ref:
                    sys.stderr.write("You didn't provide a reference genome for " + species + '\n')
                    sys.exit(1)
                dict_dna_type[species][chr_name.split(".")[0]] = type
    else:
        max_chrom = 0
        for record in SeqIO.parse(ref, 'fasta'):
            seqS = str(record.seq)
            seq_dict[record.id] = seqS
            seq_len[record.id] = len(seqS)
            if len(seqS) > max_chrom:
                max_chrom = len(seqS)

    # Special files for each mode
    if mode == "genome":
        genome_len = sum(seq_len.values())
        if len(seq_dict) > 1 and dna_type == "circular":
            sys.stderr.write("Do not choose circular if there is more than one chromosome in the genome!\n")
            sys.exit(1)
    elif mode == "metagenome":
        sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read in abundance profile\n")
        sys.stdout.flush()
        with open(abun, 'r') as abun_file:
            header = abun_file.readline()
            number_list = [int(x) for x in header.strip().split('\t')[1:]]
            sample_size = len(number_list)
            samples = ["sample" + str(x) for x in range(sample_size)]
            multi_dict_abun = {sample: {} for sample in samples}
            for line in abun_file.readlines():
                fields = line.split("\t")
                if sample_size != len(fields) - 1:  # abundance file is incorrectly formatted
                    sys.stderr.write("Abundance file is incorrectly formatted. Check that each row has the same number "
                                     "of columns\n")
                    sys.exit(1)

                species = fields[0]
                if species not in ref:
                    sys.stderr.write("You didn't provide a reference genome for " + species + '\n')
                    sys.exit(1)

                expected = [float(x) for x in fields[1:]]
                for s_idx in xrange(sample_size):
                    multi_dict_abun[samples[s_idx]][species] = expected[s_idx]

    else:
        sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read in expression profile\n")
        sys.stdout.flush()
        dict_exp = {}
        with open(exp, 'r') as exp_file:
            header = exp_file.readline()
            for line in exp_file:
                parts = line.split("\t")
                transcript_id = parts[0]
                tpm = float(parts[2])
                if  tpm > 0:
                    dict_exp[transcript_id] = tpm
        # create the ecdf dict considering the expression profiles
        #print(dict_exp)
        #print(seq_len)
        list_cdf, sorted_ids = make_cdf_fast(dict_exp, seq_len)
        assert len(list_cdf) == len(sorted_ids)
        assert list_cdf[0] == 0
        assert list_cdf[-1] >= 1.0

        if model_ir:
            global genome_fai, IR_markov_model, dict_ref_structure
            sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read in reference genome and create .fai index file\n")
            sys.stdout.flush()
            # create and read the .fai file of the reference genome
            genome_fai = pysam.Fastafile(ref_g)

            sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read in IR markov model\n")
            sys.stdout.flush()

            IR_markov_model = {}
            with open(model_prefix + "_IR_markov_model", "r") as IR_markov:
                IR_markov.readline()
                for line in IR_markov:
                    info = line.strip().split()
                    k = info[0]
                    IR_markov_model[k] = {}
                    IR_markov_model[k][(0, float(info[1]))] = "no_IR"
                    IR_markov_model[k][(float(info[1]), float(info[1]) + float(info[2]))] = "IR"

            sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read in GFF3 annotation file\n")
            sys.stdout.flush()
            dict_ref_structure = {}
            gff_file = model_prefix + "_added_intron_final.gff3"
            gff_features = HTSeq.GFF_Reader(gff_file, end_included=True)
            for feature in gff_features:
                if feature.type == "exon" or feature.type == "intron":
                    if "transcript_id" in feature.attr:
                        feature_id = feature.attr['transcript_id']
                    elif "Parent" in feature.attr:
                        info = feature.name.split(":")
                        if len(info) == 1:
                            feature_id = info[0]
                        else:
                            if info[0] == "transcript":
                                feature_id = info[1]
                            else:
                                continue
                    else:
                        continue

                    feature_id = feature_id.split(".")[0]
                    if feature_id not in dict_ref_structure:
                        dict_ref_structure[feature_id] = []

                    # remove "chr" from chromosome names to be consistent
                    if "chr" in feature.iv.chrom:
                        feature.iv.chrom = feature.iv.chrom.strip("chr")

                    dict_ref_structure[feature_id].append((feature.type, feature.iv.chrom, feature.iv.start,
                                                           feature.iv.end, feature.iv.length, feature.iv.strand))

        if polya:
            global trx_with_polya
            sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read in list of transcripts with polyA tails\n")
            sys.stdout.flush()
            trx_with_polya = {}
            with open(polya, "r") as trx_list:
                for line in trx_list.readlines():
                    transcript_id = line.strip().split("\t")[0]
                    trx_with_polya[transcript_id] = 0

    if per:  # if parameter perfect is used, all reads should be aligned, number_aligned equals total number of reads
        number_aligned_l = number_list
        number_unaligned_l = [0] * len(number_list)
    else:
        # Read model profile for match, mismatch, insertion and deletions
        sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read error profile\n")
        sys.stdout.flush()

        error_par = {}
        model_profile = model_prefix + "_model_profile"
        with open(model_profile, 'r') as mod_profile:
            mod_profile.readline()
            for line in mod_profile:
                new_line = line.strip().split("\t")
                if "mismatch" in line:
                    error_par["mis"] = [float(x) for x in new_line[1:]]
                elif "insertion" in line:
                    error_par["ins"] = [float(x) for x in new_line[1:]]
                else:
                    error_par["del"] = [float(x) for x in new_line[1:]]

        trans_error_pr = {}
        with open(model_prefix + "_error_markov_model", "r") as error_markov:
            error_markov.readline()
            for line in error_markov:
                info = line.strip().split()
                k = info[0]
                trans_error_pr[k] = {}
                trans_error_pr[k][(0, float(info[1]))] = "mis"
                trans_error_pr[k][(float(info[1]), float(info[1]) + float(info[2]))] = "ins"
                trans_error_pr[k][(1 - float(info[3]), 1)] = "del"

        with open(model_prefix + "_first_match.hist", 'r') as fm_profile:
            match_ht_list = read_ecdf(fm_profile)

        with open(model_prefix + "_match_markov_model", 'r') as mm_profile:
            match_markov_model = read_ecdf(mm_profile)

        # Read length of unaligned reads
        sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read KDF of unaligned reads\n")
        sys.stdout.flush()

        with open(model_prefix + "_reads_alignment_rate", 'r') as u_profile:
            new = u_profile.readline().strip()
            rate = new.split('\t')[1]
            if rate == "100%" or aligned_only:
                number_aligned_l = number_list
            else:
                number_aligned_l = [int(round(x * float(rate) / (float(rate) + 1))) for x in number_list]
            number_unaligned_l = [x - y for x, y in zip(number_list, number_aligned_l)]

        if min(number_unaligned_l) > 0:
            kde_unaligned = joblib.load(model_prefix + "_unaligned_length.pkl")

    # Read profile of aligned reads
    sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read KDF of aligned reads\n")
    sys.stdout.flush()

    # Read ht profile
    kde_ht = joblib.load(model_prefix + "_ht_length.pkl")

    # Read head/unaligned region ratio
    kde_ht_ratio = joblib.load(model_prefix + "_ht_ratio.pkl")

    # Read length of aligned reads
    # If "perfect" is chosen, just use the total length ecdf profile, else use the length of aligned region on reference
    if per:
        kde_aligned = joblib.load(model_prefix + "_aligned_reads.pkl")
        if mode == "transcriptome":
            kde_aligned_2d = joblib.load(model_prefix + "_aligned_region_2d.pkl")
    else:
        if mode == "transcriptome":
            kde_aligned_2d = joblib.load(model_prefix + "_aligned_region_2d.pkl")
        else:  # genome/metagenome
            kde_aligned = joblib.load(model_prefix + "_aligned_region.pkl")

    # Read chimeric reads information
    sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Read chimeric simulation information\n")
    if chimeric:
        global abun_inflation, kde_gap, segment_mean
        with open(model_prefix + "_chimeric_info") as chimeric_info:
            segment_mean = float(chimeric_info.readline().split('\t')[1])
            if mode == "metagenome":
                abun_inflation = float(chimeric_info.readline().split('\t')[1])
        kde_gap = joblib.load(model_prefix + "_gap_length.pkl")


def add_abundance_var(expected_abun, total_len, var_low, var_high):
    # Order species according to genome size and assign highest variation to species
    # w/ largest genome and lowest variation to species w/ smallest genome
    abun_var = []
    for i in range(len(total_len)):  # Generate % var for each species
        abun_var.append(random.uniform(var_low, var_high))

    abun_var_per_species_dict = {}
    for var, species in zip(sorted(abun_var, key=abs), sorted(total_len, key=lambda k: total_len[k])):
        abun_var_per_species_dict[species] = var

    # Add variation to expected abundances
    abun_with_var = {}
    for species, expected in expected_abun.items():
        abun_with_var[species] = expected + expected * abun_var_per_species_dict[species]

    # Renormalize abundances
    total = sum(abun_with_var.values())
    for species, abun in abun_with_var.items():
        abun_with_var[species] = abun * 100 / total

    return abun_with_var


def mutate_homo(seq, base_quals, k, basecaller, read_type):
    hp_arr = []  # [[base, start, end], ...]
    hp_length_hist = {}  # {length: {A/T: count, C/G: count} ...}
    hp_samples = {}  # {length: {A/T: [sample], C/G: [sample]} ...}

    # Finding homopolymers in sequence
    pattern = "A{" + re.escape(str(k)) + ",}|C{" + re.escape(str(k)) + ",}|G{" + re.escape(
        str(k)) + ",}|T{" + re.escape(str(k)) + ",}"

    for match in re.finditer(pattern, seq):
        hp_start = match.start()
        hp_end = match.end()
        length = hp_end - hp_start
        base = match.group()[0]
        hp_arr.append([base, hp_start, hp_end])
        if length not in hp_length_hist.keys():
            hp_length_hist[length] = {"A": 0, "T": 0, "C": 0, "G": 0}

        hp_length_hist[length][base] += 1

    # Obtaining samples from normal distributions
    for length in hp_length_hist.keys():
        hp_samples[length] = {}
        a_mu, a_sigma, t_mu, t_sigma, c_mu, c_sigma, g_mu, g_sigma = nd.get_nd_par(length, read_type, basecaller)

        if hp_length_hist[length]["A"] > 0:
            hp_samples[length]["A"] = np.random.normal(a_mu, a_sigma, hp_length_hist[length]["A"])
        if hp_length_hist[length]["T"] > 0:
            hp_samples[length]["T"] = np.random.normal(t_mu, t_sigma, hp_length_hist[length]["T"])
        if hp_length_hist[length]["C"] > 0:
            hp_samples[length]["C"] = np.random.normal(c_mu, c_sigma, hp_length_hist[length]["C"])
        if hp_length_hist[length]["G"] > 0:
            hp_samples[length]["G"] = np.random.normal(g_mu, g_sigma, hp_length_hist[length]["G"])

    for length in hp_samples.keys():
        for base in hp_samples[length].keys():
            hp_samples[length][base] = [0 if x < 0 else x for x in hp_samples[length][base]]

    # Mutating homopolymers in given sequence
    last_pos = 0
    mutated_seq = ""
    total_hp_size_change = 0
    mis_rate = nd.get_hpmis_rate(read_type, basecaller)
    for hp_info in hp_arr:
        base = hp_info[0]
        ref_hp_start = hp_info[1]
        ref_hp_end = hp_info[2]

        size = int(round(hp_samples[ref_hp_end - ref_hp_start][base][-1]))
        hp_samples[ref_hp_end - ref_hp_start][base] = hp_samples[ref_hp_end - ref_hp_start][base][:-1]

        mutated_hp_with_mis = ""
        mis_pos = []

        for i in xrange(size):
            p = random.random()
            if 0 < p <= mis_rate:
                tmp_bases = list(BASES)
                while True:
                    new_base = random.choice(tmp_bases)
                    if new_base != base:
                        break
                mutated_hp_with_mis += new_base
                mis_pos.append(i)
            else:
                mutated_hp_with_mis += base
            i += 1
        mutated_seq = mutated_seq + seq[last_pos: ref_hp_start] + mutated_hp_with_mis

        if len(base_quals) != 0:  # fastq
            diff = size - (ref_hp_end - ref_hp_start)
            if diff < 0:  # del, remove quals
                for i in xrange(abs(diff)):
                    base_quals.pop(ref_hp_start + total_hp_size_change)

            elif diff > 0:  # ins, add quals
                ins_quals = mm.trunc_lognorm_rvs("ins", read_type, basecaller, diff).tolist()
                base_quals = base_quals[:ref_hp_end + total_hp_size_change] + ins_quals + \
                             base_quals[ref_hp_end + total_hp_size_change:]

            if len(mis_pos) != 0:  # mis, change match quals to mis quals
                mis_quals = mm.trunc_lognorm_rvs("mis", read_type, basecaller, len(mis_pos)).tolist()
                for i in zip(mis_pos, mis_quals):
                    base_quals[ref_hp_start + total_hp_size_change + i[0]] = i[1]

        total_hp_size_change += size - (ref_hp_end - ref_hp_start)
        last_pos = ref_hp_end

    return mutated_seq + seq[last_pos:], base_quals


# Taken from https://github.com/lh3/readfq
def readfq(fp):  # this is a generator function
    last = None  # this is a buffer keeping the last unprocessed line
    while True:  # mimic closure; is it a bad idea?
        if not last:  # the first record or a record following a fastq
            for l in fp:  # search for the start of the next record
                if l[0] in '>@':  # fasta/q header line
                    last = l[:-1]  # save this line
                    break
        if not last:
            break
        name, seqs, last = last[1:].partition(" ")[0], [], None
        for l in fp:  # read the sequence
            if l[0] in '@+>':
                last = l[:-1]
                break
            seqs.append(l[:-1])
        if not last or last[0] != '+':  # this is a fasta record
            yield name, ''.join(seqs), None  # yield a fasta record
            if not last:
                break
        else:  # this is a fastq record
            seq, leng, seqs = ''.join(seqs), 0, []
            for l in fp:  # read the quality
                seqs.append(l[:-1])
                leng += len(l) - 1
                if leng >= len(seq):  # have read enough quality
                    last = None
                    yield name, seq, ''.join(seqs)  # yield a fastq record
                    break
            if last:  # reach EOF before reading enough quality
                yield name, seq, None  # yield a fasta record instead
                break


def case_convert(seq):
    base_code = {'Y': ['C', 'T'], 'R': ['A', 'G'], 'W': ['A', 'T'], 'S': ['G', 'C'], 'K': ['T', 'G'], 'M': ['C', 'A'],
                 'D': ['A', 'G', 'T'], 'V': ['A', 'C', 'G'], 'H': ['A', 'C', 'T'], 'B': ['C', 'G', 'T'],
                 'N': ['A', 'T', 'C', 'G'], 'X': ['A', 'T', 'C', 'G']}

    up_string = seq.upper()
    up_list = list(up_string)
    for i in xrange(len(up_list)):
        if up_list[i] in base_code:
            up_list[i] = random.choice(base_code[up_list[i]])
    out_seq = ''.join(up_list)

    return out_seq


def assign_species(length_list, seg_list, current_species_base_dict):
    # Deal with chimeric reads first
    seg_list_sorted = sorted(seg_list, reverse=True)
    segs_chimera = sum([x for x in seg_list if x > 1])

    # Sort lengths for non-chimeras to fit in species quota better
    length_list_nonchimera = length_list[segs_chimera:]
    length_list_sorted = length_list[:segs_chimera] + sorted(length_list_nonchimera, reverse=True)
    nonchimera_idx = sorted(range(len(length_list_nonchimera)), key=length_list_nonchimera.__getitem__, reverse=True)
    length_list_idx = list(range(segs_chimera)) + [x + segs_chimera for x in nonchimera_idx]

    species_list = [''] * len(length_list)
    bases_to_add = sum(length_list)
    current_bases = sum(current_species_base_dict.values())
    total_bases = bases_to_add + current_bases

    base_quota = {}
    total_abun = sum(dict_abun.values())
    for species, abun in dict_abun.items():
        base_quota[species] = total_bases * abun / total_abun - current_species_base_dict[species]

    length_list_pointer = 0
    pre_species = ''
    for seg in seg_list_sorted:
        for each_seg in range(seg):
            if each_seg == 0:
                available_species = [s for s, q in base_quota.items()
                                     if q - length_list_sorted[length_list_pointer] > 0]
                if len(available_species) == 0:
                    available_species = [s for s, q in base_quota.items() if q > 0]
                species = random.choice(available_species)
            else:
                available_species = [s for s, q in base_quota.items()
                                     if q - length_list_sorted[length_list_pointer] > 0 and s != pre_species]
                p = random.uniform(0, 100)
                if p <= dict_abun_inflated[pre_species] and base_quota[pre_species] > 0:
                    species = pre_species
                elif p > dict_abun_inflated[pre_species] and len(available_species) > 0:
                    species = random.choice(available_species)
                else:
                    available_species = [s for s, q in base_quota.items()
                                         if q - length_list_sorted[length_list_pointer] > 0]
                    if len(available_species) == 0:
                        available_species = [s for s, q in base_quota.items() if q > 0]
                    species = random.choice(available_species)

            species_list[length_list_idx[length_list_pointer]] = species
            base_quota[species] -= length_list_sorted[length_list_pointer]
            length_list_pointer += 1
            pre_species = species

    return species_list


def simulation_aligned_metagenome(min_l, max_l, median_l, sd_l, out_reads, out_error, kmer_bias, basecaller,
                                  read_type, fastq, num_simulate, per=False, chimeric=False):
    # Simulate aligned reads
    out_reads = open(out_reads, "w")
    out_error = open(out_error, "w")

    id_begin = '@' if fastq else '>'

    remaining_reads = num_simulate
    if chimeric:
        num_segment = np.random.geometric(1 / segment_mean, num_simulate)
    else:
        num_segment = np.ones(num_simulate, dtype=int)
    remaining_segments = num_segment
    remaining_gaps = remaining_segments - 1
    passed = 0
    current_species_bases = {species: 0 for species in dict_abun.keys()}
    while remaining_reads > 0:
        if per:
            ref_lengths = get_length_kde(kde_aligned, sum(remaining_segments)) if median_l is None else \
                np.random.lognormal(np.log(median_l), sd_l, remaining_segments)
            ref_lengths = [x for x in ref_lengths if min_l <= x <= max_l]
        else:
            remainder_lengths = get_length_kde(kde_ht, int(remaining_reads * 1.3), True)
            remainder_lengths = [x for x in remainder_lengths if x >= 0]
            head_vs_ht_ratio_list = get_length_kde(kde_ht_ratio, int(remaining_reads * 1.5))
            head_vs_ht_ratio_list = [x for x in head_vs_ht_ratio_list if 0 <= x <= 1]
            if median_l is None:
                ref_lengths = get_length_kde(kde_aligned, sum(remaining_segments))
            else:
                total_lengths = np.random.lognormal(np.log(median_l + sd_l ** 2 / 2), sd_l, remaining_reads)
                ref_lengths = total_lengths - remainder_lengths
            ref_lengths = [x for x in ref_lengths if 0 < x <= max_l]

        gap_lengths = get_length_kde(kde_gap, sum(remaining_gaps), True) if sum(remaining_gaps) > 0 else []
        gap_lengths = [max(0, int(x)) for x in gap_lengths]

        # Select strain/species to simulate
        species_pool = assign_species(ref_lengths, remaining_segments, current_species_bases)

        seg_pointer = 0
        gap_pointer = 0
        species_pointer = 0
        for each_read in xrange(remaining_reads):
            segments = remaining_segments[each_read]
            # In case too many ref length was filtered previously
            if (not per and each_read >= min(len(head_vs_ht_ratio_list), len(remainder_lengths))) or \
                seg_pointer + segments > len(ref_lengths):
                break
            ref_length_list = [int(round(ref_lengths[seg_pointer + x])) for x in range(segments)]
            gap_length_list = [int(round(gap_lengths[gap_pointer + x])) for x in range(segments - 1)]
            species_list = [species_pool[species_pointer + x] for x in range(segments)]

            if per:
                seg_pointer += 1
                gap_pointer += 1
                species_pointer += 1
                with total_simulated.get_lock():
                    sequence_index = total_simulated.value
                    total_simulated.value += 1

                # Extract middle region from reference genome
                new_read = ""
                new_read_name = ""
                base_quals = []
                for seg_idx in range(len(ref_length_list)):
                    new_seg, new_seg_name = extract_read("metagenome", ref_length_list[seg_idx], species_list[seg_idx])
                    new_read += new_seg
                    new_read_name += new_seg_name
                    if fastq:
                        base_quals.extend(mm.trunc_lognorm_rvs("match", read_type, basecaller,
                                                               ref_length_list[seg_idx]).tolist())

                new_read_name = new_read_name + "_perfect_" + str(sequence_index)
                read_mutated = case_convert(new_read)  # not mutated actually, just to be consistent with per == False

                head = 0
                tail = 0

            else:
                gap_list = []
                gap_base_qual_list = []
                seg_length_list = []
                seg_error_dict_list = []
                seg_error_count_list = []
                remainder = int(round(remainder_lengths[each_read]))
                head_vs_ht_ratio = head_vs_ht_ratio_list[each_read]

                total = remainder
                for each_gap in gap_length_list:
                    mutated_gap, gap_base_quals = simulation_gap(each_gap, basecaller, read_type, "metagenome", fastq)
                    gap_list.append(mutated_gap)
                    gap_base_qual_list.append(gap_base_quals)
                    total += len(mutated_gap)
                for each_ref in ref_length_list:
                    middle, middle_ref, error_dict, error_count = \
                        error_list(each_ref, match_markov_model, match_ht_list, error_par, trans_error_pr, fastq)
                    total += middle
                    seg_length_list.append(middle_ref)
                    seg_error_dict_list.append(error_dict)
                    seg_error_count_list.append(error_count)

                if total < min_l or total > max_l:
                    continue

                seg_pointer += segments
                gap_pointer += segments - 1
                species_pointer += segments

                with total_simulated.get_lock():
                    sequence_index = total_simulated.value
                    total_simulated.value += 1

                if remainder == 0:
                    head = 0
                    tail = 0
                else:
                    head = int(round(remainder * head_vs_ht_ratio))
                    tail = remainder - head

                # Extract middle region from reference genome
                read_mutated = ""
                new_read_name = ""
                base_quals = []
                for seg_idx in range(len(seg_length_list)):
                    new_seg, new_seg_name = extract_read("metagenome", seg_length_list[seg_idx], species_list[seg_idx])
                    # Mutate read
                    new_seg = case_convert(new_seg)
                    seg_mutated, seg_base_quals = \
                        mutate_read(new_seg, new_seg_name, out_error, seg_error_dict_list[seg_idx],
                                    seg_error_count_list[seg_idx], basecaller, read_type, fastq, kmer_bias)

                    if kmer_bias:
                        seg_mutated, seg_base_quals = mutate_homo(seg_mutated, seg_base_quals, kmer_bias, basecaller,
                                                                  None)
                    new_read_name += new_seg_name + ';'
                    read_mutated += seg_mutated
                    base_quals.extend(seg_base_quals)
                    if seg_idx < len(gap_list):
                        read_mutated += gap_list[seg_idx]
                        base_quals.extend(gap_base_qual_list[seg_idx])
                        new_read_name += "gap_" + str(len(gap_list[seg_idx])) + ';'

                    # Update base level abundance info
                    current_species_bases[species_list[seg_idx]] += len(new_seg)

                new_read_name = new_read_name + "aligned_" + str(sequence_index)
                if len(seg_length_list) > 1:
                    new_read_name += "_chimeric"
                if fastq:  # Get head/tail qualities and add to base_quals
                    ht_quals = mm.trunc_lognorm_rvs("ht", read_type, basecaller, head + tail).tolist()
                    base_quals = ht_quals[:head] + base_quals + ht_quals[head:]

            # Add head and tail region
            read_mutated = ''.join(np.random.choice(BASES, head)) + read_mutated + \
                           ''.join(np.random.choice(BASES, tail))

            # Reverse complement half of the reads
            p = random.random()
            if p > strandness_rate:
                read_mutated = reverse_complement(read_mutated)
                new_read_name += "_R"
                base_quals.reverse()
            else:
                new_read_name += "_F"

            if per:
                out_reads.write(id_begin + new_read_name + "_0_" + str(sum(ref_length_list)) + "_0" + '\n')
            else:
                out_reads.write(id_begin + new_read_name + "_" + str(head) + "_" +
                                ";".join(str(x) for x in ref_length_list) + "_" + str(tail) + '\n')
            out_reads.write(read_mutated + '\n')

            if fastq:
                out_reads.write("+\n")
                out_quals = "".join([chr(qual + 33) for qual in base_quals])
                out_reads.write(out_quals + "\n")

            if (sequence_index + 1) % 100 == 0:
                sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Number of reads simulated >> " +
                                 str(sequence_index + 1) + "\r")
                # +1 is just to ignore the zero index by python
                sys.stdout.flush()

            passed += 1

        remaining_reads = num_simulate - passed
        remaining_segments = num_segment[passed:]
        remaining_gaps = remaining_segments - 1

    sys.stdout.write('\n')
    out_reads.close()
    out_error.close()



TRUNCATION_MODES = ['ont_r9', 'ont_spatial', 'pacbio', 'none', 'custom', 'curio_le',  'curio_se', 'curio_ctrl']
TRUNCATION_DICT5 = {
    # pacbio sequel
    'pacbio': [0.41910000000000014, 0.01, 0.0143, 0.0099, 0.0092, 0.008, 0.0092, 0.0065, 0.0082, 0.0075, 0.0074, 0.0065, 0.0064, 0.0077, 0.0094, 0.0068, 0.0084, 0.009, 0.0083, 0.0077, 0.0065, 0.0066, 0.0069, 0.0139, 0.0076, 0.008, 0.0074, 0.0071, 0.0094, 0.0107, 0.0096, 0.0105, 0.0094, 0.0082, 0.0089, 0.0065, 0.0058, 0.0066, 0.0059, 0.0053, 0.0056, 0.0063, 0.0078, 0.0068, 0.0058, 0.0062, 0.0068, 0.0066, 0.0062, 0.0055, 0.0051, 0.0058, 0.0043, 0.0052, 0.0053, 0.0055, 0.0052, 0.0049, 0.0041, 0.005, 0.0044, 0.0043, 0.0054, 0.0049, 0.0041, 0.0046, 0.0042, 0.005, 0.0041, 0.0053, 0.0039, 0.0045, 0.0054, 0.0039, 0.0042, 0.0039, 0.0035, 0.0032, 0.0034, 0.0036, 0.0033, 0.0031, 0.003, 0.0033, 0.0037, 0.0028, 0.0029, 0.0022, 0.0033, 0.0027, 0.0029, 0.0022, 0.0024, 0.0023, 0.0045, 0.0023, 0.0026, 0.0024, 0.0026, 0.0029, 0.0025],
    # ont spatial
    'ont_spatial': [0.04898102326306356, 0.023963409304172272, 0.018331155064974184, 0.017022087746476944, 0.015843611214378467, 0.015427616370608226, 0.013467701423123627, 0.011759489684907603, 0.012822119551652342, 0.01064209593999564, 0.008554749660621929, 0.01360461111854168, 0.008778017779303678, 0.007636401549817752, 0.007543724217534763, 0.006825474892341588, 0.010533621335164412, 0.007389964098065256, 0.009086591169745906, 0.006708575075484634, 0.009397270863194565, 0.006632748167253096, 0.007479481975838599, 0.006342078352365535, 0.005679646056842799, 0.00603034550741366, 0.00592503035709208, 0.0057965458736997525, 0.005787067510170811, 0.006087215688587314, 0.0063094306557658466, 0.006537964531963674, 0.005960837508201418, 0.006928683739656735, 0.007285702099246893, 0.006789667741232251, 0.008074512575155525, 0.007272011129705086, 0.006911833315605283, 0.006801252407767624, 0.006719106590516792, 0.006883398225018456, 0.007255160705653634, 0.006986607072333604, 0.007256213857156849, 0.007349944340943056, 0.007101400586184126, 0.007731185185107173, 0.007637454701320968, 0.007644826761843478, 0.007635348398314535, 0.007704856397526778, 0.008029227060517246, 0.00877485832479403, 0.008512623600493296, 0.008329375238933746, 0.009349879045549856, 0.008737998022181477, 0.008644267538395272, 0.008850685233025568, 0.008990754382953268, 0.009275105288821533, 0.009420440196265313, 0.009291955712872986, 0.009630017345405258, 0.009647920920959927, 0.0100260023106144, 0.010593650970847713, 0.01037670176118526, 0.01048412321451327, 0.01070423187868537, 0.011249764357351156, 0.011207638297222523, 0.011824785078106982, 0.011954322713002525, 0.011681556473669634, 0.011894293077319224, 0.01238190222330814, 0.012989570640663653, 0.012645190099112087, 0.013084354275953074, 0.013542475179851948, 0.013472967180639703, 0.013545634634361595, 0.013800497298139817, 0.013817347722191273, 0.01368991639030216, 0.013918450266499988, 0.014003755538260469, 0.013608823724554542, 0.013727829844417928, 0.012909531126419252, 0.012268161860960834, 0.011291890417479787, 0.009998620371530787, 0.008512623600493296, 0.006321015322301221, 0.003999869409213601, 0.0018166863430472517, 0.0003054139359325815, 8.425212025726384e-06],
    # ont sirvs R9
    'ont_r9': [0.5807485641842356, 0.13738151418220534, 0.05887183003911555, 0.031937595589462714, 0.023225701666132793, 0.014341239155248566, 0.010606764688632128, 0.008649859463751573, 0.00788063216498008, 0.008692719635823885, 0.006399700429955199, 0.004554457232315667, 0.0038506480909177043, 0.004070588447604567, 0.0035111504121343915, 0.0030069794406521965, 0.002865992032519592, 0.0024012975353145264, 0.0024464135059169595, 0.004743944308845889, 0.002703574538350831, 0.0030520954112546305, 0.0021384970065553505, 0.0024520530022422637, 0.0018644174851455667, 0.0016072564527116957, 0.0018802080748564186, 0.002547924439772435, 0.0015249198063622543, 0.0016659072144948593, 0.002093381035952917, 0.0028253876589774012, 0.0018181736152780722, 0.0014234088725067786, 0.002099020532278221, 0.0011854221275789418, 0.0010275162304704242, 0.0011504572503620557, 0.0021892524734830883, 0.0013636302114585542, 0.0013884439952898927, 0.0010320278275306675, 0.0010884227907837095, 0.0010884227907837095, 0.001266630874663322, 0.0011075970782897437, 0.0010252604319403024, 0.0010184930363499375, 0.0008526918443859941, 0.0009000636135185493, 0.0008696103333619067, 0.0008831451245426367, 0.0008989357142534885, 0.0008921683186631235, 0.001091806488578892, 0.0009260052966149487, 0.00098127236060293, 0.0009429237855908614, 0.0011042133804945614, 0.0009654817708920781, 0.0008323896576148991, 0.0007365182200847278, 0.0006756116597714426, 0.0012790377665789913, 0.0007466693134702754, 0.000650797875940104, 0.0006271119913738264, 0.0006756116597714426, 0.0006429025810846782, 0.0010230046334101807, 0.0009891676554583558, 0.0009316447929402529, 0.0006000424090123664, 0.000609065603132853, 0.0005662054310605412, 0.0005797402222412712, 0.0005425195464942635, 0.0005154499641328033, 0.0005256010575183509, 0.0006282398906388872, 0.0008380291539402033, 0.0006598210700605907, 0.0005650775317954804, 0.0004049158361568412, 0.0004274738214580579, 0.0003879973471809286, 0.00049740357589183, 0.0004658223964701265, 0.0004116832317472062, 0.00032370708907246073, 0.0002594168309639929, 0.00023685884566277616, 0.0002436262412531412, 0.00020414976697601188, 0.0001917428750603426, 0.00012970841548199646, 0.00012632471768681395, 7.105765369883285e-05, 4.737176913255523e-05, 6.767395590365033e-06, 0.0],
    'curio': [0.052675690351052136, 0.029383655687376716, 0.02616030739398746, 0.023673625609709703, 0.02541095961476607, 0.021900553479798163, 0.01919829010365209, 0.018745222907907463, 0.020384565280296422, 0.018319823937857166, 0.017556642045788606, 0.016708149791162537, 0.017553183517576815, 0.016804988581092684, 0.016236637111621693, 0.01638996519567776, 0.0163634498127207, 0.01871524899673861, 0.02382349516555398, 0.016343851486187215, 0.015566835481271492, 0.014154603128123484, 0.014740247238653434, 0.015035374979392937, 0.013884837927603783, 0.013438687788282737, 0.012306596220289803, 0.012557915937013284, 0.013215036297253584, 0.012081791886523384, 0.012008009951338506, 0.011544567170958508, 0.01254984603785244, 0.012313513276713383, 0.011692131041328258, 0.011113403987221892, 0.012690492851798608, 0.010071234152735522, 0.012597112590080251, 0.010187671269199155, 0.011010800983605422, 0.010671865218849901, 0.009840665605282786, 0.00964698802542249, 0.009361083026581096, 0.00932188637351413, 0.008776591758788407, 0.008491839602684279, 0.008577149965241792, 0.008758146274992188, 0.009400279679648061, 0.008566774380606417, 0.007870457367299154, 0.01095315884674224, 0.0074024032159701, 0.007641041662583682, 0.007532674445280896, 0.007275590514871095, 0.007151083499246617, 0.00678217382332224, 0.00687324839956607, 0.0072156426925333835, 0.006920514951793881, 0.006241490579545572, 0.00651932567922612, 0.005758449472632089, 0.005803410339385372, 0.006506644409116219, 0.006035131729575373, 0.005515199655069453, 0.004957223770233831, 0.004995267580563532, 0.004801590000703235, 0.004425763268355274, 0.004135246898564826, 0.003958861959763483, 0.00377325427906403, 0.0036775683318711443, 0.0037421275251579107, 0.003638371678804179, 0.0035184760341287562, 0.0033882048048179607, 0.00288095400042194, 0.0026008132152668655, 0.0026181058563258214, 0.0025397125501918907, 0.0024359567038381596, 0.0026273285982239304, 0.0019056490446968657, 0.0015851587637375623, 0.001711971464836567, 0.0012370002570839304, 0.0010490868909099503, 0.0009499424155052737, 0.0006974698560445274, 0.000640980561918607, 0.00046228993764273635, 0.0003124203817984577, 0.00016485651142870647, 4.726655222781094e-05, 1.1528427372636817e-06],
    # no truncation
    'none' : [1.0] + [0] * 100,
    'custom' : [0] * 25 + [1.0] + [0] * 75,
    'curio_le': [0.050859936333857415, 0.023551484404012113, 0.021650368306602545, 0.020807891202211132, 0.021917337476744122, 0.016878178919227507, 0.015765961338430074, 0.014976139053063124, 0.015010318496991283, 0.015005699653217207, 0.014368299212394759, 0.014053294067002795, 0.014134585717426528, 0.014424649106438482, 0.013766925753010101, 0.014108720192291703, 0.013531364720532239, 0.015111009291266136, 0.015759494957146368, 0.013257929169106957, 0.013196960431289158, 0.011651495304483419, 0.012403443070902947, 0.011989594668745761, 0.011162821633186208, 0.011620087166819703, 0.010360066585251848, 0.010742506849745318, 0.01264824179092896, 0.011127718420503231, 0.011448266178424087, 0.011167440476960283, 0.011108319276652113, 0.010742506849745318, 0.011321709859014411, 0.011205314995907703, 0.011355889302942575, 0.012750780122713442, 0.011379907290567768, 0.010726802780913461, 0.010235357803351804, 0.010540201492440802, 0.00900859289695729, 0.011115709426690636, 0.010128200627793247, 0.011864885886845717, 0.009257086692002564, 0.008789659702066101, 0.009339302111181112, 0.009271866992079606, 0.009116673841270661, 0.009249696541964043, 0.008830305527277968, 0.00887002758373502, 0.008605829719857887, 0.008657560770127536, 0.007585065245787153, 0.01166073299203157, 0.009498190337009316, 0.008798897389614252, 0.007925012147559125, 0.007754114927918324, 0.0073605894383670725, 0.007714392871461273, 0.007977666966583589, 0.007546266958084916, 0.00753425796427232, 0.008391515368740774, 0.008266806586840729, 0.0077282494027835005, 0.0064839328900475005, 0.007282069094207785, 0.0072829928629626, 0.006487627965066761, 0.007578598864503446, 0.0061245868444244084, 0.006678848097313495, 0.006302874214103732, 0.006843278935670591, 0.006032209968942894, 0.005512128159981968, 0.006134748300727376, 0.0055130519287367825, 0.005064100313896623, 0.0056359131731271985, 0.0048276155126639455, 0.0046290052303786895, 0.004560646342522369, 0.004236403509582253, 0.003830869026218405, 0.004137560252817033, 0.004047954683599964, 0.003288616767141915, 0.003275684004574503, 0.0025135747818520087, 0.0025089559380779333, 0.0021098878359977905, 0.0017015820463694962, 0.001169491243595973, 0.0005182342714512961, 1.4780300077042315e-05],
    'curio_se': [0.0252484327064159, 0.022255568894224358, 0.021839569160997732, 0.0199171335200747, 0.018950913698812858, 0.017176037081499267, 0.015784647192210215, 0.015178571428571428, 0.01514855942376951, 0.015208583433373349, 0.014406595971722022, 0.014181505935707616, 0.015314459116980124, 0.015013505402160865, 0.013608776844070964, 0.015166900093370681, 0.01649326397225557, 0.014885954381752702, 0.014573329331732693, 0.014096471922102175, 0.014329898626117112, 0.01348372682406296, 0.013141923436041083, 0.013249466453247965, 0.012741763372015472, 0.011920601573962918, 0.011981459250366814, 0.01182139522475657, 0.012074829931972788, 0.011512938508736829, 0.011569627851140457, 0.011161964785914365, 0.01172218887555022, 0.011632986527944512, 0.011206149126317195, 0.01185140722955849, 0.010968554088301987, 0.011537114845938376, 0.012280745631585967, 0.010919367747098839, 0.011678004535147392, 0.011703014539148994, 0.011218654128317993, 0.011668834200346806, 0.010906029078297986, 0.010625917033480058, 0.010920201413898892, 0.009995664932639722, 0.010004001600640255, 0.010454181672669068, 0.010309957316259838, 0.010237428304655195, 0.009561324529811925, 0.009748065893023877, 0.009783079898626118, 0.009312891823396027, 0.009466286514605843, 0.009084467120181407, 0.009921468587434973, 0.00973389355742297, 0.008553421368547419, 0.00979141656662665, 0.00846171802054155, 0.008649293050553555, 0.008602607709750567, 0.008520908363345339, 0.008135754301720689, 0.008687641723356008, 0.008798519407763106, 0.00818077230892357, 0.00800653594771242, 0.007846471922102175, 0.007367947178871549, 0.00701530612244898, 0.007223722822462318, 0.006601807389622515, 0.006572629051620648, 0.006317527010804322, 0.00637421635320795, 0.006358376684006936, 0.006152460984393757, 0.006163298652794451, 0.005568060557556357, 0.004903628117913833, 0.004683540082699746, 0.004619347739095638, 0.004723556089102308, 0.004290049353074563, 0.0038015206082432974, 0.0031220821661998134, 0.0029836934773909566, 0.002691910097372282, 0.002364279044951314, 0.001812391623315993, 0.001342203548085901, 0.0011362878484727226, 0.0009111978124583168, 0.0004518474056289182, 0.00026760704281712685, 5.33546752034147e-05, 8.336668000533547e-07],
    'curio_ctrl': [0.028703110778369158, 0.01758042970583373, 0.014762468480828173, 0.012532611773111369, 0.011194887766095918, 0.01171838629440949, 0.010444318188296055, 0.0103151062103457, 0.009024886606988467, 0.008169807341140523, 0.009403021660107891, 0.009293761531693989, 0.00936786840140081, 0.009026786783134797, 0.008462434467675153, 0.008708507278624728, 0.009550285311448371, 0.008722758599722194, 0.009140797351914523, 0.010693241263465124, 0.01042626651490593, 0.008914676390501398, 0.009612041036204055, 0.009754554247178714, 0.00820591068792077, 0.008631550144698413, 0.007747968236655538, 0.008054846684287634, 0.008912776214355071, 0.009486629410546357, 0.009195902460158056, 0.01025715083788267, 0.008395928302553646, 0.008621099175893605, 0.00818405866223799, 0.008070998181531428, 0.009213954133548181, 0.012049016943870696, 0.009127496118890221, 0.008703756838258907, 0.009380219546351947, 0.009801108562763769, 0.00893652841618418, 0.01179629351640897, 0.011831446775116052, 0.009360267696815494, 0.008323721608993152, 0.008779763884112057, 0.013769626444371392, 0.011233841377095656, 0.009911318779250836, 0.009078091539085674, 0.009414422716985863, 0.008933678151964688, 0.009569287072911659, 0.01089370984690281, 0.00940492183625422, 0.014001447934223503, 0.009086642331744154, 0.009414422716985863, 0.009280460298669687, 0.009992076265469808, 0.00998542564895766, 0.009120845502378072, 0.01124809269819312, 0.011065675788145563, 0.012946850173011037, 0.010894659934975973, 0.012460405079550874, 0.010526975850661357, 0.00974315319030074, 0.011244292345900464, 0.010048131461786507, 0.010362610614003918, 0.010021528995737905, 0.010074733927835111, 0.01101532112026785, 0.010151691061761425, 0.010965916540463302, 0.01028470339200444, 0.010473770918564152, 0.01067898994236766, 0.010648587124026396, 0.011257593578924766, 0.010085184896639919, 0.011012470856048355, 0.011311748599095136, 0.010480421535076301, 0.010337908324101643, 0.010147890709468768, 0.010261901278248494, 0.01005478207829866, 0.009501830819716988, 0.009352666992230179, 0.008000691664117264, 0.007317578339512073, 0.006742775055247621, 0.005177979998745885, 0.0034231673276112697, 0.0009823910676519715, 1.3301233024301354e-05]
}

TRUNCATION_DICT3 = {
    # pacbio sequel
    'pacbio': [0.40170000000000206, 0.0301, 0.0574, 0.0651, 0.0552, 0.0503, 0.0545, 0.0445, 0.0409, 0.0319, 0.0273, 0.0192, 0.0152, 0.0111, 0.0092, 0.0067, 0.006, 0.0055, 0.0044, 0.0037, 0.0033, 0.003, 0.0034, 0.0033, 0.0028, 0.002, 0.0021, 0.0017, 0.0016, 0.0013, 0.0013, 0.0011, 0.001, 0.001, 0.0005, 0.0009, 0.0009, 0.0008, 0.0005, 0.0004, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0006, 0.0005, 0.0004, 0.0006, 0.0005, 0.0004, 0.0005, 0.0003, 0.0005, 0.0003, 0.0003, 0.0002, 0.0003, 0.0006, 0.0004, 0.0006, 0.0005, 0.0002, 0.0003, 0.0003, 0.0005, 0.0005, 0.0006, 0.0003, 0.0005, 0.0006, 0.0007, 0.0011, 0.0008, 0.0008, 0.0004, 0.0003, 0.0006, 0.0005, 0.0004, 0.0005, 0.0004, 0.0004, 0.0003, 0.0002, 0.0005, 0.0004, 0.0005, 0.0005, 0.0005, 0.0004, 0.0003, 0.0004, 0.0004, 0.0003, 0.0004, 0.0003, 0.0002, 0.0003, 0.0003, 0.0003],
    # ont spatial
    'ont_spatial': [0.4858177352819445, 0.1328139892220475, 0.043367725750923354, 0.021959261993552606, 0.014252299293019395, 0.011213957206241817, 0.00989541152421564, 0.008553696509118712, 0.007868094880525228, 0.00723093822107967, 0.007884945304576679, 0.007049796162526553, 0.006169361505838146, 0.0074173460371488665, 0.006339972049359104, 0.004602272069053038, 0.004988778670733236, 0.004157842134695971, 0.004529604615331148, 0.004276848254559356, 0.004084121529470865, 0.004094653044503023, 0.0067380633175746765, 0.004586474796504801, 0.005262598061569344, 0.0030794149954029943, 0.003989337894181443, 0.0036649672311909777, 0.004500116373241105, 0.004025145045290781, 0.0034775062636185653, 0.0029972691781521613, 0.002933026936455998, 0.004062005347903333, 0.0057512603590614735, 0.004417970555990273, 0.004853975278321613, 0.010214516429690026, 0.0045559334029115425, 0.00367444559471992, 0.002774001059470412, 0.0041230881350898495, 0.00413783225613487, 0.0039946036516975226, 0.003052033056319383, 0.0033047894170911743, 0.0028150739680958283, 0.0034975161421796656, 0.003081521298409425, 0.0029098576033852507, 0.004272635648546493, 0.0025286167592211313, 0.00451907310029899, 0.0028930071793337973, 0.0034827720211346442, 0.0031436572370991574, 0.0027992766955475916, 0.002478065487066773, 0.002550732940788663, 0.0023327305796229927, 0.0015734083458044024, 0.003005694390177888, 0.002131578642508775, 0.0020841868248640647, 0.001762975616383246, 0.00283719014966336, 0.0015039003465921598, 0.0013596185906515955, 0.0023643251247194665, 0.0017092648897192404, 0.0016281722239716238, 0.0020030941591164483, 0.0010900118058283512, 0.0011426693809891409, 0.0009741651404746132, 0.0013954257417609325, 0.000959421019429592, 0.0013280240455551212, 0.0015249633766564758, 0.0010310353216482662, 0.0010352479276611295, 0.0012048053196788732, 0.0010573641092286614, 0.0004202074497831034, 0.0006097747203619471, 0.0014891562255471389, 0.000443376782853851, 0.0005486919331754308, 0.0007540564763025114, 0.00040125072272521905, 0.0006118810233683787, 0.0008362022935533437, 0.00029698872390685507, 0.00022116181567531762, 0.0001727168465273909, 0.00014849436195342754, 0.00012321872587624838, 8.635842326369544e-05, 6.950799921224268e-05, 1.579727254823697e-05, 1.053151503215798e-06],
    # ont sirvs R9
    'ont_r9': [0.46713188751686213, 0.16673847625320887, 0.03520399186107891, 0.01779148300706967, 0.011167330623367366, 0.007589634154594385, 0.0057037865834126626, 0.005360905206834167, 0.004714618927954307, 0.005824471804774172, 0.003999530793905735, 0.003578824368038042, 0.00447776008229153, 0.003863054982833373, 0.004495806470532503, 0.0033385818245800832, 0.003102850878182368, 0.0023415188742663016, 0.002119322719049316, 0.0030261537281582305, 0.002912235902387086, 0.002373100053688005, 0.003551754785676582, 0.007077567888256763, 0.00419578526602632, 0.002210682559519244, 0.004627770684544622, 0.002146392301410776, 0.0036092776481946843, 0.002611086798615842, 0.0025039363684350624, 0.003856287587243008, 0.0038607991843032513, 0.0028445619464834357, 0.003017130534037744, 0.004093146432905784, 0.002603191503760416, 0.002035858173434814, 0.003538219994495851, 0.003646498323941692, 0.004468736888171044, 0.0032054897113029043, 0.0035348362967006688, 0.0019862306057721376, 0.002995700448001588, 0.0024700993904832372, 0.002491529476519393, 0.002064055655061335, 0.003814555314435757, 0.0023054260977843546, 0.004129239209387731, 0.0024678435919531154, 0.0019704400160612855, 0.001982846907976955, 0.002121578517579438, 0.0027633531993990555, 0.00470559573383382, 0.0034919761246283566, 0.0025039363684350624, 0.004561224627906033, 0.0024982968721097582, 0.0020798462447721867, 0.002897573211941295, 0.0023437746727964235, 0.0026392842802423627, 0.0032111292076282085, 0.006500083464545616, 0.006404212027015444, 0.0049221523927255005, 0.0023291119823506323, 0.002362948960302458, 0.003120897266423342, 0.0026697375603990056, 0.0038326017026767308, 0.003440092758435558, 0.008549476429161158, 0.003399488384893368, 0.0025400291449170095, 0.002512959562555549, 0.001722302177747901, 0.002155415495531263, 0.00236633265809764, 0.0031242809642185237, 0.0027306441207122912, 0.0022715891198325294, 0.003955542722568362, 0.0019287077432540345, 0.001819301514543133, 0.003299105350302954, 0.005128557958231634, 0.0014031066857356834, 0.0016433492291936425, 0.0014741643394345165, 0.0014403273614826914, 0.0015813147696152963, 0.0037209396754357077, 0.0004489039074942139, 0.000371078858205016, 0.00019851027065070765, 4.286017207231188e-05, 0.0],
    'curio': [0.2910904854736051, 0.03490577239886975, 0.01320466071261821, 0.011666768501108458, 0.009311510788878758, 0.009954797036271891, 0.008551787425021991, 0.011146836426602538, 0.007592622267618608, 0.010566956529758906, 0.009412960949757961, 0.007808203859486915, 0.007404708901444628, 0.007284813256769204, 0.006299132716408756, 0.006364844752432787, 0.007434682812613483, 0.006167708644360697, 0.007828955028757661, 0.007676779787438857, 0.007584552368457762, 0.006424792574770498, 0.006891693883362288, 0.008079121902743881, 0.007085371463222587, 0.00609969092286214, 0.007385110574911144, 0.007998422911135424, 0.008627875045681393, 0.0073897219458602, 0.0063233424138912945, 0.010414781288440101, 0.008345428575051791, 0.006422486889295971, 0.0073366911799460695, 0.00815059815245423, 0.0077494088798864685, 0.006111219350234776, 0.007209878478847065, 0.007646805876270001, 0.00914319574923826, 0.008120624241285374, 0.007674474101964329, 0.00707384303584995, 0.008288939280925871, 0.0076917667430232844, 0.0077044480131331855, 0.006870942714091542, 0.00911206899533214, 0.005509435441383135, 0.006390207292652588, 0.006587343400724677, 0.006964322975809901, 0.007257145031074877, 0.006731448742882638, 0.007386263417648409, 0.007266367772972985, 0.007600692166779454, 0.0075603426709752244, 0.0063394822122129855, 0.00934033185731035, 0.0057503795734712435, 0.009532856594433384, 0.00716145908388199, 0.007379346361224826, 0.006282992918087065, 0.007187974466839055, 0.008510285086480497, 0.008073357689057563, 0.006570050759665723, 0.006196529712792289, 0.007146472128297562, 0.007732116238827513, 0.008912627201785522, 0.007098052733332488, 0.007220254063482439, 0.012145198237072887, 0.007469268094731394, 0.005653540783541095, 0.005961349794390498, 0.008074510531794827, 0.007370123619326717, 0.00789697275025622, 0.004514532159124577, 0.005218919071592687, 0.004164067966996418, 0.004580244195148607, 0.0027356958155267167, 0.0031749288984241793, 0.0033086586559467664, 0.0027760453113309454, 0.0023494934985433835, 0.002171955717004776, 0.001314240720480597, 0.0020739640843373637, 0.005120927438925274, 0.0017753778153860695, 0.00034239429296731343, 0.0004726655222781095, 0.00101450160879204, 0.0],
    'none' : [1.0] + [0] * 100,
    'custom' : [1.0] + [0] * 100,
    'curio_le': [0.3215223339571852, 0.04074374469987677, 0.014677761745257834, 0.011715235348565665, 0.010132819471567323, 0.009608118818832318, 0.007836330347096871, 0.008591049419780845, 0.006667762872255714, 0.008783193320782395, 0.00805341600447843, 0.007411396719881905, 0.006547672934129745, 0.0060128108250917765, 0.005570325591535322, 0.005786487480162066, 0.00587516928062432, 0.00546593972224121, 0.006492246808840836, 0.005493652784885665, 0.004903364550558788, 0.005619285335540525, 0.005279338433768553, 0.007784599296827224, 0.005590648504141255, 0.004421157260545283, 0.0063804707895082045, 0.008720377045454966, 0.009099122234929174, 0.0071398087059662526, 0.005671940154564988, 0.009717123531900506, 0.00772270679025461, 0.0061375196069918215, 0.0062252776386992606, 0.006146757294539972, 0.006122739306914779, 0.005221141002215197, 0.007198006137519607, 0.007210938900087019, 0.006606794134437915, 0.006385089633282281, 0.006074703331664391, 0.006303797982858547, 0.006547672934129746, 0.007666356896210885, 0.005882559430662841, 0.006153223675823679, 0.008585506807251954, 0.005560164135232355, 0.007322714919419652, 0.007516706357930832, 0.007145351318495144, 0.006220658794925184, 0.006492246808840837, 0.0065125697214467706, 0.0065125697214467706, 0.008453407875313389, 0.007068678511845486, 0.006811870798006876, 0.008149487954979205, 0.006230820251228151, 0.009856612613877594, 0.006165232669636276, 0.0071693693061203375, 0.006277008688968908, 0.008076510223348808, 0.008588278113516399, 0.009112054997496587, 0.00603313373769771, 0.005749536729969459, 0.006619726897005327, 0.0071398087059662526, 0.009133301678857335, 0.009971159939474672, 0.007262669950356667, 0.008025702941833976, 0.007770742765504997, 0.006045142731510306, 0.006529197559033443, 0.008133783886147349, 0.008353640849793353, 0.0076404913710760615, 0.005345849784115242, 0.0052239123084796436, 0.004932925150712872, 0.0054391504283515715, 0.003926017207964365, 0.0038946090703006497, 0.004581893023883117, 0.0035278728746390375, 0.002326049724624534, 0.003290464304651545, 0.0021930270239311532, 0.0034197919303256656, 0.003923245901699919, 0.0017265238027495053, 0.00045357045861423603, 0.00033163298297863696, 0.00030761499535344317, 0.0],
    'curio_se': [0.23255385487528343, 0.05117463652127517, 0.016711684673869548, 0.012119014272375618, 0.009081966119781246, 0.00870764972655729, 0.007174536481259171, 0.006291683340002668, 0.005709783913565427, 0.007620548219287716, 0.006558456716019741, 0.006955282112845138, 0.006167466986794718, 0.006271675336801387, 0.005687274909963986, 0.005504701880752302, 0.005638922235560891, 0.005511371215152727, 0.005532212885154062, 0.005709783913565427, 0.0062391623315993065, 0.006245831665999733, 0.005651427237561692, 0.006855242096838736, 0.006097438975590236, 0.0054238362011471255, 0.006338368680805655, 0.00821578631452581, 0.01013988928904895, 0.007753101240496198, 0.006903594771241831, 0.009823929571828732, 0.007106175803654795, 0.006755202080832334, 0.00755718954248366, 0.0069777911164465795, 0.007563858876884087, 0.006540116046418567, 0.007262905162064825, 0.007333766840069361, 0.007303754835267439, 0.008151593970921703, 0.006590136054421769, 0.00718454048285981, 0.007446311858076564, 0.007546351874082966, 0.007555522208883553, 0.007450480192076831, 0.008105742296918768, 0.0073062558356676, 0.0078156262505002, 0.007875650260104042, 0.007462151527277578, 0.007221221822062158, 0.007717253568093904, 0.008551754034947312, 0.008526744030945712, 0.00947045484860611, 0.00901860744297719, 0.009038615446178472, 0.009364579164999333, 0.00847672402294251, 0.012243230625583565, 0.007770608243297319, 0.008347505668934241, 0.007985694277711085, 0.009148659463785515, 0.008478391356542617, 0.011006069094304387, 0.00905028678137922, 0.008650960384153661, 0.008518407362945178, 0.00793233960250767, 0.009631352541016408, 0.010103207949846604, 0.008997765772975858, 0.008736828064559157, 0.008052387621715352, 0.007772275576897425, 0.008806022408963585, 0.009854775243430704, 0.01164965986394558, 0.010030678938241964, 0.00793984260370815, 0.007468820861678005, 0.007333766840069361, 0.006815226090436175, 0.005773142590369482, 0.006010737628384687, 0.006653494731225823, 0.004116646658663465, 0.0032546351874082964, 0.003998265973055889, 0.00377984527144191, 0.0031245831665999728, 0.004225857009470455, 0.0012346605308790183, 0.0005527210884353741, 0.00021008403361344536, 9.170334800586902e-05, 1.6673336001067094e-06],
    'curio_ctrl': [0.3270089137263025, 0.0501057448025432, 0.02195368510660938, 0.01958606562828374, 0.015766711574162928, 0.014587652275365926, 0.012982953519791285, 0.015349622910043761, 0.012118373373211698, 0.013082712767473546, 0.013265129677521106, 0.011563521938483699, 0.010505123824978575, 0.010292304096589756, 0.010920312312951411, 0.011076126756950367, 0.009602540155472412, 0.008944129120769495, 0.008893774452891783, 0.009043888368451755, 0.008990683436354549, 0.01310741505737582, 0.009167399817963126, 0.01030655541768722, 0.008026344042092702, 0.0073726834477556066, 0.007654859605485429, 0.009739352838008083, 0.009855263582934138, 0.007447740405535593, 0.007180765656976402, 0.008963130882232783, 0.007397385737657881, 0.006376991147079334, 0.00718266583312273, 0.006415944758079074, 0.006238278288397334, 0.006142319393007732, 0.006528055150712471, 0.006645866071784855, 0.0066838695947114305, 0.005864893675643731, 0.00545255545189039, 0.005562765668377459, 0.0062800821636165675, 0.005611220160108841, 0.005507660560133925, 0.0049338073639426374, 0.005686277117888828, 0.005162778589575253, 0.005597918927084541, 0.005568466196816445, 0.00523783554735524, 0.005590318222499226, 0.005329994090452185, 0.005162778589575253, 0.004907204897894034, 0.004566123279628021, 0.005042117404283378, 0.005143776828111965, 0.005438304130792925, 0.004257344655849597, 0.004689634729139391, 0.005374648229890911, 0.004854950053869994, 0.004180387521923282, 0.005058268901527172, 0.004626928916310542, 0.004895803841016062, 0.00373764647982868, 0.004124332325606584, 0.00372244507065805, 0.0033851638046846944, 0.003765199033950447, 0.0053736981418177465, 0.0043019987952883235, 0.0031542924029057495, 0.003928614182534721, 0.0029490733791022426, 0.0031419412579546126, 0.0029766259332240095, 0.00492335639513783, 0.004093929507265323, 0.003099187294662215, 0.002794209023176449, 0.002566187885616997, 0.0026136922892752157, 0.0017652636399394222, 0.002507282425080805, 0.002624143258080024, 0.0015590945280627513, 0.001338674095088615, 0.0011021021648706835, 0.0010450968804808205, 0.0017006576509642445, 0.0011572072731142178, 0.0004674433319968761, 0.00022707104948628735, 0.0001130604807065615, 8.170757429213689e-05, 1.9001761463287647e-06]
 
}

def calculate_aligned_length(ref_len, truncation_mode='ont_r9'):
    bp5_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31,  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    bp3_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    pro5_list = TRUNCATION_DICT5[truncation_mode]
    pro3_list = TRUNCATION_DICT3[truncation_mode]
    del5 = int(np.random.choice(bp5_list, p=pro5_list) * ref_len / 100.0)
    del3 = int(np.random.choice(bp3_list, p=pro3_list) * ref_len / 100.0)
    new_len = max(min(50, ref_len - 1), ref_len - del3 - del5 - 1)
    start_pos = max(0, min(del5 - 1, ref_len - new_len - 1))
#    new_len = min(50, ref_len)
#    start_pos = max(0, ref_len - new_len)
    return new_len, start_pos


def simulation_aligned_transcriptome(model_ir, out_reads, out_error, kmer_bias, basecaller, read_type, num_simulate,
                                     polya, fastq, per=False, uracil=False, truncation_mode='ont_r9'):
    # Simulate aligned reads
    out_reads = open(out_reads, "w")
    out_error = open(out_error, "w")

    if fastq:
        id_begin = "@"
    else:
        id_begin = ">"

    if model_ir:
        # check whether chrom names contains "chr" or not.
        flag_chrom = False
        for item in genome_fai.references:
            if "chr" in item:
                flag_chrom = True
                break
    sampled_2d_lengths = get_length_kde(kde_aligned_2d, num_simulate, False, False)

    remainder_l = get_length_kde(kde_ht, num_simulate, True)
    head_vs_ht_ratio_temp = get_length_kde(kde_ht_ratio, num_simulate)
    head_vs_ht_ratio_l = [1 if x > 1 else x for x in head_vs_ht_ratio_temp]
    head_vs_ht_ratio_l = [0 if x < 0 else x for x in head_vs_ht_ratio_l]

    remaining_reads = 0
    while remaining_reads < num_simulate:
        while True:
            ref_trx = select_ref_transcript_fast(list_cdf, sorted_ids)
            ref_trx_len = seq_len[ref_trx]
            if polya and ref_trx in trx_with_polya:
                trx_has_polya = True
            else:
                trx_has_polya = False

            if model_ir:
                if ref_trx in dict_ref_structure:
                    ref_trx_len_fromstructure = ref_len_from_structure(dict_ref_structure[ref_trx])
                    if ref_trx_len == ref_trx_len_fromstructure:
                        ref_len_aligned, start_pos = calculate_aligned_length(ref_trx_len, truncation_mode)
                        # if ref_len_aligned < ref_trx_len:
                        break
            else:
                ref_len_aligned, start_pos = calculate_aligned_length(ref_trx_len, truncation_mode)
                # print(ref_trx_len, ref_len_aligned, start_pos)
                # if ref_len_aligned < ref_trx_len:
                break
        # print(ref_trx_len, ref_len_aligned, start_pos)

        if per:
            with total_simulated.get_lock():
                sequence_index = total_simulated.value
                total_simulated.value += 1

            new_read, ref_start_pos, retain_polya = extract_read_trx(ref_trx, ref_len_aligned, trx_has_polya, ref_pos=start_pos)
            new_read_name = ref_trx + "_" + str(ref_start_pos) + "_perfect_" + str(sequence_index)
            read_mutated = case_convert(new_read)  # not mutated actually, just to be consistent with per == False

            if fastq:
                base_quals = mm.trunc_lognorm_rvs("match", read_type, basecaller, ref_len_aligned).tolist()
            else:
                base_quals = []

            head = 0
            tail = 0

        else:
            middle_read, middle_ref, error_dict, error_count = error_list(ref_len_aligned, match_markov_model,
                                                                          match_ht_list, error_par, trans_error_pr,
                                                                          fastq)

            if middle_ref > ref_trx_len:
                pass #continue

            with total_simulated.get_lock():
                sequence_index = total_simulated.value
                total_simulated.value += 1

            ir_list = []
            if model_ir:
                ir_flag, ref_trx_structure_new = update_structure(dict_ref_structure[ref_trx], IR_markov_model)
                if ir_flag:
                    list_iv, retain_polya, ir_list = extract_read_pos(middle_ref, ref_trx_len, ref_trx_structure_new,
                                                                      trx_has_polya)
                    new_read = ""
                    flag = False
                    for interval in list_iv:
                        chrom = interval.chrom
                        if flag_chrom:
                            chrom = "chr" + chrom
                        if chrom not in genome_fai.references:
                            flag = True
                            break
                        start = interval.start
                        end = interval.end
                        new_read += genome_fai.fetch(chrom, start, end)  # len(new_read) > middle_ref
                    if flag:
                        continue
                    ref_start_pos = list_iv[0].start
 
                    if interval.strand == '-':  # Keep the read direction the same as reference transcripts
                        new_read = reverse_complement(new_read)

                    if fastq:  # since len(new_read) > middle_ref if IR, add more match quals for retained intron
                        error_count["match"] += len(new_read) - middle_ref
                else:
                    new_read, ref_start_pos, retain_polya = extract_read_trx(ref_trx, middle_ref, trx_has_polya, ref_pos=start_pos)

            else:
                new_read, ref_start_pos, retain_polya = extract_read_trx(ref_trx, middle_ref, trx_has_polya, ref_pos=start_pos)

            new_read_name = str(ref_trx) + "_" + str(ref_start_pos) + "_aligned_" + str(sequence_index)
            if len(ir_list) > 0:
                new_read_name += "_RetainedIntron_"
                for ir_tuple in ir_list:
                    new_read_name += '-'.join(str(x) for x in ir_tuple) + ';'

            # start HD len simulation
            remainder = int(remainder_l[remaining_reads])
            head_vs_ht_ratio = head_vs_ht_ratio_l[remaining_reads]

            if True: #remainder == 0:
                head = 0
                tail = 0
            else:
                head = int(round(remainder * head_vs_ht_ratio))
                tail = remainder - head
            # end HD len simulation

            # Mutate read
            new_read = case_convert(new_read)
            read_mutated, base_quals = mutate_read(new_read, new_read_name, out_error, error_dict, error_count,
                                                   basecaller, read_type, fastq, kmer_bias)
            if kmer_bias:
                read_mutated, base_quals = mutate_homo(read_mutated, base_quals, kmer_bias, basecaller, read_type)

        if retain_polya:
            if basecaller == "albacore":
                polya_len = int(scipy.stats.expon.rvs(loc=2.0, scale=2.409858743694814))
            else:  # guppy
                polya_len = int(scipy.stats.expon.rvs(loc=2.0, scale=4.168299657168961))
            read_mutated += "A" * (polya_len + 10)
        else:
            polya_len = 0

        if fastq:  # Get head/tail qualities
            ht_quals = mm.trunc_lognorm_rvs("ht", read_type, basecaller, head + tail + polya_len).tolist()
            for a in xrange(polya_len):
                base_quals.append(ht_quals.pop())
            base_quals = ht_quals[:head] + base_quals + ht_quals[head:]

        # Add head and tail region
        read_mutated = ''.join(np.random.choice(BASES, head)) + read_mutated + ''.join(np.random.choice(BASES, tail))
        
        # Reverse complement according to strandness rate
        p = random.random()
        if p > strandness_rate:
            read_mutated = reverse_complement(read_mutated)
            new_read_name += "_R"
            base_quals.reverse()
        else:
            new_read_name += "_F"

        if per:
            out_reads.write(id_begin + new_read_name + "_0_" + str(ref_len_aligned + polya_len) + "_0" + '\n')
        else:
            out_reads.write(id_begin + new_read_name + "_" + str(head) + "_" + str(middle_ref) + "_" +
                            str(tail + polya_len) + '\n')

        if uracil:
            read_mutated = read_mutated.translate(trantab)

        out_reads.write(read_mutated + '\n')

        if fastq:
            out_reads.write("+\n")
            out_quals = "".join([chr(qual + 33) for qual in base_quals])
            out_reads.write(out_quals + "\n")

        if (sequence_index + 1) % 100 == 0:
            sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Number of reads simulated >> " +
                             str(sequence_index + 1) + "\r")
            # +1 is just to ignore the zero index by python
            sys.stdout.flush()

        remaining_reads += 1

    sys.stdout.write('\n')
    out_reads.close()
    out_error.close()


def simulation_aligned_genome(dna_type, min_l, max_l, median_l, sd_l, out_reads, out_error, kmer_bias, basecaller,
                              read_type, fastq, num_simulate, per=False, chimeric=False):

    # Simulate aligned reads
    out_reads = open(out_reads, "w")
    out_error = open(out_error, "w")

    id_begin = '@' if fastq else '>'

    remaining_reads = num_simulate
    if chimeric:
        num_segment = np.random.geometric(1/segment_mean, num_simulate)
    else:
        num_segment = np.ones(num_simulate, dtype=int)
    remaining_segments = num_segment
    remaining_gaps = remaining_segments - 1
    passed = 0
    while remaining_reads > 0:
        if per:
            ref_lengths = get_length_kde(kde_aligned, sum(remaining_segments)) if median_l is None else \
                np.random.lognormal(np.log(median_l), sd_l, remaining_segments)
            ref_lengths = [x for x in ref_lengths if min_l <= x <= max_l]
        else:
            remainder_lengths = get_length_kde(kde_ht, int(remaining_reads * 1.3), True)
            remainder_lengths = [x for x in remainder_lengths if x >= 0]
            head_vs_ht_ratio_list = get_length_kde(kde_ht_ratio, int(remaining_reads * 1.5))
            head_vs_ht_ratio_list = [x for x in head_vs_ht_ratio_list if 0 <= x <= 1]
            if median_l is None:
                ref_lengths = get_length_kde(kde_aligned, sum(remaining_segments))
            else:
                total_lengths = np.random.lognormal(np.log(median_l + sd_l ** 2 / 2), sd_l, remaining_reads)
                ref_lengths = total_lengths - remainder_lengths
            ref_lengths = [x for x in ref_lengths]

        gap_lengths = get_length_kde(kde_gap, sum(remaining_gaps), True) if sum(remaining_gaps) > 0 else []
        gap_lengths = [max(0, int(x)) for x in gap_lengths]

        seg_pointer = 0
        gap_pointer = 0
        for each_read in xrange(remaining_reads):
            # check if the total length fits the criteria
            segments = remaining_segments[each_read]
            # In case too many ref length was filtered previously
            if seg_pointer + segments > len(ref_lengths):
                break
            ref_length_list = [int(ref_lengths[seg_pointer + x]) for x in range(segments)]
            gap_length_list = [int(gap_lengths[gap_pointer + x]) for x in range(segments - 1)]

            if per:
                seg_pointer += 1
                gap_pointer += 1
                with total_simulated.get_lock():
                    sequence_index = total_simulated.value
                    total_simulated.value += 1

                # Extract middle region from reference genome
                new_read = ""
                new_read_name = ""
                base_quals = []
                for each_ref in ref_length_list:
                    new_seg, new_seg_name = extract_read(dna_type, each_ref)
                    new_read += new_seg
                    new_read_name += new_seg_name
                    if fastq:
                        base_quals.extend(mm.trunc_lognorm_rvs("match", read_type, basecaller, each_ref).tolist())

                new_read_name = new_read_name + "_perfect_" + str(sequence_index)
                read_mutated = case_convert(new_read)  # not mutated actually, just to be consistent with per == False

                head = 0
                tail = 0

            else:
                gap_list = []
                gap_base_qual_list = []
                seg_length_list = []
                seg_error_dict_list = []
                seg_error_count_list = []
                remainder = int(remainder_lengths[each_read])
                head_vs_ht_ratio = head_vs_ht_ratio_list[each_read]

                total = remainder
                for each_gap in gap_length_list:
                    mutated_gap, gap_base_quals = simulation_gap(each_gap, basecaller, read_type, dna_type, fastq)
                    gap_list.append(mutated_gap)
                    gap_base_qual_list.append(gap_base_quals)
                for each_ref in ref_length_list:
                    middle, middle_ref, error_dict, error_count = \
                        error_list(each_ref, match_markov_model, match_ht_list, error_par, trans_error_pr, fastq)
                    total += middle
                    seg_length_list.append(middle_ref)
                    seg_error_dict_list.append(error_dict)
                    seg_error_count_list.append(error_count)

                if total < min_l or total > max_l or max(seg_length_list) > max_chrom:
                    continue

                seg_pointer += segments
                gap_pointer += segments - 1

                with total_simulated.get_lock():
                    sequence_index = total_simulated.value
                    total_simulated.value += 1

                if remainder == 0:
                    head = 0
                    tail = 0
                else:
                    head = int(round(remainder * head_vs_ht_ratio))
                    tail = remainder - head

                # Extract middle region from reference genome
                read_mutated = ""
                new_read_name = ""
                base_quals = []
                for seg_idx in range(len(seg_length_list)):
                    new_seg, new_seg_name = extract_read(dna_type, seg_length_list[seg_idx])
                    # Mutate read
                    new_seg = case_convert(new_seg)
                    seg_mutated, seg_base_quals = \
                        mutate_read(new_seg, new_seg_name, out_error, seg_error_dict_list[seg_idx],
                                    seg_error_count_list[seg_idx], basecaller, read_type, fastq, kmer_bias)

                    if kmer_bias:
                        seg_mutated, seg_base_quals = mutate_homo(seg_mutated, seg_base_quals, kmer_bias, basecaller,
                                                                  None)
                    new_read_name += new_seg_name + ';'
                    read_mutated += seg_mutated
                    base_quals.extend(seg_base_quals)
                    if seg_idx < len(gap_list):
                        read_mutated += gap_list[seg_idx]
                        base_quals.extend(gap_base_qual_list[seg_idx])
                new_read_name = new_read_name + "aligned_" + str(sequence_index)
                if len(seg_length_list) > 1:
                    new_read_name += "_chimeric"
                if fastq:  # Get head/tail qualities and add to base_quals
                    ht_quals = mm.trunc_lognorm_rvs("ht", read_type, basecaller, head + tail).tolist()
                    base_quals = ht_quals[:head] + base_quals + ht_quals[head:]

            # Add head and tail region
            read_mutated = ''.join(np.random.choice(BASES, head)) + read_mutated + \
                           ''.join(np.random.choice(BASES, tail))

            # Reverse complement half of the reads
            p = random.random()
            if p > strandness_rate:
                read_mutated = reverse_complement(read_mutated)
                new_read_name += "_R"
                base_quals.reverse()
            else:
                new_read_name += "_F"

            if per:
                out_reads.write(id_begin + new_read_name + "_0_" + str(sum(ref_length_list)) + "_0" + '\n')
            else:
                out_reads.write(id_begin + new_read_name + "_" + str(head) + "_" +
                                ";".join(str(x) for x in ref_length_list) + "_" + str(tail) + '\n')
            out_reads.write(read_mutated + '\n')

            if fastq:
                out_reads.write("+\n")
                out_quals = "".join([chr(qual + 33) for qual in base_quals])
                out_reads.write(out_quals + "\n")

            if (sequence_index + 1) % 100 == 0:
                sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Number of reads simulated >> " +
                                 str(sequence_index + 1) + "\r")
                # +1 is just to ignore the zero index by python
                sys.stdout.flush()

            passed += 1

        remaining_reads = num_simulate - passed
        remaining_segments = num_segment[passed:]
        remaining_gaps = remaining_segments - 1

    out_reads.close()
    out_error.close()


def simulation_unaligned(dna_type, min_l, max_l, median_l, sd_l, out_reads, basecaller, read_type, fastq,
                         num_simulate, uracil):
    out_reads = open(out_reads, "w")

    if fastq:
        id_begin = "@"
    else:
        id_begin = ">"

    remaining_reads = num_simulate
    passed = 0
    while remaining_reads > 0:
        # if the median length and sd is set, use log normal distribution for simulation
        ref_l = get_length_kde(kde_unaligned, remaining_reads) if median_l is None else \
            np.random.lognormal(np.log(median_l), sd_l, remaining_reads)

        for j in xrange(len(ref_l)):
            # check if the total length fits the criteria
            ref = int(ref_l[j])

            unaligned, middle_ref, error_dict, error_count = unaligned_error_list(ref, error_par)

            if unaligned < min_l or unaligned > max_l:
                continue

            with total_simulated.get_lock():
                sequence_index = total_simulated.value
                total_simulated.value += 1

            new_read, new_read_name = extract_read(dna_type, middle_ref)
            new_read_name = new_read_name + "_unaligned_" + str(sequence_index)
            # Change lowercase to uppercase and replace N with any base
            new_read = case_convert(new_read)
            # no quals returned here since unaligned quals are not based on mis/ins/match qual distributions
            read_mutated, _ = mutate_read(new_read, new_read_name, None, error_dict, error_count, basecaller,
                                          read_type, False, False)

            if fastq:
                base_quals = mm.trunc_lognorm_rvs("unaligned", read_type, basecaller, len(read_mutated)).tolist()
            else:
                base_quals = []

            # Reverse complement some of the reads based on direction information
            p = random.random()
            if p > strandness_rate:
                read_mutated = reverse_complement(read_mutated)
                new_read_name += "_R"
                base_quals.reverse()
            else:
                new_read_name += "_F"

            out_reads.write(id_begin + new_read_name + "_0_" + str(middle_ref) + "_0" + '\n')
            if uracil:
                read_mutated = read_mutated.traslate(trantab)
            out_reads.write(read_mutated + "\n")

            if fastq:
                out_reads.write("+\n")
                out_quals = "".join([chr(qual + 33) for qual in base_quals])
                out_reads.write(out_quals + "\n")

            if (sequence_index + 1) % 100 == 0:
                sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Number of reads simulated >> " +
                                 str(sequence_index + 1) + "\r")
                # +1 is just to ignore the zero index by python
                sys.stdout.flush()
            passed += 1

        remaining_reads = num_simulate - passed
    out_reads.close()


def simulation_gap(ref, basecaller, read_type, dna_type, fastq):
    if ref == 0:
        return '', []

    unaligned, middle_ref, error_dict, error_count = unaligned_error_list(ref, error_par)
    new_gap, new_gap_name = extract_read(dna_type, middle_ref)
    new_gap = case_convert(new_gap)

    # no quals returned here since unaligned quals are not based on mis/ins/match qual distributions
    gap_mutated, _ = mutate_read(new_gap, new_gap_name, None, error_dict, error_count, basecaller, read_type, False,
                                 False)

    if fastq:
        base_quals = mm.trunc_lognorm_rvs("unaligned", read_type, basecaller, len(gap_mutated)).tolist()
    else:
        base_quals = []

    return gap_mutated, base_quals


def simulation(mode, out, dna_type, per, kmer_bias, basecaller, read_type, max_l, min_l, num_threads, fastq,
               median_l=None, sd_l=None, model_ir=False, uracil=False, polya=None, chimeric=False,
               truncation_mode='ont_r9'):
    global total_simulated  # Keeps track of number of reads that have been simulated so far
    total_simulated = mp.Value("i", 0, lock=True)

    # Start simulation
    sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Start simulation of aligned reads\n")
    sys.stdout.flush()
    if fastq:
        ext = ".fastq"
    else:
        ext = ".fasta"

    procs = []
    aligned_subfiles = []
    error_subfiles = []
    num_simulate = int(number_aligned / num_threads)

    for i in range(num_threads):
        aligned_subfile = out + "_aligned_reads{}".format(i) + ext
        error_subfile = out + "_error_profile{}".format(i)
        aligned_subfiles.append(aligned_subfile)
        error_subfiles.append(error_subfile)
        if i == num_threads - 1:  # Last process will simulate the remaining reads
            num_simulate += number_aligned % num_threads

        if mode == "genome":
            p = mp.Process(target=simulation_aligned_genome,
                           args=(dna_type, min_l, max_l, median_l, sd_l, aligned_subfile, error_subfile,
                                 kmer_bias, basecaller, read_type, fastq, num_simulate, per, chimeric))
            procs.append(p)
            p.start()

        elif mode == "metagenome":
            p = mp.Process(target=simulation_aligned_metagenome,
                           args=(min_l, max_l, median_l, sd_l, aligned_subfile, error_subfile, kmer_bias,
                                 basecaller, read_type, fastq, num_simulate, per, chimeric))
            procs.append(p)
            p.start()

        else:
            p = mp.Process(target=simulation_aligned_transcriptome,
                           args=(model_ir, aligned_subfile, error_subfile, kmer_bias, basecaller, read_type,
                                 num_simulate, polya, fastq, per, uracil, truncation_mode))
            procs.append(p)
            p.start()

    for p in procs:
        p.join()

    sys.stdout.write('\n')  # Start a new line because the "Number of reads simulated" is not returned
    # Merging aligned reads subfiles and error subfiles
    with open(out + "_aligned_reads" + ext, 'w') as out_aligned_reads:
        for fname in aligned_subfiles:
            with open(fname) as infile:
                out_aligned_reads.write(infile.read())
    for fname in aligned_subfiles:
        os.remove(fname)

    with open(out + "_aligned_error_profile", 'w') as out_error:
        out_error.write("Seq_name\tSeq_pos\terror_type\terror_length\tref_base\tseq_base\n")
        for fname in error_subfiles:
            with open(fname) as infile:
                out_error.write(infile.read())
    for fname in error_subfiles:
        os.remove(fname)

    # Simulate unaligned reads, if per, number_unaligned = 0, taken care of in read_ecdf
    if not per:
        sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Start simulation of random reads\n")
        sys.stdout.flush()
        unaligned_subfiles = []
        # unaligned_error_subfiles = []
        num_simulate = int(number_unaligned / num_threads)
        for i in range(num_threads):
            unaligned_subfile = out + "_unaligned_reads{}".format(i) + ext
            # unaligned_error_subfile = out + "_unaligned_error_profile{}".format(i) + ext
            unaligned_subfiles.append(unaligned_subfile)
            # unaligned_error_subfiles.append(unaligned_error_subfile)
            if i == num_threads - 1:
                num_simulate += number_unaligned % num_threads

            # Dividing number of unaligned reads that need to be simulated amongst the number of processes
            p = mp.Process(target=simulation_unaligned,
                           args=(dna_type, min_l, max_l, median_l, sd_l, unaligned_subfile,
                                 basecaller, read_type, fastq, num_simulate, uracil))
            procs.append(p)
            p.start()

        for p in procs:
            p.join()

        sys.stdout.write('\n')  # Start a new line because the "Number of reads simulated" is not returned
        # Merging unaligned reads subfiles and error subfiles
        with open(out + "_unaligned_reads" + ext, 'w') as out_unaligned_reads:
            for fname in unaligned_subfiles:
                with open(fname) as infile:
                    out_unaligned_reads.write(infile.read())
        for fname in unaligned_subfiles:
            os.remove(fname)


def reverse_complement(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    seq_list = list(seq)
    reverse_seq_list = reversed([comp.get(base, base) for base in seq_list])
    reverse_seq = ''.join(reverse_seq_list)
    return reverse_seq


def extract_read_trx(key, length, trx_has_polya, ref_pos=-1, buffer=10):
    # buffer: if the extracted read is within 10 base to the reference 3' end, it's considered as reaching to the end
    if ref_pos == -1:
        ref_pos = random.randint(0, seq_len[key] - length)
    template = seq_dict[key]
    # if ref_pos + length > len(template):
    #    ref_pos = len(template) - length
    assert ref_pos + length <= len(template)
    new_read = template[ref_pos: ref_pos + length]

    retain_polya = False
    if trx_has_polya and ref_pos + length + buffer >= seq_len[key]:  # Read reaches end of transcript
        retain_polya = True
    return new_read, ref_pos, retain_polya


def extract_read(dna_type, length, s=None):
    if dna_type == "transcriptome":
        while True:
            key = random.choice(list(seq_len.keys()))  # added "list" thing to be compatible with Python v3
            if length < seq_len[key]:
                ref_pos = random.randint(0, seq_len[key] - length)
                new_read = seq_dict[key][ref_pos: ref_pos + length]
                new_read_name = key + "_" + str(ref_pos)
                break
        return new_read, new_read_name
    elif dna_type == "metagenome":
        while True:
            if not s or length > max(seq_len[s].values()):  # if the length is too long, change to a different species
                s = random.choice(list(seq_len.keys()))  # added "list" thing to be compatible with Python v3
            key = random.choice(list(seq_len[s].keys()))
            if length < seq_len[s][key]:
                if dict_dna_type[s][key] == "circular":
                    ref_pos = random.randint(0, seq_len[s][key])
                    if length + ref_pos > seq_len[s][key]:
                        new_read = seq_dict[s][key][ref_pos:]
                        new_read = new_read + seq_dict[s][key][0: length - seq_len[s][key] + ref_pos]
                    else:
                        new_read = seq_dict[s][key][ref_pos: ref_pos + length]
                else:
                    ref_pos = random.randint(0, seq_len[s][key] - length)
                    new_read = seq_dict[s][key][ref_pos: ref_pos + length]
                new_read_name = s + '-' + key + "_" + str(ref_pos)
                break
        return new_read, new_read_name
    else:
        # Extract the aligned region from reference
        if dna_type == "circular":
            ref_pos = random.randint(0, genome_len)
            chromosome = list(seq_dict.keys())[0]
            new_read_name = chromosome + "_" + str(ref_pos)
            if length + ref_pos <= genome_len:
                new_read = seq_dict[chromosome][ref_pos: ref_pos + length]
            else:
                new_read = seq_dict[chromosome][ref_pos:]
                new_read = new_read + seq_dict[chromosome][0: length - genome_len + ref_pos]
        else:
            # Generate a random number within the size of the genome. Suppose chromosomes are connected
            # tail to head one by one in the order of the dictionary. If the start position fits in one
            # chromosome, but the end position does not, then restart generating random number.
            # This is designed for genomes with multiple chromosomes which varies a lot in lengths

            while True:
                new_read = ""
                ref_pos = random.randint(0, genome_len)
                for key in seq_len:
                    if ref_pos + length <= seq_len[key]:
                        new_read = seq_dict[key][ref_pos: ref_pos + length]
                        new_read_name = key + "_" + str(ref_pos)
                        break
                    elif ref_pos < seq_len[key]:
                        break
                    else:
                        ref_pos -= seq_len[key]
                if new_read != "":
                    break
        return new_read, new_read_name


def unaligned_error_list(m_ref, error_p):
    l_new = m_ref
    e_dict = {}
    error_rate = {(0, 0.4): "match", (0.4, 0.7): "mis", (0.7, 0.85): "ins", (0.85, 1): "del"}
    pos = 0
    middle_ref = m_ref
    last_is_ins = False
    e_count = {"match": 0, "mis": 0, "ins": 0}  # Not used; added to be consistent with error_list()
    if m_ref == 0:
        return l_new, middle_ref, e_dict, e_count
    while pos < middle_ref:
        p = random.random()
        for k_error in error_rate.keys():
            if k_error[0] <= p < k_error[1]:
                error_type = error_rate[k_error]
                break

        if error_type == "match":
            step = 1

        elif error_type == "mis":
            step = mm.pois_geom(error_p["mis"][0], error_p["mis"][2], error_p["mis"][3])
            e_dict[pos] = ["mis", step]

        elif error_type == "ins":
            step = mm.wei_geom(error_p["ins"][0], error_p["ins"][1], error_p["ins"][2], error_p["ins"][3])
            if last_is_ins:
                e_dict[pos + 0.1][1] += step
            else:
                e_dict[pos + 0.1] = ["ins", step]
                last_is_ins = True
            l_new += step

        else:
            step = mm.wei_geom(error_p["del"][0], error_p["del"][1], error_p["del"][2], error_p["del"][3])
            e_dict[pos] = ["del", step]
            l_new -= step

        if error_type != "ins":
            pos += step
            last_is_ins = False

        if pos > middle_ref:
            l_new += pos - middle_ref
            middle_ref = pos

    return l_new, middle_ref, e_dict, e_count


def error_list(m_ref, m_model, m_ht_list, error_p, trans_p, fastq):
    # l_old is the original length, and l_new is used to control the new length after introducing errors
    l_new = m_ref
    pos = 0
    e_dict = {}
    middle_ref = m_ref
    prev_error = "start"
    e_count = {"mis": 0, "ins": 0, "match": 0}

    # The first match come from m_ht_list
    p = random.random()
    k1 = list(m_ht_list.keys())[0]
    for k2, v2 in m_ht_list[k1].items():
        if k2[0] < p <= k2[1]:
            prev_match = int(np.floor((p - k2[0]) / (k2[1] - k2[0]) * (v2[1] - v2[0]) + v2[0]))
            if prev_match < 2:
                prev_match = 2
    pos += prev_match
    if fastq:
        if prev_match > middle_ref: 
            e_count["match"] += middle_ref
        else:
            e_count["match"] += prev_match

    # Select an error, then the step size, and then a match and so on so forth.
    while pos < middle_ref:
        # pick the error based on Markov chain
        p = random.random()
        for k in trans_p[prev_error].keys():
            if k[0] <= p < k[1]:
                error = trans_p[prev_error][k]
                break

        if error == "mis":
            step = mm.pois_geom(error_p[error][0], error_p[error][2], error_p[error][3])
            if pos + step > middle_ref:
                continue
        elif error == "ins":
            step = mm.wei_geom(error_p[error][0], error_p[error][1], error_p[error][2], error_p[error][3])
            l_new += step
        else:
            step = mm.wei_geom(error_p[error][0], error_p[error][1], error_p[error][2], error_p[error][3])
            if pos + step > middle_ref:
                continue
            l_new -= step

        if error != "ins":
            e_dict[pos] = [error, step]
            pos += step
            if pos > middle_ref:
                l_new += pos - middle_ref
                middle_ref = pos
        else:
            e_dict[pos - 0.5] = [error, step]

        prev_error = error

        if fastq:
            if error == "mis" or error == "ins":
                e_count[error] += step

        # Randomly select a match length
        for k1 in m_model.keys():
            if k1[0] <= prev_match < k1[1]:
                break
        p = random.random()
        for k2, v2 in m_model[k1].items():
            if k2[0] < p <= k2[1]:
                step = int(np.floor((p - k2[0]) / (k2[1] - k2[0]) * (v2[1] - v2[0]) + v2[0]))
                break
        # there are no two 0 base matches together
        if prev_match == 0 and step == 0:
            step = 1

        if pos + step > middle_ref:
            step = middle_ref - pos

        prev_match = step

        if fastq:
            e_count["match"] += step

        if pos + prev_match > middle_ref:
            l_new += pos + prev_match - middle_ref
            middle_ref = pos + prev_match

        pos += prev_match
        if prev_match == 0:
            prev_error += "0"

    assert middle_ref == m_ref
    return l_new, middle_ref, e_dict, e_count


def mutate_read(read, read_name, error_log, e_dict, e_count, basecaller, read_type, fastq, k):
    if k:  # First remove any errors that land in hp regions
        pattern = "A{" + re.escape(str(k)) + ",}|C{" + re.escape(str(k)) + ",}|G{" + re.escape(str(k)) + ",}|T{" + \
                  re.escape(str(k)) + ",}"

        hp_pos = []  # [[start, end], ...]
        for match in re.finditer(pattern, read):
            hp_pos.append([match.start(), match.end()])

        new_e_dict = {}
        for err_start in e_dict.keys():
            err = e_dict[err_start][0]
            err_end = err_start + e_dict[err_start][1]
            hp_err = False

            for hp in hp_pos:
                hp_start = hp[0]
                hp_end = hp[1]
                if not (hp_end <= err_start or err_end <= hp_start):  # Lands in hp; remove err
                    hp_err = True
                    if fastq:  # Convert err qual to match qual
                        if err != "ins":
                            e_count["match"] += e_dict[err_start][1]
                        if err != "del":
                            e_count[err] -= e_dict[err_start][1]
                    break

            if not hp_err:
                new_e_dict[err_start] = [err, e_dict[err_start][1]]

    else:
        new_e_dict = e_dict

    if fastq:  # Sample base qualities for mis/ins/match
        mis_quals = mm.trunc_lognorm_rvs("mis", read_type, basecaller, e_count["mis"]).tolist()
        ins_quals = mm.trunc_lognorm_rvs("ins", read_type, basecaller, e_count["ins"]).tolist()
        match_quals = mm.trunc_lognorm_rvs("match", read_type, basecaller, e_count["match"]).tolist()

    # Mutate read
    quals = []
    prev = len(read)
    for key in sorted(new_e_dict.keys(), reverse=True):
        val = new_e_dict[key]
        key = math.ceil(key)  # Ceil instead of round for consistent match calculations during base qual sim
        err_quals = []

        if val[0] == "mis":
            event_len = min(val[1], len(read) - key)
            ref_base = read[key: key + event_len]
            new_bases = ""
            # print(len(read), key, val, event_len)
            for i in xrange(event_len):
                tmp_bases = list(BASES)
                # if key + i >= len(read):
                #    print("ALARM: %d" % i)
                tmp_bases.remove(read[key + i])
                # tmp_bases.remove(read[key]) ## Edited this part for testing
                new_base = random.choice(tmp_bases)
                new_bases += new_base
                if fastq:
                    err_quals.append(mis_quals.pop())

            new_read = read[:key] + new_bases + read[key + event_len:]
            err_end = key + event_len

        elif val[0] == "del":
            new_bases = val[1] * "-"
            ref_base = read[key: key + val[1]]
            new_read = read[: key] + read[key + val[1]:]
            err_end = key + val[1]

        elif val[0] == "ins":
            ref_base = val[1] * "-"
            new_bases = ""
            for i in xrange(val[1]):
                new_base = random.choice(BASES)
                new_bases += new_base
                if fastq:
                    err_quals.append(ins_quals.pop())
            new_read = read[:key] + new_bases + read[key:]
            err_end = key

        if fastq:
            if err_end != prev:  # Match after error
                for j in xrange(prev - err_end):
                    quals.append(match_quals.pop())
            quals += err_quals

        read = new_read
        prev = key

        if val[0] != "match" and error_log:
            error_log.write(read_name + "\t" + str(key) + "\t" + val[0] + "\t" + str(val[1]) +
                            "\t" + ref_base + "\t" + new_bases + "\n")

    if fastq:  # Add first match quals
        while len(match_quals) > 0:
            quals.append(match_quals.pop())

    quals.reverse()
    if fastq:
        assert len(quals) == len(read)
    return read, quals


def inflate_abun(original_dict, inflated_species):
    rest_abun = (1 - original_dict[inflated_species]) * abun_inflation
    inflated_prob = 1 - rest_abun

    return inflated_prob


def main():
    global number_aligned, number_unaligned
    parser = argparse.ArgumentParser(
        description=dedent('''
        Simulation step
        -----------------------------------------------------------
        Given error profiles, reference genome, metagenome,
        and/or transcriptome, simulate ONT DNA or RNA reads
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-v', '--version', action='version', version='NanoSim ' + VERSION)
    subparsers = parser.add_subparsers(help="You may run the simulator on genome, transcriptome, or metagenome mode.",
                                       dest='mode', description=dedent('''
        There are two modes in read_analysis.
        For detailed usage of each mode:
            simulator.py mode -h
        -------------------------------------------------------
        '''))

    parser_g = subparsers.add_parser('genome', help="Run the simulator on genome mode")
    parser_g.add_argument('-rg', '--ref_g', help='Input reference genome', required=True)
    parser_g.add_argument('-c', '--model_prefix', help='Location and prefix of error profiles generated from '
                                                       'characterization step (Default = training)',
                          default="training")
    parser_g.add_argument('-o', '--output', help='Output location and prefix for simulated reads (Default = simulated)',
                          default="simulated")
    parser_g.add_argument('-n', '--number', help='Number of reads to be simulated (Default = 20000)', type=int,
                          default=20000)
    parser_g.add_argument('-max', '--max_len', help='The maximum length for simulated reads (Default = Infinity)',
                          type=int, default=float("inf"))
    parser_g.add_argument('-min', '--min_len', help='The minimum length for simulated reads (Default = 50)',
                          type=int, default=50)
    parser_g.add_argument('-med', '--median_len', help='The median read length (Default = None), Note: this simulation '
                                                       'is not compatible with chimeric reads simulation',
                          type=int, default=None)
    parser_g.add_argument('-sd', '--sd_len', help='The standard deviation of read length in log scale (Default = None),'
                                                  ' Note: this simulation is not compatible with chimeric reads '
                                                  'simulation', type=float, default=None)
    parser_g.add_argument('--seed', help='Manually seeds the pseudo-random number generator', type=int, default=None)
    parser_g.add_argument('-k', '--KmerBias', help='Minimum homopolymer length to simulate homopolymer contraction and '
                                                   'expansion events in, a typical k is 6',
                          type=int, default=None)
    parser_g.add_argument('-b', '--basecaller', help='Simulate homopolymers and/or base qualities with respect to '
                                                     'chosen basecaller: albacore, guppy, or guppy-flipflop',
                          choices=["albacore", "guppy", "guppy-flipflop"], default=None)
    parser_g.add_argument('-s', '--strandness', help='Proportion of sense sequences. Overrides the value '
                                                      'profiled in characterization stage. Should be between 0 and 1',
                          type=float, default=None)
    parser_g.add_argument('-dna_type', help='Specify the dna type: circular OR linear (Default = linear)',
                          choices=["linear", "circular"], default="linear")
    parser_g.add_argument('--perfect', help='Ignore error profiles and simulate perfect reads', action='store_true',
                          default=False)
    parser_g.add_argument('--fastq', help='Output fastq files instead of fasta files', action='store_true',
                          default=False)
    parser_g.add_argument('--chimeric', help='Simulate chimeric reads', action='store_true', default=False)
    parser_g.add_argument('-t', '--num_threads', help='Number of threads for simulation (Default = 1)', type=int,
                          default=1)

    parser_t = subparsers.add_parser('transcriptome', help="Run the simulator on transcriptome mode")
    parser_t.add_argument('-rt', '--ref_t', help='Input reference transcriptome', required=True)
    parser_t.add_argument('-rg', '--ref_g', help='Input reference genome, required if intron retention simulation is '
                                                 'on', default='')
    parser_t.add_argument('-e', '--exp', help='Expression profile in the specified format as described in README',
                          required=True)
    parser_t.add_argument('-c', '--model_prefix', help='Location and prefix of error profiles generated from '
                                                       'characterization step (Default = training)',
                          default="training")
    parser_t.add_argument('-o', '--output', help='Output location and prefix for simulated reads (Default = simulated)',
                          default="simulated")
    parser_t.add_argument('-n', '--number', help='Number of reads to be simulated (Default = 20000)', type=int,
                          default=20000)
    parser_t.add_argument('-max', '--max_len', help='The maximum length for simulated reads (Default = Infinity)',
                          type=int, default=float("inf"))
    parser_t.add_argument('-min', '--min_len', help='The minimum length for simulated reads (Default = 50)',
                          type=int, default=50)
    parser_t.add_argument('--seed', help='Manually seeds the pseudo-random number generator', type=int, default=None)
    parser_t.add_argument('-k', '--KmerBias', help='Minimum homopolymer length to simulate homopolymer contraction and '
                                                   'expansion events in, a typical k is 6',
                          type=int, default=None)
    parser_t.add_argument('-b', '--basecaller', help='Simulate homopolymers and/or base qualities with respect to '
                                                     'chosen basecaller: albacore or guppy',
                          choices=["albacore", "guppy"], default=None)
    parser_t.add_argument('-r', '--read_type', help='Simulate homopolymers and/or base qualities with respect to '
                                                    'chosen read type: dRNA, cDNA_1D or cDNA_1D2',
                          choices=["dRNA", "cDNA_1D", "cDNA_1D2"], default=None)
    parser_t.add_argument('-s', '--strandness', help='Proportion of sense sequences. Overrides the value '
                                                      'profiled in characterization stage. Should be between 0 and 1',
                          type=float, default=None)
    parser_t.add_argument('--no_model_ir', help='Ignore simulating intron retention events', action='store_false',
                          default=True)
    parser_t.add_argument('--perfect', help='Ignore profiles and simulate perfect reads', action='store_true',
                          default=False)
    parser_t.add_argument('--polya', help='Simulate polyA tails for given list of transcripts', default=None)
    parser_t.add_argument('--truncation_mode', help='truncation mode: none, pacbio, ont_r9, ont_spatial, curio',
                          default='ont_r9', choices=['none', 'pacbio', 'ont_r9', 'ont_spatial', 'curio', 'custom', 'curio_ctrl', 'curio_se', 'curio_le'])

    parser_t.add_argument('--fastq', help='Output fastq files instead of fasta files', action='store_true',
                          default=False)
    parser_t.add_argument('-t', '--num_threads', help='Number of threads for simulation (Default = 1)', type=int,
                          default=1)
    parser_t.add_argument('--uracil', help='Converts the thymine (T) bases to uracil (U) in the output fasta format',
                          action='store_true', default=False)
    parser_t.add_argument('--aligned_only', action='store_true', default=False,
                          help='Do not add background noise reads')

    parser_mg = subparsers.add_parser('metagenome', help="Run the simulator on metagenome mode")
    parser_mg.add_argument('-gl', '--genome_list', help="Reference metagenome list, tsv file, the first column is "
                                                        "species/strain name, the second column is the reference "
                                                        "genome fasta/fastq file directory", required=True)
    parser_mg.add_argument('-a', '--abun', help="Abundance list, tsv file with header, the abundance of all species in "
                                                "each sample need to sum up to 100. See example in README and provided "
                                                "config files", required=True)
    parser_mg.add_argument('-dl', '--dna_type_list',
                           help="DNA type list, tsv file, the first column is species/strain, "
                                "the second column is the chromosome name, the third column is "
                                "the DNA type: circular OR linear",
                           required=True)
    parser_mg.add_argument('-c', '--model_prefix', help='Location and prefix of error profiles generated from '
                                                        'characterization step (Default = training)',
                           default="training")
    parser_mg.add_argument('-o', '--output',
                           help='Output location and prefix for simulated reads (Default = simulated)',
                           default="simulated")
    parser_mg.add_argument('-max', '--max_len', help='The maximum length for simulated reads (Default = Infinity)',
                           type=int, default=float("inf"))
    parser_mg.add_argument('-min', '--min_len', help='The minimum length for simulated reads (Default = 50)',
                           type=int, default=50)
    parser_mg.add_argument('-med', '--median_len', help='The median read length (Default = None), Note: this simulation'
                                                        'is not compatible with chimeric reads simulation',
                           type=int, default=None)
    parser_mg.add_argument('-sd', '--sd_len',
                           help='The standard deviation of read length in log scale (Default = None), Note: this '
                                'simulation is not compatible with chimeric reads simulation',
                           type=float, default=None)
    parser_mg.add_argument('--seed', help='Manually seeds the pseudo-random number generator', type=int, default=None)
    parser_mg.add_argument('-k', '--KmerBias', help='Minimum homopolymer length to simulate homopolymer contraction and'
                                                    'expansion events in, a typical k is 6',
                          type=int, default=None)
    parser_mg.add_argument('-b', '--basecaller', help='Simulate homopolymers and/or base qualities with respect to '
                                                      'chosen basecaller: albacore, guppy, or guppy-flipflop',
                          choices=["albacore", "guppy", "guppy-flipflop"], default=None)
    parser_mg.add_argument('-s', '--strandness', help='Percentage of antisense sequences. Overrides the value profiled '
                                                      'in characterization stage. Should be between 0 and 1',
                           type=float, default=None)
    parser_mg.add_argument('--perfect', help='Ignore error profiles and simulate perfect reads', action='store_true',
                           default=False)
    parser_mg.add_argument('--abun_var', help='Simulate random variation in abundance values, takes in two values, '
                                              'format: relative_var_low, relative_var_high, Example: -0.5 0.5)',
                           nargs='+', type=float, default=None)
    parser_mg.add_argument('--fastq', help='Output fastq files instead of fasta files', action='store_true',
                           default=False)
    parser_mg.add_argument('--chimeric', help='Simulate chimeric reads', action='store_true', default=False)
    parser_mg.add_argument('-t', '--num_threads', help='Number of threads for simulation (Default = 1)', type=int,
                           default=1)

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.mode == "genome":
        ref_g = args.ref_g
        model_prefix = args.model_prefix
        out = args.output
        number = [args.number]
        max_len = args.max_len
        min_len = args.min_len
        median_len = args.median_len
        sd_len = args.sd_len
        chimeric = args.chimeric
        if args.seed:
            random.seed(int(args.seed))
            np.random.seed(int(args.seed))
        perfect = args.perfect
        kmer_bias = args.KmerBias
        basecaller = args.basecaller
        strandness = args.strandness
        dna_type = args.dna_type
        num_threads = max(args.num_threads, 1)
        fastq = args.fastq

        if kmer_bias and kmer_bias < 0:
            print("\nPlease input proper kmer bias value >= 0\n")
            parser_g.print_help(sys.stderr)
            sys.exit(1)

        if kmer_bias and basecaller is None:
            print("\nPlease input basecaller to simulate homopolymer contraction and expansion events from\n")
            parser_g.print_help(sys.stderr)
            sys.exit(1)

        if strandness and (strandness < 0 or strandness > 1):
            print("\nPlease input proper strandness value between 0 and 1\n")
            parser_g.print_help(sys.stderr)
            sys.exit(1)

        if (median_len and not sd_len) or (sd_len and not median_len):
            sys.stderr.write("\nPlease provide both mean and standard deviation of read length!\n")
            parser_g.print_help(sys.stderr)
            sys.exit(1)

        if max_len < min_len:
            sys.stderr.write("\nMaximum read length must be longer than Minimum read length!\n")
            parser_g.print_help(sys.stderr)
            sys.exit(1)

        if fastq and basecaller is None:
            print("\nPlease input basecaller to simulate base qualities from.\n")
            parser_g.print_help(sys.stderr)
            sys.exit(1)

        if perfect and chimeric:
            print("\nPerfect reads cannot be chimeric\n")
            parser_g.print_help(sys.stderr)
            sys.exit(1)

        if fastq and basecaller == "guppy-flipflop":
            print("\nBase quality simulation isn't supported for guppy-flipflop. Please choose either albacore "
                  "or guppy as the basecaller.\n")
            parser_g.print_help(sys.stderr)
            sys.exit(1)

        print("\nrunning the code with following parameters:\n")
        print("ref_g", ref_g)
        print("model_prefix", model_prefix)
        print("out", out)
        print("number", number)
        print("perfect", perfect)
        print("kmer_bias", kmer_bias)
        print("basecaller", basecaller)
        print("dna_type", dna_type)
        print("strandness", strandness)
        print("sd_len", sd_len)
        print("median_len", median_len)
        print("max_len", max_len)
        print("min_len", min_len)
        print("fastq", fastq)
        print("chimeric", chimeric)
        print("num_threads", num_threads)

        sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ': ' + ' '.join(sys.argv) + '\n')
        sys.stdout.flush()

        dir_name = os.path.dirname(out)
        if dir_name != '':
            call("mkdir -p " + dir_name, shell=True)

        read_profile(ref_g, number, model_prefix, perfect, args.mode, strandness, dna_type=dna_type, chimeric=chimeric,
                     aligned_only=args.aligned_only)

        if median_len and sd_len:
            sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Simulating read length with log-normal distribution\n")
            sys.stdout.flush()

        number_aligned = number_aligned_l[0]
        number_unaligned = number_unaligned_l[0]
        max_len = min(max_len, max_chrom)
        simulation(args.mode, out, dna_type, perfect, kmer_bias, basecaller, "DNA", max_len, min_len, num_threads,
                   fastq, median_len, sd_len, chimeric=chimeric)

    elif args.mode == "transcriptome":
        ref_g = args.ref_g
        ref_t = args.ref_t
        exp = args.exp
        model_prefix = args.model_prefix
        out = args.output
        number = [args.number]
        max_len = args.max_len
        min_len = args.min_len
        kmer_bias = args.KmerBias
        if args.seed:
            random.seed(int(args.seed))
            np.random.seed(int(args.seed))
        basecaller = args.basecaller
        read_type = args.read_type
        strandness = args.strandness
        perfect = args.perfect
        model_ir = args.no_model_ir
        dna_type = "transcriptome"
        polya = args.polya
        uracil = args.uracil
        num_threads = max(args.num_threads, 1)
        fastq = args.fastq
        truncation_mode = args.truncation_mode

        if kmer_bias and kmer_bias < 0:
            print("\nPlease input proper kmer bias value >= 0\n")
            parser_t.print_help(sys.stderr)
            sys.exit(1)

        if kmer_bias and (basecaller is None or read_type is None):
            print("\nPlease input basecaller and read_type to simulate homopolymer contraction and expansion events "
                  "from\n")
            parser_t.print_help(sys.stderr)
            sys.exit(1)

        if strandness and (strandness < 0 or strandness > 1):
            print("\nPlease input proper strandness value between 0 and 1\n")
            parser_t.print_help(sys.stderr)
            sys.exit(1)

        if max_len < min_len:
            sys.stderr.write("\nMaximum read length must be longer than Minimum read length!\n")
            parser_t.print_help(sys.stderr)
            sys.exit(1)

        if model_ir and ref_g == '':
            sys.stderr.write("\nPlease provide a reference genome to simulate intron retention events!\n")
            parser_t.print_help(sys.stderr)
            sys.exit(1)

        if polya and basecaller is None:
            print("\nPlease input basecaller to simulate polyA tails from.\n")
            parser_t.print_help(sys.stderr)
            sys.exit(1)

        if fastq and (basecaller is None or read_type is None):
            print("\nPlease input basecaller and read_type to simulate base qualities from.\n")
            parser_t.print_help(sys.stderr)
            sys.exit(1)

        print("\nrunning the code with following parameters:\n")
        print("ref_g", ref_g)
        print("ref_t", ref_t)
        print("exp", exp)
        print("model_prefix", model_prefix)
        print("out", out)
        print("number", number)
        print("perfect", perfect)
        print("kmer_bias", kmer_bias)
        print("basecaller", basecaller)
        print("read_type", read_type)
        print("model_ir", model_ir)
        print("dna_type", dna_type)
        print("strandness", strandness)
        print("max_len", max_len)
        print("min_len", min_len)
        print("uracil", uracil)
        print("polya", polya)
        print("truncation_mode", truncation_mode)
        print("fastq", fastq)
        print("num_threads", num_threads)

        sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ': ' + ' '.join(sys.argv) + '\n')
        sys.stdout.flush()

        dir_name = os.path.dirname(out)
        if dir_name != '':
            call("mkdir -p " + dir_name, shell=True)

        read_profile(ref_g, number, model_prefix, perfect, args.mode, strandness, ref_t=ref_t, dna_type="linear",
                     model_ir=model_ir, polya=polya, exp=exp, aligned_only=args.aligned_only)

        number_aligned = number_aligned_l[0]
        number_unaligned = number_unaligned_l[0]
        sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ': Aligned reads count: %d, Unadligned reads count %d\n' %
                         (number_aligned, number_unaligned))
        sys.stdout.flush()
        max_len = min(max_len, max_chrom)
        simulation(args.mode, out, dna_type, perfect, kmer_bias, basecaller, read_type, max_len, min_len, num_threads,
                   fastq, None, None, model_ir, uracil, polya, truncation_mode=truncation_mode)

    elif args.mode == "metagenome":
        genome_list = args.genome_list
        abun = args.abun
        dna_type_list = args.dna_type_list
        model_prefix = args.model_prefix
        out = args.output
        max_len = args.max_len
        min_len = args.min_len
        median_len = args.median_len
        sd_len = args.sd_len
        if args.seed:
            random.seed(int(args.seed))
            np.random.seed(int(args.seed))
        perfect = args.perfect
        kmer_bias = args.KmerBias
        basecaller = args.basecaller
        strandness = args.strandness
        abun_var = args.abun_var
        fastq = args.fastq
        chimeric = args.chimeric
        num_threads = max(args.num_threads, 1)

        if kmer_bias and kmer_bias < 0:
            print("\nPlease input proper kmer bias value >= 0\n")
            parser_mg.print_help(sys.stderr)
            sys.exit(1)

        if kmer_bias and basecaller is None:
            print("\nPlease input basecaller to simulate homopolymer contraction and expansion events from\n")
            parser_mg.print_help(sys.stderr)
            sys.exit(1)

        if strandness and (strandness < 0 or strandness > 1):
            print("\nPlease input proper strandness value between 0 and 1\n")
            parser_mg.print_help(sys.stderr)
            sys.exit(1)

        if (median_len and not sd_len) or (sd_len and not median_len):
            sys.stderr.write("\nPlease provide both mean and standard deviation of read length!\n")
            parser_mg.print_help(sys.stderr)
            sys.exit(1)

        if max_len < min_len:
            sys.stderr.write("\nMaximum read length must be longer than Minimum read length!\n")
            parser_mg.print_help(sys.stderr)
            sys.exit(1)

        if fastq and basecaller is None:
            print("\nPlease input basecaller to simulate base qualities from.\n")
            parser_mg.print_help(sys.stderr)
            sys.exit(1)

        if fastq and basecaller == "guppy-flipflop":
            print("\nBase quality simulation isn't supported for guppy-flipflop. Please choose either albacore "
                  "or guppy as the basecaller.\n")
            parser_mg.print_help(sys.stderr)
            sys.exit(1)

        print("\nrunning the code with following parameters:\n")
        print("genome_list", genome_list)
        print("abun", abun)
        print("dna_type_list", dna_type_list)
        print("model_prefix", model_prefix)
        print("out", out)
        print("perfect", perfect)
        print("kmer_bias", kmer_bias)
        print("basecaller", basecaller)
        print("strandness", strandness)
        print("sd_len", sd_len)
        print("median_len", median_len)
        print("max_len", max_len)
        print("min_len", min_len)
        print("abun_var", abun_var)
        print("fastq", fastq)
        print("chimeric", chimeric)
        print("num_threads", num_threads)

        sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ': ' + ' '.join(sys.argv) + '\n')
        sys.stdout.flush()

        dir_name = os.path.dirname(out)
        if dir_name != '':
            call("mkdir -p " + dir_name, shell=True)

        read_profile(genome_list, [], model_prefix, perfect, args.mode, strandness, dna_type=dna_type_list, abun=abun,
                     chimeric=chimeric, aligned_only=args.aligned_only)

        # Add abundance variation
        global dict_abun, dict_abun_inflated
        for s in range(len(multi_dict_abun)):
            sample = list(multi_dict_abun.keys())[s]
            if abun_var:
                total_len = {}
                for species in multi_dict_abun[sample]:
                    total_len[species] = sum(seq_len[species].values())
                var_low = float(abun_var[0])
                var_high = float(abun_var[1])
                dict_abun = add_abundance_var(multi_dict_abun[sample], total_len, var_low, var_high)
            else:
                dict_abun = multi_dict_abun[sample]

            # when simulating chimeric reads, the source species for succedent segments have an inflated abundance dist
            if chimeric:
                dict_abun_inflated = {}
                for species in dict_abun:
                    dict_abun_inflated[species] = inflate_abun(dict_abun, species)

            sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Simulating sample " + sample + '\n')
            sys.stdout.flush()
            if median_len and sd_len:
                sys.stdout.write(
                    strftime("%Y-%m-%d %H:%M:%S") + ": Simulating read length from log-normal distribution\n")
                sys.stdout.flush()

            number_aligned = number_aligned_l[s]
            number_unaligned = number_unaligned_l[s]
            max_len = min(max_len, max(max_chrom.values()))
            simulation(args.mode, out + "_" + sample, "metagenome", perfect, kmer_bias, basecaller, "DNA", max_len,
                       min_len, num_threads, fastq, median_len, sd_len, chimeric=chimeric)

    sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Finished!\n")
    sys.stdout.close()


if __name__ == "__main__":
    main()

