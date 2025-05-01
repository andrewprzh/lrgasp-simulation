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



TRUNCATION_MODES_OLD = ['ont_r9', 'ont_spatial', 'pacbio', 'none', 'custom', 'curio']
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
    'custom' : [0] * 25 + [1.0] + [0] * 75
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
    'custom' : [1.0] + [0] * 100
}


def calculate_aligned_length_simple(ref_len, truncation_mode='ont_r9'):
    bp5_list = list(range(101))
    bp3_list = list(range(101))
    pro5_list = TRUNCATION_DICT5[truncation_mode]
    pro3_list = TRUNCATION_DICT3[truncation_mode]
    del5 = int(np.random.choice(bp5_list, p=pro5_list) * ref_len / 100.0)
    del3 = int(np.random.choice(bp3_list, p=pro3_list) * ref_len / 100.0)
    new_len = max(min(50, ref_len - 1), ref_len - del3 - del5 - 1)
    start_pos = max(0, min(del5 - 1, ref_len - new_len - 1))
#    new_len = min(50, ref_len)
#    start_pos = max(0, ref_len - new_len)
    return new_len, start_pos


TR3_STEPS = 100
TR5_STEPS = 10
TR3 = list(range(TR3_STEPS))
TR5 = list(range(TR5_STEPS))
TR_PERC3 = [float(x)/TR3_STEPS for x in TR3]
TR_PERC5 = [float(x)/TR5_STEPS for x in TR5]

TRUNCATION_DICT = {
    'none': {math.inf: ([1.0] + [0] * (TR3_STEPS - 1), [[1.0] + [0] * (TR5_STEPS - 1) for _ in range(TR3_STEPS)])},
    'pacbio': None,
    'ont_spatial': None,
    'ont_r9': None,
    'curio_ctrl':
{500: ([0.5872816810938122, 0.044015538494478944, 0.011256232220964732, 0.0066069189123053864, 0.004374025020646622, 0.0026917077050133058, 0.0024470070045575507, 0.0011929159147218058, 0.0019576056036460406, 0.002171718716544826, 0.0011623283271648365, 0.0014376166151775609, 0.0011623283271648365, 0.000795277276481204, 0.0008258648640381733, 0.0009788028018230203, 0.0025081821796714894, 0.0018046676658611935, 0.0015599669654054384, 0.0018964304285321017, 0.0032422842810387546, 0.07441960052610651, 0.007096320313216897, 0.003425809806380571, 0.000642339338696357, 0.0008564524515951427, 0.0010705655644939283, 0.010705655644939284, 0.0014682042027345304, 0.0011317407396078671, 0.004221087082861775, 0.0011011531520508978, 0.0013458538525066529, 0.0040987367326338975, 0.0007341021013672652, 0.03168874070902028, 0.0023552442418866425, 0.002814058055241183, 0.0009482152142660508, 0.0011011531520508978, 0.0003670510506836326, 0.0004894014009115101, 0.000397638638240602, 0.0005505765760254489, 0.0004588138133545407, 0.0003364634631266632, 0.0004588138133545407, 0.0003670510506836326, 0.0004894014009115101, 0.0007035145138102958, 0.0005199889884684795, 0.0005505765760254489, 0.0011011531520508978, 0.13476891077600708, 0.003915211207292081, 0.0017434924907472548, 0.0021105435414308874, 0.0005505765760254489, 0.0009176276267090814, 0.0008258648640381733, 0.0008564524515951427, 0.0009788028018230203, 0.000795277276481204, 0.0001835255253418163, 0.00042822622579757137, 0.0012235035022787753, 0.0001835255253418163, 0.0003670510506836326, 0.00024470070045575507, 0.000397638638240602, 0.0008870400391521121, 0.0007646896889242345, 0.000397638638240602, 0.000642339338696357, 0.003701098094393295, 0.0016211421405193773, 0.0011929159147218058, 0.00021411311289878569, 0.00012235035022787753, 0.0, 6.117517511393877e-05, 6.117517511393877e-05, 3.0587587556969384e-05, 9.176276267090815e-05, 0.00015293793778484692, 0.0, 9.176276267090815e-05, 3.0587587556969384e-05, 0.0, 6.117517511393877e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[[0.28328125, 0.13223958333333333, 0.1446875, 0.07729166666666666, 0.09895833333333333, 0.09473958333333334, 0.0875, 0.05, 0.030729166666666665, 0.0005729166666666667], [0.20152883947185546, 0.12369701181375956, 0.11674774148714386, 0.11883252258512857, 0.12786657400972898, 0.12786657400972898, 0.0965948575399583, 0.05350938151494093, 0.03335649756775538, 0.0], [0.41032608695652173, 0.059782608695652176, 0.10597826086956522, 0.07065217391304347, 0.057065217391304345, 0.09782608695652174, 0.15217391304347827, 0.035326086956521736, 0.010869565217391304, 0.0], [0.2037037037037037, 0.06018518518518518, 0.13425925925925927, 0.09722222222222222, 0.11574074074074074, 0.1111111111111111, 0.21296296296296297, 0.041666666666666664, 0.023148148148148147, 0.0], [0.4965034965034965, 0.06293706293706294, 0.04895104895104895, 0.06293706293706294, 0.06993006993006994, 0.04895104895104895, 0.16083916083916083, 0.04895104895104895, 0.0, 0.0], [0.29545454545454547, 0.13636363636363635, 0.13636363636363635, 0.045454545454545456, 0.125, 0.056818181818181816, 0.13636363636363635, 0.06818181818181818, 0.0, 0.0], [0.225, 0.05, 0.275, 0.1125, 0.1375, 0.0625, 0.125, 0.0125, 0.0, 0.0], [0.46153846153846156, 0.20512820512820512, 0.10256410256410256, 0.02564102564102564, 0.05128205128205128, 0.02564102564102564, 0.10256410256410256, 0.02564102564102564, 0.0, 0.0], [0.359375, 0.09375, 0.0625, 0.046875, 0.328125, 0.03125, 0.046875, 0.03125, 0.0, 0.0], [0.5633802816901409, 0.08450704225352113, 0.08450704225352113, 0.056338028169014086, 0.056338028169014086, 0.04225352112676056, 0.11267605633802817, 0.0, 0.0, 0.0], [0.5263157894736842, 0.07894736842105263, 0.15789473684210525, 0.07894736842105263, 0.07894736842105263, 0.05263157894736842, 0.0, 0.02631578947368421, 0.0, 0.0], [0.6595744680851063, 0.1702127659574468, 0.06382978723404255, 0.02127659574468085, 0.0, 0.0851063829787234, 0.0, 0.0, 0.0, 0.0], [0.3684210526315789, 0.07894736842105263, 0.13157894736842105, 0.10526315789473684, 0.21052631578947367, 0.05263157894736842, 0.02631578947368421, 0.02631578947368421, 0.0, 0.0], [0.6153846153846154, 0.0, 0.11538461538461539, 0.07692307692307693, 0.11538461538461539, 0.07692307692307693, 0.0, 0.0, 0.0, 0.0], [0.6296296296296297, 0.037037037037037035, 0.1111111111111111, 0.07407407407407407, 0.07407407407407407, 0.0, 0.07407407407407407, 0.0, 0.0, 0.0], [0.6875, 0.09375, 0.0625, 0.03125, 0.03125, 0.03125, 0.0625, 0.0, 0.0, 0.0], [0.8658536585365854, 0.06097560975609756, 0.024390243902439025, 0.0, 0.036585365853658534, 0.012195121951219513, 0.0, 0.0, 0.0, 0.0], [0.5932203389830508, 0.11864406779661017, 0.11864406779661017, 0.05084745762711865, 0.06779661016949153, 0.05084745762711865, 0.0, 0.0, 0.0, 0.0], [0.47058823529411764, 0.17647058823529413, 0.1568627450980392, 0.058823529411764705, 0.09803921568627451, 0.0392156862745098, 0.0, 0.0, 0.0, 0.0], [0.5645161290322581, 0.14516129032258066, 0.1774193548387097, 0.016129032258064516, 0.06451612903225806, 0.016129032258064516, 0.016129032258064516, 0.0, 0.0, 0.0], [0.4056603773584906, 0.2830188679245283, 0.22641509433962265, 0.03773584905660377, 0.03773584905660377, 0.009433962264150943, 0.0, 0.0, 0.0, 0.0], [0.458281956432388, 0.24373201808466913, 0.24496506370735718, 0.006987258528565557, 0.04192355117139334, 0.004110152075626798, 0.0, 0.0, 0.0, 0.0], [0.49137931034482757, 0.2413793103448276, 0.21551724137931033, 0.008620689655172414, 0.03017241379310345, 0.01293103448275862, 0.0, 0.0, 0.0, 0.0], [0.4732142857142857, 0.17857142857142858, 0.25, 0.05357142857142857, 0.0, 0.044642857142857144, 0.0, 0.0, 0.0, 0.0], [0.42857142857142855, 0.23809523809523808, 0.2857142857142857, 0.0, 0.0, 0.0, 0.047619047619047616, 0.0, 0.0, 0.0], [0.42857142857142855, 0.21428571428571427, 0.14285714285714285, 0.07142857142857142, 0.10714285714285714, 0.03571428571428571, 0.0, 0.0, 0.0, 0.0], [0.5428571428571428, 0.11428571428571428, 0.08571428571428572, 0.14285714285714285, 0.05714285714285714, 0.05714285714285714, 0.0, 0.0, 0.0, 0.0], [0.4828571428571429, 0.22857142857142856, 0.24857142857142858, 0.017142857142857144, 0.02, 0.002857142857142857, 0.0, 0.0, 0.0, 0.0], [0.5625, 0.14583333333333334, 0.1875, 0.0625, 0.041666666666666664, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5945945945945946, 0.10810810810810811, 0.16216216216216217, 0.02702702702702703, 0.08108108108108109, 0.02702702702702703, 0.0, 0.0, 0.0, 0.0], [0.47101449275362317, 0.1956521739130435, 0.30434782608695654, 0.014492753623188406, 0.007246376811594203, 0.007246376811594203, 0.0, 0.0, 0.0, 0.0], [0.4444444444444444, 0.16666666666666666, 0.3888888888888889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.18181818181818182, 0.18181818181818182, 0.06818181818181818, 0.06818181818181818, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8432835820895522, 0.05970149253731343, 0.04477611940298507, 0.04477611940298507, 0.007462686567164179, 0.0, 0.0, 0.0, 0.0, 0.0], [0.625, 0.125, 0.08333333333333333, 0.125, 0.041666666666666664, 0.0, 0.0, 0.0, 0.0, 0.0], [0.013513513513513514, 0.0019305019305019305, 0.006756756756756757, 0.0, 0.0, 0.9777992277992278, 0.0, 0.0, 0.0, 0.0], [0.2077922077922078, 0.03896103896103896, 0.1038961038961039, 0.012987012987012988, 0.09090909090909091, 0.5454545454545454, 0.0, 0.0, 0.0, 0.0], [0.16304347826086957, 0.021739130434782608, 0.010869565217391304, 0.021739130434782608, 0.16304347826086957, 0.6195652173913043, 0.0, 0.0, 0.0, 0.0], [0.3225806451612903, 0.0, 0.03225806451612903, 0.0, 0.1935483870967742, 0.45161290322580644, 0.0, 0.0, 0.0, 0.0], [0.3611111111111111, 0.05555555555555555, 0.1111111111111111, 0.027777777777777776, 0.0, 0.4444444444444444, 0.0, 0.0, 0.0, 0.0], [0.5833333333333334, 0.08333333333333333, 0.16666666666666666, 0.08333333333333333, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0], [0.625, 0.125, 0.0, 0.1875, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0], [0.5384615384615384, 0.15384615384615385, 0.15384615384615385, 0.15384615384615385, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6666666666666666, 0.05555555555555555, 0.16666666666666666, 0.05555555555555555, 0.05555555555555555, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4666666666666667, 0.2, 0.2, 0.13333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5454545454545454, 0.18181818181818182, 0.0, 0.09090909090909091, 0.18181818181818182, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6, 0.06666666666666667, 0.13333333333333333, 0.13333333333333333, 0.06666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 0.16666666666666666, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4375, 0.25, 0.0625, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4782608695652174, 0.2608695652173913, 0.13043478260869565, 0.13043478260869565, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47058823529411764, 0.23529411764705882, 0.23529411764705882, 0.058823529411764705, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6111111111111112, 0.1111111111111111, 0.2777777777777778, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19444444444444445, 0.6944444444444444, 0.1111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.003631411711302769, 0.9464366772582842, 0.04993191103041307, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.046875, 0.90625, 0.046875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.14035087719298245, 0.8245614035087719, 0.03508771929824561, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2753623188405797, 0.6521739130434783, 0.057971014492753624, 0.014492753623188406, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5555555555555556, 0.4444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36666666666666664, 0.5666666666666667, 0.06666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.07407407407407407, 0.7407407407407407, 0.18518518518518517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4642857142857143, 0.5, 0.03571428571428571, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21875, 0.75, 0.03125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.46153846153846156, 0.038461538461538464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.3333333333333333, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8571428571428571, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.175, 0.8, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.6666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.5833333333333334, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6153846153846154, 0.3076923076923077, 0.07692307692307693, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3793103448275862, 0.6206896551724138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.32, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.38095238095238093, 0.6190476190476191, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0743801652892562, 0.9256198347107438, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16981132075471697, 0.8301886792452831, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1282051282051282, 0.8717948717948718, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7142857142857143, 0.2857142857142857, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
1000: ([0.7001150687797479, 0.027437179171056467, 0.011559181965583974, 0.010273998191776318, 0.009235390374571294, 0.009272750367995936, 0.0064744868604903125, 0.0043076072418611255, 0.009048590407448088, 0.004053559286573565, 0.003844343323395575, 0.006347462882846533, 0.005488183034079786, 0.003956423303669499, 0.0034744793884916277, 0.0035006313838888766, 0.005667511002518064, 0.006403502872983495, 0.005405991048545575, 0.0034707433891491635, 0.004845591147175958, 0.011943989897857778, 0.003448327393094379, 0.004486935210299403, 0.003007479470683613, 0.002734751518683733, 0.002491911561423565, 0.006104622925586365, 0.004109599276710527, 0.004337495236600838, 0.003934007307614714, 0.0063213108874492835, 0.0018567916732046656, 0.0024358715712866036, 0.0024470795693139957, 0.006728534815777873, 0.0023760955818071775, 0.002734751518683733, 0.00428519124580634, 0.002719807521313876, 0.002417191574574283, 0.0021780876166565795, 0.0017708636883279909, 0.004176847264874882, 0.002009967646245694, 0.005009975118244379, 0.0017559196909581343, 0.0020697436357251203, 0.0012141997863008376, 0.0018642636718895938, 0.0018605276725471296, 0.0015056077350130387, 0.001382319756711723, 0.01796268483856747, 0.0021071036291497613, 0.0018904156672868426, 0.0014047357527665076, 0.0008107118573147131, 0.0011768397928761964, 0.0009601518310132777, 0.001486927738300718, 0.0013636397599994022, 0.0017073516995061008, 0.001867999671232058, 0.0011133278040543065, 0.001382319756711723, 0.0013038637705199763, 0.0012440877810405504, 0.0004333759237258374, 0.0007584078665202155, 0.0007173118737531102, 0.0007808238625750002, 0.0005865518967668662, 0.0006762158809860049, 0.0009825678270680624, 0.0005902878961093303, 0.000646327886246292, 0.000452055920438158, 0.00034371193950669864, 0.0003511839381916269, 0.0002166879618629187, 0.000291407948712201, 0.0002166879618629187, 0.00019427196580813403, 0.00022042396120538284, 0.0002166879618629187, 0.00013449597632870817, 0.00014943997369856464, 0.00024657595660263165, 0.00019800796515059813, 0.0004744719164929427, 0.00020174396449306226, 0.0001419679750136364, 7.845598619174644e-05, 1.4943997369856463e-05, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[[0.28049008255201524, 0.14257432082690757, 0.09403565692086853, 0.11639460610361958, 0.1096922576135157, 0.07404067300970667, 0.0654012604257272, 0.06822948072807995, 0.038837334642496946, 0.0103043271770626], [0.15631808278867101, 0.11587690631808278, 0.11301742919389979, 0.12023420479302832, 0.10620915032679738, 0.1090686274509804, 0.12295751633986927, 0.08946078431372549, 0.05664488017429194, 0.010212418300653595], [0.19553975436328377, 0.11667744020685197, 0.09340659340659341, 0.13994828700711054, 0.10795087265675501, 0.09631544925662573, 0.09663865546218488, 0.083710407239819, 0.06593406593406594, 0.003878474466709761], [0.18727272727272729, 0.1789090909090909, 0.11709090909090909, 0.10472727272727272, 0.10909090909090909, 0.08945454545454545, 0.08472727272727272, 0.0669090909090909, 0.06036363636363636, 0.0014545454545454545], [0.16747572815533981, 0.13794498381877024, 0.16464401294498382, 0.10558252427184465, 0.10072815533980582, 0.12378640776699029, 0.09344660194174757, 0.059870550161812294, 0.043284789644012944, 0.003236245954692557], [0.30016116035455276, 0.11281224818694602, 0.09307010475423046, 0.11281224818694602, 0.10072522159548751, 0.03585817888799355, 0.0427074939564867, 0.14020950846091862, 0.06164383561643835, 0.0], [0.31332948643969993, 0.16387766878245816, 0.09924985574148874, 0.1257934218118869, 0.0738603577611079, 0.06347374495095211, 0.06462781304096941, 0.06520484708597807, 0.030582804385458743, 0.0], [0.2532523850823938, 0.15871639202081528, 0.11795316565481354, 0.1222896790980052, 0.08933217692974849, 0.0797918473547268, 0.07198612315698179, 0.06851691240242845, 0.03816131830008673, 0.0], [0.4240297274979356, 0.1494632535094963, 0.07679603633360858, 0.061932287365813375, 0.08133773740710157, 0.06358381502890173, 0.061932287365813375, 0.06358381502890173, 0.017341040462427744, 0.0], [0.2672811059907834, 0.11705069124423963, 0.07373271889400922, 0.2184331797235023, 0.07096774193548387, 0.09861751152073733, 0.05806451612903226, 0.08294930875576037, 0.012903225806451613, 0.0], [0.27405247813411077, 0.2031098153547133, 0.12147716229348883, 0.11661807580174927, 0.07580174927113703, 0.07094266277939747, 0.08357628765792031, 0.04664723032069971, 0.007774538386783284, 0.0], [0.23366686286050617, 0.20247204237786934, 0.16303708063566805, 0.125956444967628, 0.09535020600353149, 0.051206592113007654, 0.051206592113007654, 0.07474985285462037, 0.002354326074161271, 0.0], [0.21238938053097345, 0.22600408441116407, 0.11572498298162015, 0.12321307011572498, 0.08032675289312458, 0.06943498978897208, 0.05037440435670524, 0.1191286589516678, 0.0034036759700476512, 0.0], [0.251180358829084, 0.11614730878186968, 0.1643059490084986, 0.11803588290840415, 0.10009442870632672, 0.10953729933899906, 0.0679886685552408, 0.07176581680830972, 0.0009442870632672333, 0.0], [0.2752688172043011, 0.14623655913978495, 0.13870967741935483, 0.1064516129032258, 0.10967741935483871, 0.08494623655913978, 0.1010752688172043, 0.03763440860215054, 0.0, 0.0], [0.23372465314834578, 0.25827107790821774, 0.16542155816435433, 0.10779082177161152, 0.08858057630736393, 0.09284951974386339, 0.0288153681963714, 0.02454642475987193, 0.0, 0.0], [0.4495715227422544, 0.15095583388266315, 0.13381674357284112, 0.1041529334212261, 0.06394199077125906, 0.04087013843111404, 0.02109426499670402, 0.035596572181938034, 0.0, 0.0], [0.470828471411902, 0.14935822637106183, 0.1161026837806301, 0.11901983663943991, 0.05542590431738623, 0.044924154025670945, 0.028004667444574097, 0.01633605600933489, 0.0, 0.0], [0.38147892190739463, 0.18659295093296477, 0.13476157567380787, 0.13890808569454044, 0.061506565307532825, 0.038700760193503804, 0.04008293020041465, 0.01796821008984105, 0.0, 0.0], [0.333692142088267, 0.15715823466092574, 0.13240043057050593, 0.14531754574811626, 0.09041980624327234, 0.060279870828848225, 0.05597416576964478, 0.024757804090419805, 0.0, 0.0], [0.3053199691595991, 0.17733230531996916, 0.15497301464919044, 0.13338473400154202, 0.08172706245181187, 0.059367771781033155, 0.06630686198920586, 0.02158828064764842, 0.0, 0.0], [0.4460431654676259, 0.23052862058179543, 0.2108226462308414, 0.030028151391929936, 0.05223647169221145, 0.018454801376290273, 0.009696590553644042, 0.002189552705661558, 0.0, 0.0], [0.4452871072589382, 0.22318526543878656, 0.11592632719393282, 0.07258938244853738, 0.0628385698808234, 0.044420368364030335, 0.030335861321776816, 0.005417118093174431, 0.0, 0.0], [0.3996669442131557, 0.1340549542048293, 0.1340549542048293, 0.16652789342214822, 0.06078268109908409, 0.058284762697751874, 0.03996669442131557, 0.006661115736885929, 0.0, 0.0], [0.5142857142857142, 0.16024844720496895, 0.09192546583850932, 0.11055900621118013, 0.05217391304347826, 0.05093167701863354, 0.017391304347826087, 0.002484472049689441, 0.0, 0.0], [0.4849726775956284, 0.16666666666666666, 0.11065573770491803, 0.09699453551912568, 0.07923497267759563, 0.04371584699453552, 0.017759562841530054, 0.0, 0.0, 0.0], [0.31634182908545727, 0.25487256371814093, 0.15142428785607195, 0.08695652173913043, 0.10944527736131934, 0.053973013493253376, 0.026986506746626688, 0.0, 0.0, 0.0], [0.3543451652386781, 0.19033047735618114, 0.2331701346389229, 0.06731946144430845, 0.0966952264381885, 0.04528763769889841, 0.012851897184822521, 0.0, 0.0, 0.0], [0.41818181818181815, 0.24363636363636362, 0.11636363636363636, 0.07818181818181819, 0.08727272727272728, 0.03636363636363636, 0.02, 0.0, 0.0, 0.0], [0.4427217915590009, 0.17571059431524547, 0.11800172265288544, 0.09043927648578812, 0.12661498708010335, 0.03359173126614987, 0.012919896640826873, 0.0, 0.0, 0.0], [0.4254510921177588, 0.15954415954415954, 0.15194681861348527, 0.15384615384615385, 0.05887939221272555, 0.0389363722697056, 0.011396011396011397, 0.0, 0.0, 0.0], [0.3747044917257683, 0.15780141843971632, 0.208628841607565, 0.17848699763593381, 0.031323877068557916, 0.04196217494089834, 0.0070921985815602835, 0.0, 0.0, 0.0], [0.4164989939637827, 0.22334004024144868, 0.13279678068410464, 0.12474849094567404, 0.05432595573440644, 0.0482897384305835, 0.0, 0.0, 0.0, 0.0], [0.5153374233128835, 0.1579754601226994, 0.09815950920245399, 0.09355828220858896, 0.07975460122699386, 0.05521472392638037, 0.0, 0.0, 0.0, 0.0], [0.35877862595419846, 0.24580152671755726, 0.16030534351145037, 0.13129770992366413, 0.07633587786259542, 0.02748091603053435, 0.0, 0.0, 0.0, 0.0], [0.1799000555247085, 0.09550249861188229, 0.03498056635202665, 0.06718489727928928, 0.03720155469183787, 0.5852304275402554, 0.0, 0.0, 0.0, 0.0], [0.449685534591195, 0.21540880503144655, 0.08490566037735849, 0.08962264150943396, 0.07075471698113207, 0.08962264150943396, 0.0, 0.0, 0.0, 0.0], [0.43169398907103823, 0.20765027322404372, 0.09153005464480875, 0.08196721311475409, 0.09153005464480875, 0.09562841530054644, 0.0, 0.0, 0.0, 0.0], [0.5710549258936356, 0.1918047079337402, 0.05579773321708806, 0.07585004359197908, 0.08020924149956409, 0.025283347863993024, 0.0, 0.0, 0.0, 0.0], [0.635989010989011, 0.18269230769230768, 0.07280219780219781, 0.04395604395604396, 0.027472527472527472, 0.03708791208791209, 0.0, 0.0, 0.0, 0.0], [0.41731066460587324, 0.15765069551777433, 0.12210200927357033, 0.2009273570324575, 0.080370942812983, 0.021638330757341576, 0.0, 0.0, 0.0, 0.0], [0.5248713550600344, 0.19897084048027444, 0.09605488850771869, 0.0686106346483705, 0.09777015437392796, 0.0137221269296741, 0.0, 0.0, 0.0, 0.0], [0.5443037974683544, 0.22784810126582278, 0.08438818565400844, 0.08438818565400844, 0.052742616033755275, 0.006329113924050633, 0.0, 0.0, 0.0, 0.0], [0.4347048300536673, 0.3148479427549195, 0.06708407871198568, 0.10554561717352415, 0.07334525939177101, 0.004472271914132379, 0.0, 0.0, 0.0, 0.0], [0.4739776951672863, 0.19516728624535315, 0.1449814126394052, 0.10408921933085502, 0.07992565055762081, 0.0018587360594795538, 0.0, 0.0, 0.0, 0.0], [0.6331096196868009, 0.16405667412378822, 0.0947054436987323, 0.07755406413124534, 0.030574198359433258, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4978723404255319, 0.17446808510638298, 0.1148936170212766, 0.15319148936170213, 0.059574468085106386, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5595667870036101, 0.17148014440433212, 0.13357400722021662, 0.09025270758122744, 0.04512635379061372, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6461538461538462, 0.20615384615384616, 0.04923076923076923, 0.07076923076923076, 0.027692307692307693, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7154308617234469, 0.1683366733466934, 0.05410821643286573, 0.04609218436873747, 0.01603206412825651, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6265060240963856, 0.20883534136546184, 0.09839357429718876, 0.05823293172690763, 0.008032128514056224, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5880893300248139, 0.17866004962779156, 0.15384615384615385, 0.07692307692307693, 0.0024813895781637717, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6702702702702703, 0.1864864864864865, 0.07297297297297298, 0.05675675675675676, 0.013513513513513514, 0.0, 0.0, 0.0, 0.0, 0.0], [0.06405990016638935, 0.8810316139767055, 0.050124792013311145, 0.00478369384359401, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.549645390070922, 0.29609929078014185, 0.08865248226950355, 0.05673758865248227, 0.008865248226950355, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5454545454545454, 0.21739130434782608, 0.16996047430830039, 0.06719367588932806, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.651595744680851, 0.25, 0.07180851063829788, 0.026595744680851064, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7419354838709677, 0.17050691244239632, 0.055299539170506916, 0.03225806451612903, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7206349206349206, 0.1873015873015873, 0.07301587301587302, 0.01904761904761905, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6536964980544747, 0.245136186770428, 0.08949416342412451, 0.011673151750972763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8140703517587939, 0.12562814070351758, 0.05025125628140704, 0.010050251256281407, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6356164383561644, 0.29315068493150687, 0.06575342465753424, 0.005479452054794521, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7177242888402626, 0.19693654266958424, 0.0700218818380744, 0.015317286652078774, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.704, 0.232, 0.062, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7684563758389261, 0.17114093959731544, 0.05704697986577181, 0.003355704697986577, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.754054054054054, 0.2, 0.04594594594594595, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.504297994269341, 0.45272206303724927, 0.04297994269340974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6456456456456456, 0.2972972972972973, 0.057057057057057055, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6896551724137931, 0.2413793103448276, 0.06896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7931034482758621, 0.16748768472906403, 0.03940886699507389, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.234375, 0.015625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6985645933014354, 0.2966507177033493, 0.004784688995215311, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8089171974522293, 0.18471337579617833, 0.006369426751592357, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7458563535911602, 0.2541436464088398, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.41064638783269963, 0.5817490494296578, 0.0076045627376425855, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5759493670886076, 0.4240506329113924, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6416184971098265, 0.3583815028901734, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.768595041322314, 0.23140495867768596, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7282608695652174, 0.2717391304347826, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8936170212765957, 0.10638297872340426, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8793103448275862, 0.1206896551724138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9102564102564102, 0.08974358974358974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9310344827586207, 0.06896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9615384615384616, 0.038461538461538464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9830508474576272, 0.01694915254237288, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
1500: ([0.6467586752739235, 0.025312260394251936, 0.014226707229601048, 0.011221587395810145, 0.009344315005812372, 0.011384828473201257, 0.01058840988350523, 0.008960945808909005, 0.009190967327051025, 0.00795923919764537, 0.004016719843684301, 0.0068536519007692115, 0.005377062155276891, 0.0042195345155944695, 0.003853478766293191, 0.004897232321733324, 0.005171774133709283, 0.006114120353194331, 0.004857658727214266, 0.005100046993643492, 0.005772798100467463, 0.00989339862976429, 0.003588830352946996, 0.009139026984244762, 0.0033390220375454477, 0.0030273799807078727, 0.004397615690930227, 0.008055699834285572, 0.0043778288936706985, 0.004217061165937028, 0.0041849076203902946, 0.008525636269199376, 0.004778511538176152, 0.0030521134772822835, 0.0028591922040018796, 0.005567510078899854, 0.0032870816947391853, 0.002658850881749153, 0.006141327199426183, 0.005194034280626252, 0.003037273379337637, 0.004691944300165714, 0.0024016225173752815, 0.004152754074843561, 0.002495609804358042, 0.00444955603373649, 0.002119660656426999, 0.0025055032029878063, 0.0016769310677450472, 0.0019959931735549453, 0.0026415374341470653, 0.002349682174569019, 0.0020330934184165615, 0.01317058692587371, 0.002117187306769558, 0.002522816650589894, 0.0024016225173752815, 0.0017016645643194578, 0.0014518562489179096, 0.001226781430090772, 0.0016447775221983133, 0.001911899285201949, 0.00200588657218471, 0.0018129652989043062, 0.0015928371793920506, 0.004894758972075882, 0.0013504489129628255, 0.002569810294081274, 0.000717271400657911, 0.0007988919393534664, 0.002151814201973733, 0.0012861418218693577, 0.0006653310578516485, 0.0006925379040835002, 0.0008755657787341397, 0.0007370581979174396, 0.0009398728698276075, 0.00046004303628403945, 0.000477356483886127, 0.00037100244861616087, 0.0002448616160866662, 0.0003116420568375751, 0.00025475501471643045, 0.00028443521060572333, 0.0002671217630036358, 0.0003388489030694269, 0.00019539462293784472, 0.0001434542801315822, 0.0002226014691696965, 0.00017808117533575722, 0.0003858425465608073, 0.00019786797259528578, 0.00016571442704855184, 0.00014840097944646434, 8.409388835299647e-05, 9.89339862976429e-06, 0.0, 0.0, 0.0, 0.0] ,
[[0.21723883422374002, 0.1279240968140395, 0.08602972951267922, 0.12143438971130938, 0.10296339070943168, 0.08191486513876195, 0.112049745497933, 0.08063375030115759, 0.050307658772194834, 0.019503539318752844], [0.12829783076021106, 0.1025014657025601, 0.09663865546218488, 0.11921047488762947, 0.10875512995896033, 0.11461794019933555, 0.1385577486808677, 0.09976548759038499, 0.07152628493257768, 0.020128981825288255], [0.1310848400556328, 0.10465924895688457, 0.09040333796940195, 0.14082058414464535, 0.10448539638386647, 0.11474269819193324, 0.11161335187760779, 0.0844923504867872, 0.10448539638386647, 0.01321279554937413], [0.1421644258320476, 0.15428697377121445, 0.11307031077804716, 0.12497244875468372, 0.10623760193960767, 0.10359268238924399, 0.10249063257659247, 0.06744544853427376, 0.08133127617368305, 0.004408199250606128], [0.14002117522498678, 0.13816834303864478, 0.13022763366860773, 0.13155108523028058, 0.10958178930651138, 0.12308099523557438, 0.10852302805717311, 0.06193753308628904, 0.0513499205929063, 0.00555849655902594], [0.18987616771670648, 0.17705844014772973, 0.08994134260265045, 0.17879643710623505, 0.09167933956115577, 0.04670866825983055, 0.06908537910058657, 0.10036932435368238, 0.05605040191179665, 0.00043449923962633063], [0.23615977575332867, 0.2319551506657323, 0.09110021023125438, 0.16444755898154637, 0.06283578603130109, 0.0539593552908199, 0.06867554309740714, 0.06587245970567625, 0.024760569960289653, 0.000233590282644242], [0.20397460667954734, 0.1675407121170301, 0.08032017664918575, 0.16367651117858129, 0.08363234888214187, 0.07645597571073696, 0.07369583218327352, 0.11951421473916643, 0.031189621860336737, 0.0], [0.3110871905274489, 0.13966630785791173, 0.0984930032292788, 0.08234660925726588, 0.10279870828848224, 0.08180839612486544, 0.08476856835306781, 0.07238966630785791, 0.026641550053821312, 0.0], [0.14729645742697328, 0.11187072715972654, 0.09198259788688626, 0.22249844623990056, 0.09912989434431324, 0.09695463020509633, 0.11653200745804848, 0.0711622125543816, 0.04257302672467371, 0.0], [0.21982758620689655, 0.15517241379310345, 0.12253694581280788, 0.1440886699507389, 0.0812807881773399, 0.09421182266009852, 0.10406403940886699, 0.06403940886699508, 0.014778325123152709, 0.0], [0.17466618549260196, 0.18946228798267772, 0.14687838325514255, 0.1526524720317575, 0.1035727174305305, 0.06495849873691809, 0.06676290147961025, 0.09274630097437749, 0.008300252616383976, 0.0], [0.17755289788408463, 0.20423183072677092, 0.10993560257589696, 0.14121435142594296, 0.1062557497700092, 0.09015639374425023, 0.06393744250229991, 0.09889604415823366, 0.0078196872125115, 0.0], [0.1951934349355217, 0.12602579132473624, 0.13188745603751464, 0.134232121922626, 0.13833528722157093, 0.11019929660023446, 0.09202813599062133, 0.06858147713950762, 0.0035169988276670576, 0.0], [0.21566110397946084, 0.14441591784338895, 0.1386392811296534, 0.15019255455712452, 0.1123234916559692, 0.08023106546854943, 0.10077021822849808, 0.055198973042362, 0.0025673940949935813, 0.0], [0.14343434343434344, 0.19090909090909092, 0.1409090909090909, 0.25707070707070706, 0.09090909090909091, 0.07828282828282829, 0.0691919191919192, 0.028282828282828285, 0.00101010101010101, 0.0], [0.34576757532281205, 0.15447154471544716, 0.1406025824964132, 0.1406025824964132, 0.07556193208990913, 0.05978000956480153, 0.04495456719273075, 0.03825920612147298, 0.0, 0.0], [0.3547734627831715, 0.18446601941747573, 0.12378640776699029, 0.154126213592233, 0.06351132686084142, 0.05137540453074434, 0.045307443365695796, 0.022653721682847898, 0.0, 0.0], [0.310081466395112, 0.1710794297352342, 0.14663951120162932, 0.1705702647657841, 0.07077393075356415, 0.051934826883910386, 0.05091649694501019, 0.0280040733197556, 0.0, 0.0], [0.19641125121241512, 0.18525703200775945, 0.16100872938894278, 0.17604267701260912, 0.09262851600387972, 0.07129000969932105, 0.08098933074684772, 0.036372453928225024, 0.0, 0.0], [0.23436161096829478, 0.16966580976863754, 0.14181662382176521, 0.15509854327335046, 0.07497857754927163, 0.07497857754927163, 0.09854327335047129, 0.050556983718937444, 0.0, 0.0], [0.38025, 0.23275, 0.19875, 0.07, 0.0585, 0.03025, 0.02225, 0.00725, 0.0, 0.0], [0.31977946243969674, 0.17436250861474845, 0.1578221915920055, 0.1330117160578911, 0.09510682288077188, 0.05513439007580979, 0.05306685044796692, 0.01171605789110958, 0.0, 0.0], [0.22895805142083897, 0.09634641407307172, 0.1699594046008119, 0.21055480378890393, 0.172936400541272, 0.0722598105548038, 0.04032476319350474, 0.008660351826792964, 0.0, 0.0], [0.34, 0.17851851851851852, 0.1111111111111111, 0.16518518518518518, 0.07851851851851852, 0.0725925925925926, 0.04888888888888889, 0.005185185185185185, 0.0, 0.0], [0.3431372549019608, 0.19444444444444445, 0.136437908496732, 0.1454248366013072, 0.09313725490196079, 0.04820261437908497, 0.0392156862745098, 0.0, 0.0, 0.0], [0.32677165354330706, 0.2525309336332958, 0.1327334083239595, 0.10573678290213723, 0.0984251968503937, 0.04611923509561305, 0.03712035995500562, 0.0005624296962879641, 0.0, 0.0], [0.24378262204482654, 0.1946576604237028, 0.22290451335584893, 0.15013816395455942, 0.0961007061713233, 0.0604851089960086, 0.03193122505373042, 0.0, 0.0, 0.0], [0.30112994350282485, 0.2423728813559322, 0.12316384180790961, 0.12768361581920903, 0.1152542372881356, 0.06610169491525424, 0.024293785310734464, 0.0, 0.0, 0.0], [0.3618768328445748, 0.18357771260997066, 0.14604105571847506, 0.12316715542521994, 0.11906158357771261, 0.04750733137829912, 0.0187683284457478, 0.0, 0.0, 0.0], [0.34515366430260047, 0.1702127659574468, 0.17966903073286053, 0.14598108747044916, 0.08392434988179669, 0.051418439716312055, 0.02364066193853428, 0.0, 0.0, 0.0], [0.30432259936176387, 0.15346678270960254, 0.2178706121264868, 0.14070205976211198, 0.08906295329271831, 0.08064984044096316, 0.01392515230635335, 0.0, 0.0, 0.0], [0.30434782608695654, 0.2468944099378882, 0.18219461697722567, 0.12681159420289856, 0.06314699792960662, 0.06884057971014493, 0.007763975155279503, 0.0, 0.0, 0.0], [0.3565640194489465, 0.21961102106969205, 0.13290113452188007, 0.15153970826580226, 0.08346839546191248, 0.0526742301458671, 0.0032414910858995136, 0.0, 0.0, 0.0], [0.2811418685121107, 0.25519031141868515, 0.19204152249134948, 0.14965397923875431, 0.07958477508650519, 0.03806228373702422, 0.004325259515570935, 0.0, 0.0, 0.0], [0.1754775655264327, 0.11772545535317637, 0.05819635717458907, 0.11061750333185251, 0.050644158151932475, 0.48733896046201686, 0.0, 0.0, 0.0, 0.0], [0.35289691497366443, 0.22648607975921745, 0.15575620767494355, 0.12791572610985705, 0.062452972159518436, 0.0744920993227991, 0.0, 0.0, 0.0, 0.0], [0.34604651162790695, 0.24, 0.12, 0.12279069767441861, 0.08744186046511628, 0.08372093023255814, 0.0, 0.0, 0.0, 0.0], [0.38219895287958117, 0.23076923076923078, 0.08135320177204994, 0.08175594039468385, 0.1103503826016915, 0.11357229158276279, 0.0, 0.0, 0.0, 0.0], [0.28904761904761905, 0.29, 0.13285714285714287, 0.20285714285714285, 0.05714285714285714, 0.028095238095238097, 0.0, 0.0, 0.0, 0.0], [0.31840390879478825, 0.1978827361563518, 0.15716612377850162, 0.18485342019543974, 0.11970684039087948, 0.021986970684039087, 0.0, 0.0, 0.0, 0.0], [0.26673695308381656, 0.18555614127569847, 0.15498154981549817, 0.13442277279915657, 0.22456510279388509, 0.03373748023194518, 0.0, 0.0, 0.0, 0.0], [0.33779608650875387, 0.2615859938208033, 0.14212152420185376, 0.12255406797116375, 0.10092687950566426, 0.035015447991761074, 0.0, 0.0, 0.0, 0.0], [0.391304347826087, 0.3019654556283502, 0.0905300774270399, 0.1209053007742704, 0.08814770696843359, 0.00714711137581894, 0.0, 0.0, 0.0, 0.0], [0.4658077304261645, 0.21605550049554015, 0.13875123885034688, 0.09514370664023786, 0.07928642220019821, 0.004955401387512388, 0.0, 0.0, 0.0, 0.0], [0.5308504724847137, 0.20066703724291274, 0.1122846025569761, 0.09949972206781545, 0.055586436909394105, 0.0011117287381878821, 0.0, 0.0, 0.0, 0.0], [0.411901983663944, 0.20536756126021002, 0.13885647607934656, 0.16802800466744458, 0.073512252042007, 0.002333722287047841, 0.0, 0.0, 0.0, 0.0], [0.43237907206317866, 0.23593287265547877, 0.16386969397828233, 0.11648568608094768, 0.05133267522211254, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5132743362831859, 0.2536873156342183, 0.08849557522123894, 0.08849557522123894, 0.05604719764011799, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5811648079306072, 0.21065675340768278, 0.09541511771995044, 0.08302354399008674, 0.02973977695167286, 0.0, 0.0, 0.0, 0.0, 0.0], [0.43352059925093633, 0.2752808988764045, 0.11329588014981273, 0.10767790262172285, 0.0702247191011236, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4242105263157895, 0.33052631578947367, 0.1231578947368421, 0.10736842105263159, 0.014736842105263158, 0.0, 0.0, 0.0, 0.0, 0.0], [0.49878345498783455, 0.2542579075425791, 0.13990267639902676, 0.08759124087591241, 0.019464720194647202, 0.0, 0.0, 0.0, 0.0, 0.0], [0.08582159624413145, 0.831924882629108, 0.06309859154929577, 0.018591549295774647, 0.0005633802816901409, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4929906542056075, 0.2967289719626168, 0.10630841121495327, 0.0969626168224299, 0.007009345794392523, 0.0, 0.0, 0.0, 0.0, 0.0], [0.48823529411764705, 0.26372549019607844, 0.17058823529411765, 0.07647058823529412, 0.000980392156862745, 0.0, 0.0, 0.0, 0.0, 0.0], [0.490216271884655, 0.3171987641606591, 0.11225540679711637, 0.07723995880535531, 0.003089598352214212, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47238372093023256, 0.28343023255813954, 0.1555232558139535, 0.08866279069767442, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5076660988074957, 0.27597955706984667, 0.16695059625212946, 0.049403747870528106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5564516129032258, 0.3004032258064516, 0.09475806451612903, 0.04838709677419355, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6406015037593985, 0.22556390977443608, 0.09022556390977443, 0.04360902255639098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5265200517464425, 0.2833117723156533, 0.15523932729624837, 0.03492884864165589, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6744759556103576, 0.2120838471023428, 0.07891491985203453, 0.0345252774352651, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6425648021828103, 0.25375170532060026, 0.0941336971350614, 0.009549795361527967, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6226708074534162, 0.2096273291925466, 0.16304347826086957, 0.004658385093167702, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7367357251136938, 0.2112177867609904, 0.05103587670540677, 0.001010611419909045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.49633699633699635, 0.43772893772893773, 0.056776556776556776, 0.009157509157509158, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7112608277189605, 0.23965351299326276, 0.04908565928777671, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7241379310344828, 0.2, 0.07586206896551724, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7213622291021672, 0.22291021671826625, 0.05572755417956656, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.43908045977011495, 0.5218390804597701, 0.03908045977011494, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5288461538461539, 0.4519230769230769, 0.019230769230769232, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7137546468401487, 0.26765799256505574, 0.01858736059479554, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6857142857142857, 0.30357142857142855, 0.010714285714285714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4858757062146893, 0.5028248587570622, 0.011299435028248588, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6879194630872483, 0.3087248322147651, 0.003355704697986577, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5105263157894737, 0.48947368421052634, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7311827956989247, 0.26881720430107525, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7357512953367875, 0.26424870466321243, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8066666666666666, 0.19333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8787878787878788, 0.12121212121212122, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8968253968253969, 0.10317460317460317, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9223300970873787, 0.07766990291262135, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9478260869565217, 0.05217391304347826, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9907407407407407, 0.009259259259259259, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
2000: ([0.5907305662643846, 0.025437664966721722, 0.013383674377626577, 0.014074748128944955, 0.011697599151896925, 0.010655561244149844, 0.010324497117209364, 0.008801240314565193, 0.008140921154274073, 0.011657799092920473, 0.005572008256703144, 0.007533065708088277, 0.008039611913243108, 0.005638944719527176, 0.0051287803271926665, 0.005479744483621371, 0.007612665826041178, 0.006339063938794746, 0.005032898366931216, 0.005761963083636206, 0.005704072088761368, 0.013132210368639001, 0.006917973887543124, 0.00855701267993697, 0.004774197983584285, 0.0040831242322659075, 0.006402382214439099, 0.008122830218375687, 0.004470270260491386, 0.004432279295104774, 0.004499215757928805, 0.0073015017285889255, 0.00506908023872799, 0.0033034048950454356, 0.00335225042197108, 0.005505071793879113, 0.003169531969397373, 0.002767913192453185, 0.005669699310554433, 0.004790479825892833, 0.003109831880932696, 0.004379815580999452, 0.002968722580925279, 0.004871889037435574, 0.002695549448859638, 0.004005333207902844, 0.0030067135463118914, 0.0030664136347765677, 0.0030700318219562452, 0.005161344011809763, 0.006512736923419259, 0.0033088321758149514, 0.0031261137232412445, 0.011947254067294664, 0.0027480131629649598, 0.003328732205303177, 0.003410141416845918, 0.0027480131629649598, 0.0022939306719154504, 0.002328303450122385, 0.0022686033616577087, 0.002152821371908033, 0.002454940001411093, 0.0065615824503449036, 0.001861557303944005, 0.0041482516015001, 0.0018181390577878766, 0.0025218764642351245, 0.0014309930295623983, 0.0009208286372278898, 0.002020757539849809, 0.001342347443660303, 0.0010637470308251457, 0.0015938114526478798, 0.0012537018577582074, 0.000919019543638051, 0.0011741017398053054, 0.0006784100961895062, 0.0006277554756740231, 0.0004504643038698321, 0.0004088551513035424, 0.000548155357721121, 0.0005499644513109596, 0.0004378006487409613, 0.0004088551513035424, 0.00044684611669015476, 0.0003401095948896724, 0.00036724599873725266, 0.00031478228463193087, 0.0001917639205229004, 0.00035639143719822056, 0.0003618187179677366, 0.0002442276346282222, 0.00028221860001483457, 0.0002532731025774156, 5.2463714105321814e-05, 1.0854561539032099e-05, 5.4272807695160494e-06, 0.0, 0.0] ,
[[0.17727097331365188, 0.1052815327040982, 0.07265705868301617, 0.10331542810243344, 0.09993140071171762, 0.09646468667887571, 0.13485578837119566, 0.10131869881849975, 0.07412704343192439, 0.03477738918458721], [0.096436953275016, 0.07709266766232843, 0.07488798805205889, 0.09352108669369177, 0.09778820852001992, 0.168764668231278, 0.17416968921129364, 0.10596685868714885, 0.08171538297418392, 0.029656496692980584], [0.10935387942687212, 0.08556366585563666, 0.07610164909434983, 0.12287104622871046, 0.11016490943498243, 0.12124898621248986, 0.14233576642335766, 0.10043254933765883, 0.11192214111922141, 0.020005406866720737], [0.08637532133676093, 0.093573264781491, 0.07197943444730077, 0.08341902313624679, 0.08933161953727506, 0.24344473007712084, 0.16915167095115682, 0.07159383033419023, 0.07904884318766067, 0.012082262210796915], [0.09078255490256727, 0.0867615218063718, 0.08335910918651407, 0.08660686668728734, 0.10980513454995361, 0.25394370553665324, 0.14676770801113517, 0.07036807918342097, 0.06263532322919889, 0.008969996906897619], [0.15042444821731749, 0.14159592529711376, 0.07538200339558573, 0.14634974533106962, 0.10764006791171477, 0.10424448217317488, 0.10594227504244483, 0.10203735144312394, 0.06485568760611206, 0.0015280135823429542], [0.1801296653232872, 0.1766251971263361, 0.07552128964429648, 0.13930261082880674, 0.07394427895566848, 0.11109164184335027, 0.1121429823024356, 0.08393201331697915, 0.0466094270194498, 0.0007008936393902226], [0.15847893114080164, 0.12702980472764647, 0.06413155190133607, 0.1358684480986639, 0.08859198355601233, 0.14779033915724563, 0.10298047276464542, 0.11963001027749229, 0.0552929085303186, 0.00020554984583761563], [0.25955555555555554, 0.12, 0.08755555555555555, 0.07777777777777778, 0.11022222222222222, 0.10733333333333334, 0.11044444444444444, 0.0908888888888889, 0.036222222222222225, 0.0], [0.07526381129733085, 0.05819366852886406, 0.04903786468032278, 0.12011173184357542, 0.09279950341402855, 0.38687150837988826, 0.13656114214773432, 0.0521415270018622, 0.029019242706393545, 0.0], [0.12175324675324675, 0.09480519480519481, 0.08149350649350649, 0.09707792207792208, 0.1331168831168831, 0.23214285714285715, 0.13116883116883116, 0.07857142857142857, 0.02987012987012987, 0.0], [0.12079731027857829, 0.13328530259365995, 0.10470701248799232, 0.1186359269932757, 0.13280499519692604, 0.18419788664745437, 0.09486071085494717, 0.09462055715658022, 0.016090297790585975, 0.0], [0.09473447344734473, 0.10531053105310531, 0.06233123312331233, 0.08303330333033303, 0.10306030603060307, 0.3892889288928893, 0.07875787578757876, 0.06975697569756975, 0.013726372637263727, 0.0], [0.12447866538338145, 0.0792428617260186, 0.08533846647417388, 0.10362528071863972, 0.15527751042669233, 0.25665704202759065, 0.1017003529034328, 0.08020532563362208, 0.013474494706448507, 0.0], [0.14038800705467372, 0.1001763668430335, 0.09559082892416226, 0.13686067019400353, 0.22257495590828924, 0.12310405643738977, 0.09664902998236331, 0.07513227513227513, 0.009523809523809525, 0.0], [0.11125784087157478, 0.13898976559920767, 0.10762627930009905, 0.18586992406734895, 0.12545394519643446, 0.1914823374050842, 0.09210960713106636, 0.04522944866292506, 0.0019808517662594917, 0.0], [0.18512357414448669, 0.08555133079847908, 0.08531368821292776, 0.09149239543726236, 0.11692015209125475, 0.3341254752851711, 0.05869771863117871, 0.04230038022813688, 0.0004752851711026616, 0.0], [0.2585616438356164, 0.13641552511415525, 0.1021689497716895, 0.12557077625570776, 0.09674657534246575, 0.16666666666666666, 0.07248858447488585, 0.04138127853881279, 0.0, 0.0], [0.23112868439971243, 0.13407620416966212, 0.12437095614665708, 0.15240833932422718, 0.13227893601725377, 0.10460100647016535, 0.08087706685837527, 0.040258806613946804, 0.0, 0.0], [0.15384615384615385, 0.13783359497645212, 0.1315541601255887, 0.1497645211930926, 0.1792778649921507, 0.11397174254317112, 0.09105180533751962, 0.04270015698587127, 0.0, 0.0], [0.18553758325404376, 0.13606089438629876, 0.11988582302568981, 0.13574373612432603, 0.11195686647637171, 0.15667618141452586, 0.10244211861718998, 0.051696796701554075, 0.0, 0.0], [0.21201267392202783, 0.13142306102768977, 0.11378977820636452, 0.0469761675161868, 0.1210910593745695, 0.3340680534508885, 0.031271525003444, 0.00936768149882904, 0.0, 0.0], [0.12578451882845187, 0.07479079497907949, 0.07322175732217573, 0.09309623430962342, 0.16422594142259414, 0.40193514644351463, 0.05543933054393305, 0.011506276150627616, 0.0, 0.0], [0.1843551797040169, 0.080338266384778, 0.14482029598308668, 0.1828752642706131, 0.18520084566596196, 0.16405919661733614, 0.048625792811839326, 0.009725158562367865, 0.0, 0.0], [0.17999242137173171, 0.10155361879499811, 0.0799545282303903, 0.12277377794619174, 0.13338385752178855, 0.3008715422508526, 0.07616521409624857, 0.005305039787798408, 0.0, 0.0], [0.20115197164377493, 0.1227292866637129, 0.09924678777137794, 0.1546300398759415, 0.14754098360655737, 0.19893664155959237, 0.07221976074435091, 0.003544528134692069, 0.0, 0.0], [0.18310257134783836, 0.14665159649618537, 0.10285391353489687, 0.160497315625883, 0.15851935575021192, 0.17632099463125175, 0.07035885843458604, 0.0016953941791466515, 0.0, 0.0], [0.1866369710467706, 0.1510022271714922, 0.17527839643652562, 0.13140311804008908, 0.1271714922048998, 0.1861915367483296, 0.042316258351893093, 0.0, 0.0, 0.0], [0.22784297855119384, 0.1906110886280858, 0.11493322541481182, 0.12788344799676243, 0.15783083771752326, 0.12343180898421692, 0.05746661270740591, 0.0, 0.0, 0.0], [0.26816326530612244, 0.14775510204081632, 0.13183673469387755, 0.13387755102040816, 0.15224489795918367, 0.1126530612244898, 0.053469387755102044, 0.0, 0.0, 0.0], [0.2513067953357459, 0.13791716928025735, 0.15520707679935666, 0.1467631684760756, 0.12625653397667874, 0.08805790108564536, 0.09449135504624046, 0.0, 0.0, 0.0], [0.26734390485629334, 0.13974231912784935, 0.19672943508424182, 0.14246778989098116, 0.10431119920713577, 0.11546085232903865, 0.03394449950445986, 0.0, 0.0, 0.0], [0.2259100642398287, 0.21020699500356888, 0.15060670949321914, 0.12919343326195576, 0.08493932905067808, 0.14953604568165596, 0.0496074232690935, 0.0, 0.0, 0.0], [0.27546549835706463, 0.17031763417305587, 0.13088718510405256, 0.1566265060240964, 0.10350492880613363, 0.13964950711938665, 0.023548740416210297, 0.0, 0.0, 0.0], [0.20561252023745277, 0.20021586616297896, 0.1651376146788991, 0.15434430652995143, 0.11980572045331894, 0.13491635186184567, 0.019967620075553156, 0.0, 0.0, 0.0], [0.15018074268813672, 0.10417351298061124, 0.0782122905027933, 0.14393690437068682, 0.10187315149523496, 0.4170226749917844, 0.004600722970752547, 0.0, 0.0, 0.0], [0.29337899543379, 0.1969178082191781, 0.1506849315068493, 0.136986301369863, 0.08789954337899543, 0.1329908675799087, 0.001141552511415525, 0.0, 0.0, 0.0], [0.26405228758169935, 0.2169934640522876, 0.14444444444444443, 0.1392156862745098, 0.12026143790849673, 0.11437908496732026, 0.00065359477124183, 0.0, 0.0, 0.0], [0.3283343969368219, 0.2131461391193363, 0.10146777281429484, 0.104977664326739, 0.13146139119336311, 0.1206126356094448, 0.0, 0.0, 0.0, 0.0], [0.270392749244713, 0.26132930513595165, 0.1506797583081571, 0.1952416918429003, 0.08496978851963746, 0.037386706948640484, 0.0, 0.0, 0.0, 0.0], [0.2716695753344968, 0.19313554392088422, 0.16870273414776032, 0.1820826061663758, 0.14485165794066318, 0.03955788248981966, 0.0, 0.0, 0.0, 0.0], [0.24576621230896323, 0.18380834365964477, 0.1697645600991326, 0.1416769929781082, 0.22057001239157373, 0.03841387856257745, 0.0, 0.0, 0.0, 0.0], [0.2687385740402194, 0.238878732480195, 0.17184643510054845, 0.12797074954296161, 0.15843997562461914, 0.03412553321145643, 0.0, 0.0, 0.0, 0.0], [0.331229112513925, 0.25547716301522466, 0.1318232454511697, 0.16450055699962868, 0.10434459710360193, 0.012625324916450055, 0.0, 0.0, 0.0, 0.0], [0.38456375838926177, 0.21946308724832214, 0.17248322147651007, 0.09731543624161074, 0.11006711409395974, 0.016107382550335572, 0.0, 0.0, 0.0, 0.0], [0.46973803071364045, 0.1996386630532972, 0.13504968383017163, 0.11517615176151762, 0.07768744354110207, 0.0027100271002710027, 0.0, 0.0, 0.0, 0.0], [0.34416365824308065, 0.2352587244283995, 0.1690734055354994, 0.1371841155234657, 0.11311672683513839, 0.0012033694344163659, 0.0, 0.0, 0.0, 0.0], [0.36932153392330386, 0.25309734513274335, 0.20530973451327433, 0.10619469026548672, 0.06607669616519174, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4295816146140248, 0.26222746022392457, 0.16087212728344136, 0.07719505008839128, 0.07012374779021803, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44970206799859797, 0.2576235541535226, 0.18226428321065544, 0.07045215562565721, 0.03995793901156677, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44055555555555553, 0.26805555555555555, 0.17666666666666667, 0.06944444444444445, 0.04527777777777778, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36686714051394204, 0.3225806451612903, 0.17495899398578457, 0.0989611809732094, 0.036632039365773646, 0.0, 0.0, 0.0, 0.0, 0.0], [0.41550925925925924, 0.2650462962962963, 0.1909722222222222, 0.0931712962962963, 0.03530092592592592, 0.0, 0.0, 0.0, 0.0, 0.0], [0.15006056935190792, 0.7204724409448819, 0.08918837068443368, 0.03255602665051484, 0.007722592368261659, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4575378538512179, 0.29295589203423306, 0.15273206056616195, 0.08426596445029624, 0.012508229098090849, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45163043478260867, 0.2668478260869565, 0.18858695652173912, 0.08641304347826087, 0.006521739130434782, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4742705570291777, 0.30875331564986735, 0.1336870026525199, 0.08063660477453581, 0.002652519893899204, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4970375246872943, 0.2705727452271231, 0.14944042132982224, 0.08294930875576037, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44873817034700314, 0.2941640378548896, 0.19085173501577288, 0.06624605678233439, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5042735042735043, 0.2867132867132867, 0.15384615384615385, 0.05516705516705517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5829346092503987, 0.2511961722488038, 0.12280701754385964, 0.0430622009569378, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5016806722689076, 0.2991596638655462, 0.16134453781512606, 0.037815126050420166, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5630066322770818, 0.26897568165070007, 0.1156963890935888, 0.05232129697862933, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7267714364488558, 0.17866004962779156, 0.08243727598566308, 0.01213123793768955, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.577259475218659, 0.2108843537414966, 0.2001943634596696, 0.011661807580174927, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7104230266027038, 0.22328826864369822, 0.0623637156563454, 0.0039249890972525075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5890547263681593, 0.3154228855721393, 0.08955223880597014, 0.005970149253731343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6836441893830703, 0.24390243902439024, 0.07245337159253945, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7370417193426043, 0.168141592920354, 0.09481668773704172, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7210216110019646, 0.18860510805500982, 0.09037328094302555, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5058191584601611, 0.4386750223813787, 0.05550581915846016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5390835579514824, 0.4110512129380054, 0.04986522911051213, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7091836734693877, 0.21768707482993196, 0.07312925170068027, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6401816118047673, 0.2542565266742338, 0.10556186152099886, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5685425685425686, 0.38095238095238093, 0.050505050505050504, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6850393700787402, 0.2992125984251969, 0.015748031496062992, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6040061633281972, 0.39599383667180277, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6826666666666666, 0.31733333333333336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7550432276657061, 0.24495677233429394, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7991967871485943, 0.20080321285140562, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7920353982300885, 0.2079646017699115, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8415841584158416, 0.15841584158415842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8585526315789473, 0.14144736842105263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9090909090909091, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9557522123893806, 0.04424778761061947, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9878542510121457, 0.012145748987854251, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9787234042553191, 0.02127659574468085, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
3000: ([0.5559157549429194, 0.02295701576623423, 0.012990949123734928, 0.013369802283221708, 0.011459193800476612, 0.010300348842046461, 0.010273606266082688, 0.009000362510474176, 0.009275216763435174, 0.011221482014131966, 0.005852167040072264, 0.00726655216882291, 0.008614080857664125, 0.006107707210392759, 0.00526234467020461, 0.005443599907292403, 0.007588948779052838, 0.006452389300592496, 0.005453999797944982, 0.006485074671214885, 0.0055743413897819585, 0.012114386911589043, 0.006639587332338905, 0.008270884466129043, 0.004804749481491166, 0.004320411716813949, 0.0061805064449608075, 0.0075532920111011405, 0.00472897884959381, 0.004862691729412673, 0.005246001984893416, 0.00738392236333058, 0.005118231899733168, 0.003800417184185035, 0.004474924377937969, 0.0056753688989784334, 0.00369196118166529, 0.0032729941582328504, 0.00533217250744335, 0.005066232446470277, 0.0033829358594172497, 0.004529895228530169, 0.00405595735450553, 0.005361886480736431, 0.0035359628218766155, 0.0046160657510801026, 0.00333539350214832, 0.0047913781935092796, 0.0035627053978403884, 0.005079603734452163, 0.006357304586054638, 0.0038969875973875476, 0.003469106381967184, 0.010783200908059024, 0.0031897950330122245, 0.0037692175122273, 0.0036429331257317064, 0.004191155932989047, 0.0032640799662449265, 0.0038182455681608834, 0.0028005419828728657, 0.0034111641340456764, 0.0030174539879123555, 0.0063662187780425625, 0.0028272845588366386, 0.004528409529865515, 0.003775160306885916, 0.003286365446214737, 0.0019076370854157877, 0.0016491255177659846, 0.0021958626263586713, 0.0023028329302137622, 0.001551069405898818, 0.0019700364293312573, 0.007137296384998009, 0.001439642006049765, 0.0015555265018927802, 0.0015064984459591969, 0.0010637602438922928, 0.0007086782630399772, 0.0006314219324779671, 0.0017962096855667347, 0.0013356430995239822, 0.0007577063189735605, 0.0010459318599164442, 0.0006180506444960807, 0.0006433075217951994, 0.0014708416780074998, 0.0004100528314445151, 0.00037439606349281814, 0.0004382811060729419, 0.00042045272209709336, 0.00028228274628426764, 0.00030753962358338633, 0.00026296866364376514, 8.17134265559722e-05, 5.348515192754545e-05, 2.9713973293080803e-05, 2.9713973293080803e-06, 0.0] ,
[[0.1554286997097638, 0.09272859441228506, 0.06466173853085964, 0.09240254638166862, 0.09092196761968903, 0.0919642523077252, 0.13314251505967747, 0.11797059153664834, 0.10060452511906098, 0.06017456932262186], [0.08866166192078695, 0.07073517991198551, 0.06885840020709293, 0.08775563033911468, 0.09260937095521615, 0.15972042454051255, 0.16858659073259125, 0.12114936577789283, 0.10024592285788247, 0.04167745275692467], [0.0932067703568161, 0.0736505032021958, 0.06575937785910338, 0.11036139066788656, 0.10098353156450138, 0.11653705397987191, 0.1492451967063129, 0.13517840805123513, 0.12614364135407136, 0.02893412625800549], [0.07745305033892655, 0.08256472941437938, 0.06534059339926658, 0.0783420380042227, 0.08223135903989331, 0.22669185465051672, 0.17013001444604955, 0.10023335926214023, 0.09756639626625181, 0.01944660517835315], [0.07753144042525606, 0.07480876442370024, 0.07364190328017632, 0.07740178918708673, 0.10216517567742772, 0.22922338908336576, 0.156618695708544, 0.09879424348502529, 0.09425645014909892, 0.015558148580318941], [0.12894850713976633, 0.1224578104716573, 0.06591663060724073, 0.12909274484350208, 0.10038944180008655, 0.10370690898600894, 0.12664070387999424, 0.12635222847252273, 0.08870618779749026, 0.007788836001730853], [0.1499638467100506, 0.14895155459146783, 0.06637744034707159, 0.12176428054953001, 0.07433116413593636, 0.11482284887924801, 0.13651482284887925, 0.11395516992046276, 0.06796818510484454, 0.005350686912509038], [0.12958071970947507, 0.10283922086497194, 0.05348299768900627, 0.118190822053483, 0.08204027731924728, 0.14509739187850776, 0.1269395840211291, 0.15236051502145923, 0.08880818752063387, 0.0006602839220864972], [0.18981259010091303, 0.09066154092583693, 0.06823642479577126, 0.06615409258369374, 0.09306423193977255, 0.10379625180201826, 0.13567195258689732, 0.12109562710235464, 0.13150728816274226, 0.0], [0.06527207732027009, 0.05084072553952072, 0.044088441678803124, 0.10883092810803655, 0.08685290613001456, 0.3554878856083675, 0.1443135178074937, 0.07043558850787766, 0.07387792929961605, 0.0], [0.09672505712109672, 0.07717694846407717, 0.0682914445290683, 0.08631632394008631, 0.12261995430312261, 0.2168062960142168, 0.16197004315816196, 0.10332571718710333, 0.06676821528306677, 0.0], [0.10509098343896954, 0.11531384175015334, 0.09139235330198324, 0.10672664076875894, 0.12615007156000818, 0.18564710693109793, 0.122469842567982, 0.11551829891637702, 0.0316908607646698, 0.0], [0.0753708175232839, 0.08347706105553639, 0.05933080372542256, 0.07054156605726113, 0.09313556398758192, 0.3439116936874784, 0.15367368057951017, 0.09072093825457055, 0.02983787512935495, 0.0], [0.09778642666018, 0.06494770128922403, 0.07248844563366577, 0.08513743614692289, 0.13257115057163707, 0.23619557285332035, 0.16759912430065677, 0.11457066407200195, 0.028703478472391145, 0.0], [0.1163184641445511, 0.08328627893845285, 0.07989836250705816, 0.11914172783738002, 0.19932241671372106, 0.13890457368718237, 0.14088085827216262, 0.10163749294184077, 0.020609824957651044, 0.0], [0.09497816593886463, 0.11763100436681223, 0.09143013100436681, 0.16512008733624453, 0.126910480349345, 0.19486899563318777, 0.12745633187772926, 0.07314410480349345, 0.008460698689956333, 0.0], [0.15407204385277995, 0.07184808144087705, 0.07576350822239625, 0.08496476115896633, 0.1139389193422083, 0.3191072826938136, 0.11237274862960063, 0.0644087705559906, 0.003523884103367267, 0.0], [0.21344692608795762, 0.1130554915956712, 0.08795763297259959, 0.10545705733364034, 0.10499654616624453, 0.18005986645176145, 0.1234169928620769, 0.06976744186046512, 0.0018420446695832373, 0.0], [0.18360119858349225, 0.11386543176246254, 0.1089621356578589, 0.13293380550258785, 0.12830291473712885, 0.15254698992100246, 0.11250340506673931, 0.06728411876872786, 0.0, 0.0], [0.1140893470790378, 0.10607101947308133, 0.10515463917525773, 0.12462772050400917, 0.18258877434135165, 0.17525773195876287, 0.13104238258877435, 0.061168384879725084, 0.0, 0.0], [0.15911513859275053, 0.1170042643923241, 0.10820895522388059, 0.12553304904051174, 0.11940298507462686, 0.17537313432835822, 0.13246268656716417, 0.0628997867803838, 0.0, 0.0], [0.19119450576404218, 0.12030905077262694, 0.10779985283296542, 0.055432916360068675, 0.12349766985528575, 0.32303164091243564, 0.057149865096884966, 0.021584498405690457, 0.0, 0.0], [0.11143432535242784, 0.06601029313045424, 0.07070933094652047, 0.09353322891027076, 0.17140299843365406, 0.3797270082792571, 0.08368762586708436, 0.023495189080331172, 0.0, 0.0], [0.1656188252200467, 0.0749056942698042, 0.13023172265133826, 0.16490030537093586, 0.18465960122148375, 0.17747440273037543, 0.0772408837794144, 0.0249685647566014, 0.0, 0.0], [0.1518243661100804, 0.08688930117501546, 0.07421150278293136, 0.11595547309833024, 0.1468769325912183, 0.29684601113172543, 0.11193568336425479, 0.015460729746444033, 0.0, 0.0], [0.16609353507565336, 0.09938101788170564, 0.09594222833562586, 0.1454607977991747, 0.1643741403026135, 0.22077028885832187, 0.09766162310866575, 0.01031636863823934, 0.0, 0.0], [0.16033653846153847, 0.13052884615384616, 0.0985576923076923, 0.15336538461538463, 0.1658653846153846, 0.196875, 0.090625, 0.0038461538461538464, 0.0, 0.0], [0.16817466561762393, 0.13709677419354838, 0.16089693154996065, 0.12726199842643587, 0.1414240755310779, 0.1966955153422502, 0.06805664830841857, 0.0003933910306845004, 0.0, 0.0], [0.18504555450832547, 0.16148287778825007, 0.09927741124725102, 0.12315425699026077, 0.1787621740496387, 0.1470311027332705, 0.10461828463713478, 0.0006283380458686773, 0.0, 0.0], [0.20989917506874428, 0.11793461655973113, 0.11060189428658723, 0.12923923006416133, 0.17659639474488237, 0.15368163764130766, 0.10204705163458601, 0.0, 0.0, 0.0], [0.18096856414613424, 0.10875106202209006, 0.13820447465307278, 0.16595865193996034, 0.14216935712262815, 0.14868309260832624, 0.11526479750778816, 0.0, 0.0, 0.0], [0.22313883299798792, 0.12193158953722334, 0.17484909456740444, 0.14124748490945674, 0.14507042253521127, 0.14104627766599598, 0.05271629778672032, 0.0, 0.0, 0.0], [0.19042089985486213, 0.1811320754716981, 0.1416545718432511, 0.13236574746008709, 0.13062409288824384, 0.16632801161103047, 0.057474600870827286, 0.0, 0.0, 0.0], [0.209147771696638, 0.13799843627834246, 0.11962470680218922, 0.16849100860046912, 0.15363565285379202, 0.17318217357310398, 0.03792025019546521, 0.0, 0.0, 0.0], [0.17363877822045154, 0.15637450199203187, 0.15770252324037184, 0.15139442231075698, 0.17264276228419656, 0.1394422310756972, 0.04880478087649402, 0.0, 0.0, 0.0], [0.12643979057591623, 0.09554973821989529, 0.08900523560209424, 0.1612565445026178, 0.13272251308900523, 0.3824607329842932, 0.012565445026178011, 0.0, 0.0, 0.0], [0.2164989939637827, 0.15412474849094568, 0.14245472837022133, 0.17142857142857143, 0.1557344064386318, 0.15412474849094568, 0.005633802816901409, 0.0, 0.0, 0.0], [0.20608261461643212, 0.1693145710394916, 0.1525192918747163, 0.16568315932818883, 0.17067635043123014, 0.13345438039037677, 0.0022696323195642307, 0.0, 0.0, 0.0], [0.2906101978266927, 0.19643354694901086, 0.1100585121203678, 0.12872666480913902, 0.14628030091947616, 0.12789077737531346, 0.0, 0.0, 0.0, 0.0], [0.22551319648093843, 0.21964809384164222, 0.14633431085043988, 0.19765395894428153, 0.12023460410557185, 0.0906158357771261, 0.0, 0.0, 0.0, 0.0], [0.21826965305226176, 0.16600790513833993, 0.17083882301273606, 0.21256038647342995, 0.1686429512516469, 0.06368028107158542, 0.0, 0.0, 0.0, 0.0], [0.21252869793374876, 0.16038045260741227, 0.17153164972122006, 0.16431616923581502, 0.232863233847163, 0.05837979665464087, 0.0, 0.0, 0.0, 0.0], [0.1989010989010989, 0.17545787545787545, 0.19633699633699633, 0.19633699633699633, 0.1652014652014652, 0.06776556776556776, 0.0, 0.0, 0.0, 0.0], [0.2593516209476309, 0.22416181767802715, 0.14740925464117485, 0.19340537545026323, 0.14103629814353005, 0.03463563313937379, 0.0, 0.0, 0.0, 0.0], [0.2605042016806723, 0.1726890756302521, 0.16008403361344536, 0.20756302521008405, 0.1638655462184874, 0.03529411764705882, 0.0, 0.0, 0.0, 0.0], [0.3559703894431928, 0.1789507563566141, 0.145799806887673, 0.20276794335371742, 0.10846475700032185, 0.00804634695848085, 0.0, 0.0, 0.0, 0.0], [0.2730512249443207, 0.19955456570155902, 0.18262806236080179, 0.18574610244988865, 0.15322939866369711, 0.005790645879732739, 0.0, 0.0, 0.0, 0.0], [0.21829457364341084, 0.19968992248062015, 0.20186046511627906, 0.26232558139534884, 0.1172093023255814, 0.00062015503875969, 0.0, 0.0, 0.0, 0.0], [0.3244370308590492, 0.22226855713094246, 0.19683069224353628, 0.15095913261050875, 0.10550458715596331, 0.0, 0.0, 0.0, 0.0, 0.0], [0.38871014916642294, 0.23398654577361802, 0.20941795846738812, 0.10587891196256215, 0.062006434630008773, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3865389109605048, 0.2523954194905352, 0.2002804393549895, 0.10586585650853003, 0.054919373685440524, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2969881814715974, 0.2752573389248952, 0.2051086542127335, 0.14944719786504004, 0.07319862752573389, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3314775160599572, 0.24539614561027837, 0.2423982869379015, 0.1297644539614561, 0.050963597430406855, 0.0, 0.0, 0.0, 0.0, 0.0], [0.14397905759162305, 0.6784238082116285, 0.11160099200881786, 0.04932488288784789, 0.016671259300082668, 0.0, 0.0, 0.0, 0.0, 0.0], [0.35398230088495575, 0.26595249184909175, 0.21238938053097345, 0.13786679087098277, 0.029809035863996275, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3661805281828932, 0.26054394954670873, 0.22506897910918408, 0.12613322822230982, 0.02207331493890422, 0.0, 0.0, 0.0, 0.0, 0.0], [0.39477977161500816, 0.29853181076672103, 0.16761827079934746, 0.132952691680261, 0.006117455138662317, 0.0, 0.0, 0.0, 0.0, 0.0], [0.369727047146402, 0.25168380007089686, 0.26657213753987946, 0.11024459411556185, 0.001772421127259837, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3682294037323623, 0.26217569412835684, 0.2594446973145198, 0.11015020482476104, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.40233463035019457, 0.31439688715953307, 0.19494163424124514, 0.08832684824902724, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4636604774535809, 0.2986737400530504, 0.1687002652519894, 0.06896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3558362369337979, 0.35322299651567945, 0.21254355400696864, 0.078397212543554, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4544559330379124, 0.30428360413589367, 0.1708517971442639, 0.07040866568193008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6499416569428238, 0.2175029171528588, 0.11061843640606768, 0.02193698949824971, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.41355754072517076, 0.33841303205465056, 0.21965317919075145, 0.02837624802942722, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6384514435695539, 0.25196850393700787, 0.09973753280839895, 0.00984251968503937, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45533254624163716, 0.3876426603699331, 0.1487603305785124, 0.008264462809917356, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5840867992766727, 0.2879746835443038, 0.12477396021699819, 0.0031645569620253164, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5848909657320872, 0.2585669781931464, 0.15654205607476634, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5198198198198198, 0.34864864864864864, 0.13153153153153152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5175913396481732, 0.40730717185385656, 0.07510148849797023, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4664516129032258, 0.4309677419354839, 0.10258064516129033, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6561302681992337, 0.2547892720306513, 0.08908045977011494, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6085972850678733, 0.2873303167420814, 0.10407239819004525, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6654870940882598, 0.2785179017485429, 0.05599500416319734, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5944272445820433, 0.3539731682146543, 0.05159958720330237, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5004775549188156, 0.49283667621776506, 0.0066857688634192934, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6883629191321499, 0.3076923076923077, 0.0039447731755424065, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7360335195530726, 0.2639664804469274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7442348008385744, 0.2557651991614256, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7082352941176471, 0.2917647058823529, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8122415219189413, 0.18775847808105872, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7263626251390434, 0.27363737486095663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8803921568627451, 0.11960784313725491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9403409090909091, 0.05965909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9110576923076923, 0.0889423076923077, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9561200923787528, 0.04387990762124711, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.996969696969697, 0.0030303030303030303, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
5000: ([0.5214682489427902, 0.021505496497488272, 0.013098723402141983, 0.012530077400418662, 0.011127086173459378, 0.010311364725572434, 0.010250527052898977, 0.009124409315861746, 0.009012666651767644, 0.011197856527385645, 0.006196751516596269, 0.007485516909148247, 0.00847133552348955, 0.006202959442379275, 0.00558589161954851, 0.005931052293083626, 0.007573669455266928, 0.0064624507401089114, 0.005868973035253569, 0.006378022949460035, 0.00558589161954851, 0.011593922192341406, 0.006512114146372957, 0.00769534480061384, 0.004767687001348361, 0.0047850691935407775, 0.005978232529034469, 0.007199952323129986, 0.004775136512287968, 0.004895570272478279, 0.005322675566349069, 0.007162704768431952, 0.00510539816394387, 0.004145652837891193, 0.004776378097444569, 0.005598307471114521, 0.0038836783698483527, 0.0036788168190091653, 0.005573475767982498, 0.006666070705791498, 0.003874987273752145, 0.004675809699759877, 0.004283468790273919, 0.005650454047691769, 0.004144411252734592, 0.004772653341974766, 0.003805458504982481, 0.00506194268346283, 0.004062466632398916, 0.005474148955454408, 0.00748800007946145, 0.005330125077288676, 0.004145652837891193, 0.01012388536692566, 0.0036887495002619745, 0.003932100190955797, 0.004123304305072372, 0.004480880830173499, 0.0036415692643111315, 0.004156827104300603, 0.00341063442518332, 0.003965622990184028, 0.0035161691634944168, 0.006637514247189672, 0.0034528483205077585, 0.004751546394312547, 0.004544201673160157, 0.00402273590738768, 0.002707897226547077, 0.0025266257936833116, 0.0032839927392100043, 0.002881719148471236, 0.002624711021054801, 0.002814673550014775, 0.007286863284092066, 0.002479445557732468, 0.0028196398906411794, 0.002149183906076566, 0.0019530134513335866, 0.0018710688309979116, 0.0013756763535140585, 0.0039358249464256, 0.0022634097404838705, 0.0017146291012661685, 0.0019703956435260027, 0.0019492886958637832, 0.0010938365229656007, 0.0016724152059417298, 0.0020771719669937, 0.0008256541291397553, 0.0007263273166116644, 0.0005922361196987418, 0.00046931918919522936, 0.00041841419777458277, 0.00033398640712570553, 0.00018872094380337264, 0.00018127143286376582, 8.815254611868064e-05, 2.1106947662219308e-05, 0.0] ,
[[0.13848743577411537, 0.0826329398431436, 0.05766639206479969, 0.0825377022014181, 0.081551992609559, 0.08359722096561445, 0.12353988790529569, 0.11989466716825158, 0.13158032580797235, 0.09851143565983018], [0.07909474048842445, 0.06310259222908608, 0.061428323999769066, 0.07857514000346401, 0.08359794469141504, 0.14514173546561976, 0.15743894694301716, 0.12949598752958835, 0.13342185786040067, 0.0687027307892154], [0.07725118483412323, 0.06113744075829384, 0.054597156398104266, 0.09175355450236967, 0.08587677725118484, 0.10085308056872037, 0.13402843601895734, 0.14720379146919432, 0.17924170616113744, 0.06805687203791469], [0.06906460562822037, 0.07362267142290924, 0.05846214823622672, 0.070055489496631, 0.07481173206500198, 0.20412207689258818, 0.15893777249306382, 0.11781609195402298, 0.13386841062227506, 0.039239001189060645], [0.06672617719259094, 0.0644945324704307, 0.06360187458156662, 0.06683775942869895, 0.08982370006694934, 0.20051327828609686, 0.14873912073197948, 0.12296362419102878, 0.14360633787101093, 0.0326935951796474], [0.10764599638771824, 0.10282962071041542, 0.05502709211318483, 0.10860927152317881, 0.08573148705599037, 0.09199277543648404, 0.12667068031306442, 0.15809753160746537, 0.1411198073449729, 0.022275737507525588], [0.12560562015503876, 0.1247577519379845, 0.05571705426356589, 0.10234980620155039, 0.06468023255813954, 0.10222868217054264, 0.140140503875969, 0.15588662790697674, 0.11785368217054264, 0.010780038759689923], [0.10681725404816982, 0.0847734385630698, 0.044904068580759286, 0.09824465913729759, 0.07116614505374881, 0.128316777792897, 0.13580078922302355, 0.18206558715471494, 0.1406994148863791, 0.007211865559940128], [0.16324562611930019, 0.0779721724755476, 0.05896128943380631, 0.05730816916930707, 0.0821049731367957, 0.09505441520870643, 0.13624466179914588, 0.15883730541396887, 0.1686182669789227, 0.0016531202644992423], [0.05466237942122187, 0.042909413460472334, 0.03725468455482869, 0.09213881805078168, 0.07783568023062423, 0.3061315001663156, 0.1469120745093691, 0.12240824925158, 0.11974720035480652, 0.0], [0.07633740733319976, 0.06090963734722501, 0.0546984572230014, 0.06972550591063915, 0.10238429172510519, 0.18573432177920257, 0.16690042075736325, 0.17110799438990182, 0.11220196353436185, 0.0], [0.08525460275335876, 0.09437717697793996, 0.07546856858517167, 0.09006468734450157, 0.10764637585005805, 0.16387460607065849, 0.142809752861171, 0.1736606402388456, 0.06684358931829491, 0.0], [0.0641946357907079, 0.07122966437051151, 0.051150520298988714, 0.06140993697786897, 0.08280814890810494, 0.30177341345449216, 0.17338414187307635, 0.14289901802726074, 0.051150520298988714, 0.0], [0.08046437149719776, 0.05344275420336269, 0.06084867894315452, 0.072257806244996, 0.11509207365892714, 0.2099679743795036, 0.1833466773418735, 0.17133706965572457, 0.053242594075260205, 0.0], [0.09157590575683486, 0.06601466992665037, 0.06445876861524784, 0.09513225161146921, 0.16403645254501, 0.12491664814403201, 0.17314958879751055, 0.17114914425427874, 0.049566570348966436, 0.0], [0.07284906845300397, 0.0906426627590538, 0.07222105924220222, 0.1297885702323634, 0.10592422022189658, 0.1735398785848859, 0.17333054218128532, 0.14883818296001675, 0.032865815365292025, 0.0], [0.12901639344262295, 0.060163934426229505, 0.06409836065573771, 0.07360655737704919, 0.1001639344262295, 0.28114754098360656, 0.1483606557377049, 0.13065573770491803, 0.012786885245901639, 0.0], [0.1780979827089337, 0.09452449567723344, 0.07377521613832853, 0.09125840537944284, 0.09221902017291066, 0.17118155619596542, 0.16560999039385207, 0.1254562920268972, 0.007877041306436119, 0.0], [0.143008250475989, 0.08885127988153163, 0.08546646921937805, 0.11275650518299132, 0.11508356251322191, 0.15654749312460334, 0.17410619843452507, 0.12291093716945209, 0.0012693039983075946, 0.0], [0.09713840763091298, 0.09149308935176173, 0.09013042631886314, 0.10803971189410161, 0.1637142300953864, 0.17909285575238465, 0.17403153591590423, 0.09635974304068523, 0.0, 0.0], [0.13291842631695933, 0.097799511002445, 0.09135363414092021, 0.10691264725494555, 0.10935763503000667, 0.17870637919537674, 0.17781729273171815, 0.10513447432762836, 0.0, 0.0], [0.1670593274791176, 0.1052687941743414, 0.09455986292568001, 0.05225958449346755, 0.11715570786035553, 0.31259370314842577, 0.09616620261297923, 0.0549368173056329, 0.0, 0.0], [0.09494756911344138, 0.05700667302192564, 0.06158245948522402, 0.08484270734032412, 0.15805529075309818, 0.353860819828408, 0.14451858913250715, 0.045185891325071496, 0.0, 0.0], [0.14875766376250404, 0.0680864795095192, 0.1185866408518877, 0.15004840271055178, 0.17247499193288157, 0.1868344627299129, 0.11374636979670862, 0.041464988706034205, 0.0, 0.0], [0.128125, 0.07447916666666667, 0.06380208333333333, 0.103125, 0.134375, 0.29140625, 0.16458333333333333, 0.04010416666666667, 0.0, 0.0], [0.12584327970939285, 0.07524649714582252, 0.07732226258432798, 0.12065386611312921, 0.15049299429164503, 0.23274519979242345, 0.18240788790866633, 0.03528801245459263, 0.0, 0.0], [0.13894080996884736, 0.1133956386292835, 0.08764278296988577, 0.1391484942886812, 0.1601246105919003, 0.21349948078920042, 0.13291796469366562, 0.014330218068535825, 0.0, 0.0], [0.14778410070701845, 0.12053802379720642, 0.1438178996378686, 0.11657182272805656, 0.1396792550439731, 0.2191757199517158, 0.10932919468873943, 0.0031039834454216243, 0.0, 0.0], [0.1544461778471139, 0.13442537701508062, 0.08606344253770151, 0.11024440977639105, 0.17368694747789912, 0.18616744669786792, 0.15418616744669786, 0.00078003120124805, 0.0, 0.0], [0.1747400456505199, 0.0996703018006594, 0.0933299518133401, 0.11666243976667512, 0.17727618564544764, 0.1993406036013188, 0.13898047172203906, 0.0, 0.0, 0.0], [0.1490552834149755, 0.09097270818754374, 0.11803125728947983, 0.14508980639141592, 0.1497550734779566, 0.19290879402845812, 0.15418707721017028, 0.0, 0.0, 0.0], [0.19258103657479633, 0.1060842433697348, 0.15427283758016988, 0.1315652626105044, 0.1502860114404576, 0.18651412723175595, 0.07869648119258103, 0.0, 0.0, 0.0], [0.1602626459143969, 0.15393968871595332, 0.1242704280155642, 0.1247568093385214, 0.14275291828793774, 0.20598249027237353, 0.08803501945525292, 0.0, 0.0, 0.0], [0.16232404911650195, 0.11171009284216832, 0.10092842168313866, 0.14824797843665768, 0.1775980832584606, 0.22970949386043726, 0.06948188080263552, 0.0, 0.0, 0.0], [0.1367299194177281, 0.1260722641018976, 0.13465037691707823, 0.13465037691707823, 0.20769430725240448, 0.19599688068624901, 0.06420587470756434, 0.0, 0.0, 0.0], [0.10778443113772455, 0.08272344200487913, 0.07784431137724551, 0.1483699268130406, 0.15790640940341538, 0.3896651142160124, 0.03570636504768241, 0.0, 0.0, 0.0], [0.17263427109974425, 0.12436061381074169, 0.12084398976982097, 0.15824808184143221, 0.19501278772378516, 0.2113171355498721, 0.01758312020460358, 0.0, 0.0, 0.0], [0.15491056361795477, 0.12959838002024976, 0.12959838002024976, 0.1501856226797165, 0.2291596355045562, 0.1910226122173473, 0.015524805939925751, 0.0, 0.0, 0.0], [0.23323680106928046, 0.1601693027400312, 0.1000222766763199, 0.12586322120739585, 0.20427712185341948, 0.17509467587435953, 0.0013366005791935842, 0.0, 0.0, 0.0], [0.14453343266902588, 0.14434717824548332, 0.11417396163158874, 0.17303035947103743, 0.22127025516856025, 0.20264481281430433, 0.0, 0.0, 0.0, 0.0], [0.16084588272989425, 0.12495994873438, 0.14322332585709707, 0.199935917975008, 0.2306952899711631, 0.14033963473245756, 0.0, 0.0, 0.0, 0.0], [0.17365905469994689, 0.13595326606479022, 0.15135422198619225, 0.16702071163037704, 0.26447158789166225, 0.10754115772703134, 0.0, 0.0, 0.0, 0.0], [0.15884057971014492, 0.14521739130434783, 0.1736231884057971, 0.20173913043478262, 0.2234782608695652, 0.09710144927536232, 0.0, 0.0, 0.0, 0.0], [0.21050318611294222, 0.1874313337727972, 0.13249835201054713, 0.1995165897604922, 0.20566908371786422, 0.06438145462535706, 0.0, 0.0, 0.0, 0.0], [0.19233073696824446, 0.14469742360695026, 0.1506890353505093, 0.22917914919113241, 0.21689634511683642, 0.06620730976632715, 0.0, 0.0, 0.0, 0.0], [0.29136316337148804, 0.1571279916753382, 0.1404786680541103, 0.22398543184183142, 0.16155046826222685, 0.025494276795005204, 0.0, 0.0, 0.0, 0.0], [0.20554649265905384, 0.16019575856443719, 0.16052202283849917, 0.21892332789559543, 0.22414355628058727, 0.03066884176182708, 0.0, 0.0, 0.0, 0.0], [0.1768457198920775, 0.16777041942604856, 0.18763796909492272, 0.27691930340936965, 0.18003433897473634, 0.010792249202845229, 0.0, 0.0, 0.0, 0.0], [0.24724938875305624, 0.17512224938875307, 0.17512224938875307, 0.2139364303178484, 0.1873471882640587, 0.0012224938875305623, 0.0, 0.0, 0.0, 0.0], [0.3041506010433205, 0.18847811295078248, 0.19664322975731457, 0.17396234973916988, 0.1365388977092311, 0.00022680880018144704, 0.0, 0.0, 0.0, 0.0], [0.2865196484828387, 0.19300281876968994, 0.17161333112253357, 0.23644503399104627, 0.11241916763389156, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19240624272070814, 0.19310505474027487, 0.18332168646634056, 0.3060796645702306, 0.12508735150244585, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23959269242288112, 0.1931716082659479, 0.23390236597783767, 0.22941000299490866, 0.10392333033842467, 0.0, 0.0, 0.0, 0.0, 0.0], [0.13048810399803779, 0.6133186166298749, 0.12288447387785136, 0.10203581064508217, 0.03127299484915379, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2662403231235274, 0.21575227196230226, 0.2231571861326153, 0.23426455738808483, 0.060585661393470214, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3040732554467951, 0.22734449005367857, 0.2447110830438901, 0.18629617934954215, 0.03757499210609409, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3065341764528756, 0.24751580849141824, 0.21439325504366155, 0.21017765733212887, 0.021379102679915687, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3028539761706844, 0.23247436963147686, 0.28401219174286507, 0.17290108063175394, 0.0077583818232197285, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2904875554040232, 0.22059324923286738, 0.29457892942379815, 0.1922945789294238, 0.002045687009887487, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3270609318996416, 0.2807646356033453, 0.25149342891278376, 0.14068100358422939, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.33891518019657807, 0.24936294139060794, 0.265744448489261, 0.14597742992355298, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.28240450845335, 0.31559173450219163, 0.2802128991859737, 0.12179085785848466, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.355225988700565, 0.2817796610169492, 0.2560028248587571, 0.10699152542372882, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5396558174335953, 0.22166105499438832, 0.17676767676767677, 0.061915450804339696, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.321826681049982, 0.3268608414239482, 0.28694714131607335, 0.0643653362099964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5322707081264698, 0.25633655604912464, 0.17820747321661876, 0.03318526260778678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3510928961748634, 0.3721311475409836, 0.25191256830601094, 0.024863387978142075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4567901234567901, 0.3101851851851852, 0.22530864197530864, 0.007716049382716049, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.41173773498395233, 0.30536451169188444, 0.2773956900504356, 0.005502063273727648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36805896805896804, 0.4152334152334152, 0.2167076167076167, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3659735349716446, 0.4506616257088847, 0.1833648393194707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3864713485566566, 0.46057733735458856, 0.15295131408875484, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4791863765373699, 0.38174077578051085, 0.1390728476821192, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.48963387737097486, 0.3850904278782532, 0.12527569475077194, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6176520701993525, 0.3118078037144318, 0.07054012608621571, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5292939409113671, 0.4041061592388583, 0.06659989984977466, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44077498899163364, 0.5024218405988551, 0.05680317040951123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6088965915655691, 0.36626227614095896, 0.02484113229347198, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6026700572155118, 0.39669421487603307, 0.0006357279084551812, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6264100862641009, 0.3735899137358991, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6958483754512635, 0.30415162454873645, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7564668769716089, 0.24353312302839117, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7707076247942951, 0.2292923752057049, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8124547429398986, 0.1875452570601014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8853182104599874, 0.1146817895400126, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9273885350318471, 0.07261146496815286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.94211123723042, 0.05788876276958002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9806978470675576, 0.019302152932442463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9898386132695756, 0.010161386730424387, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
10000: ([0.49008882614259314, 0.020408208213940397, 0.012947296782392469, 0.012380157389440856, 0.010936429730878593, 0.009961830968233587, 0.009856111780421247, 0.008900234123951342, 0.008815438525393529, 0.010671030519808031, 0.006367378582614039, 0.007520378474692368, 0.008170110983122373, 0.006126206685417139, 0.005632850475626221, 0.005888338512839375, 0.007271497886717485, 0.006351961201058073, 0.0064257443842187685, 0.00644446549039387, 0.005825567745075798, 0.010943037180116863, 0.006512742465856006, 0.00810954269843822, 0.004838855325493962, 0.004952283204084284, 0.006076650816130105, 0.007066666960331077, 0.004859778914748487, 0.004869690088605895, 0.005356438849158541, 0.007035832197219145, 0.005099849570405675, 0.004332284217226501, 0.004945675754846013, 0.005666988963357289, 0.00416709798626972, 0.003962267059883312, 0.005511713906257915, 0.006587626890556413, 0.004194629024762518, 0.004809121803921741, 0.004429193472721146, 0.005752885803454815, 0.004672567852997469, 0.004846564016271945, 0.0039864943737569734, 0.005338818984523152, 0.004387346294212095, 0.005474271693907712, 0.007152563800428603, 0.005298073047553812, 0.004693491442251995, 0.009668900718670229, 0.003905002499818295, 0.004174806677047703, 0.004381840086513536, 0.0048443615331925215, 0.0037210951626864127, 0.004234273720192145, 0.003870864012087227, 0.004277222140240907, 0.003753031167338057, 0.007030325989520586, 0.003821308142800193, 0.004902727334797251, 0.004716617514585944, 0.0044633319604522135, 0.00391381243213599, 0.0029138851140776112, 0.0035856424533018526, 0.003325749449929851, 0.0034733158162512418, 0.0035074543039823095, 0.007104109172681281, 0.0028951640079025094, 0.004728731171522775, 0.003835624282816447, 0.00345459471007614, 0.003204612880561545, 0.003539390308633954, 0.007848548453526507, 0.0034678096085526823, 0.002402909039651303, 0.0027486988831208303, 0.0028312919985992208, 0.0018511870282556554, 0.002095662650071691, 0.002905075181759916, 0.0018159472989848755, 0.0012587190798906687, 0.0012025557613653632, 0.0009944211103598197, 0.0009602826226287516, 0.0021331048624218946, 0.0005274946975219863, 0.0003711183988829006, 0.00015747754017879758, 0.00025328555413373034, 5.506207698559356e-06] ,
[[0.13069817294447827, 0.07798522806173924, 0.05442293043437228, 0.07789984113537647, 0.0769718200672759, 0.07894021342237542, 0.11692391350753764, 0.11531279702853496, 0.13667525778987177, 0.13416982560843801], [0.07392618173969351, 0.05897906324195985, 0.057414202460608674, 0.07344053529030865, 0.07813511763436218, 0.1356572415281675, 0.1484459313619685, 0.12459529462551262, 0.14925534211094324, 0.10015109000647529], [0.0693204048651867, 0.05486093391171217, 0.04899208981883133, 0.08233392872331377, 0.07723058603385217, 0.09092455558390745, 0.12248022454707834, 0.13966147826826572, 0.2058348218082844, 0.10836097643956792], [0.061999644191425014, 0.06609144280377158, 0.05248176481053193, 0.06288916562889166, 0.06715886852873154, 0.1835972246931151, 0.14490304216331615, 0.11599359544565024, 0.17247820672478206, 0.07240704500978473], [0.06021548685932937, 0.05820159097774645, 0.05739603262511328, 0.06031618165340852, 0.08105930923371261, 0.181250629342463, 0.13603866680092638, 0.12214278521800423, 0.1795388178431175, 0.06384049944617863], [0.09882821136413884, 0.09440636745522883, 0.050519566659296924, 0.09971258014592085, 0.07870882157859828, 0.08500994914879505, 0.11739995578156091, 0.15542781339818704, 0.18041123148352864, 0.03957550298474464], [0.11586592178770949, 0.11508379888268157, 0.05139664804469274, 0.09452513966480447, 0.05966480446927374, 0.09497206703910614, 0.13150837988826816, 0.15631284916201119, 0.15832402234636872, 0.0223463687150838], [0.09712942341004702, 0.07708487998020291, 0.04083147735708983, 0.08933432318732987, 0.06483543677307597, 0.11705023509032418, 0.12608265280871073, 0.18386537985647117, 0.18770106409304627, 0.016085127443702055], [0.14803247970018737, 0.07070580886945659, 0.05346658338538413, 0.052092442223610244, 0.07470331043098064, 0.08644597126795753, 0.12604622111180513, 0.16427232979387882, 0.2192379762648345, 0.004996876951905059], [0.05087719298245614, 0.039938080495356035, 0.0346749226006192, 0.08575851393188855, 0.07244582043343653, 0.2853457172342621, 0.13890608875128999, 0.13374613003095975, 0.15830753353973168, 0.0], [0.06589415427187824, 0.05257696298858527, 0.04721549636803874, 0.06018678657903839, 0.08872362504323764, 0.1618817018332757, 0.15185057073676927, 0.19197509512279487, 0.17969560705638188, 0.0], [0.07526724264167521, 0.08332113047298287, 0.06662761751354518, 0.07951383804363743, 0.09503587640943037, 0.14731293015082736, 0.13603748718699663, 0.2004685898374579, 0.11641528774344706, 0.0], [0.059037606146380914, 0.06550748079255965, 0.04704138023992452, 0.05647661409893517, 0.07629060520285753, 0.2787437660062003, 0.16686885024935974, 0.1729343577301523, 0.07709933953362987, 0.0], [0.0722631673557433, 0.04799568578105339, 0.05464677332374618, 0.06489304332194859, 0.10354125471867698, 0.1905446701420097, 0.17346755347833903, 0.20240877224519144, 0.09023907963329139, 0.0], [0.08054740957966765, 0.05806451612903226, 0.056695992179863146, 0.08367546432062561, 0.14447702834799608, 0.11280547409579668, 0.16500488758553275, 0.2179863147605083, 0.08074291300097752, 0.0], [0.06508322423789041, 0.08097998877875444, 0.06452216195997756, 0.1161398915279596, 0.09500654572657564, 0.15709743781559754, 0.1638301851505517, 0.20647091827192818, 0.050869646530764916, 0.0], [0.11918824776616689, 0.0555807966076026, 0.05921550810237771, 0.06799939421475087, 0.0928365894290474, 0.26185067393608963, 0.14932606391034378, 0.1671967287596547, 0.02680599727396638, 0.0], [0.16071428571428573, 0.08529819694868239, 0.06657420249653259, 0.08269764216366159, 0.08373786407766991, 0.15759361997226076, 0.16608876560332872, 0.17735783633841887, 0.019937586685159502, 0.0], [0.11585261353898886, 0.07197943444730077, 0.06923736075407026, 0.09134532990574122, 0.09443016281062554, 0.13161953727506426, 0.16263924592973436, 0.2260497000856898, 0.03684661525278492, 0.0], [0.08526999316473001, 0.08031442241968557, 0.07911825017088175, 0.09501025290498975, 0.14490772385509226, 0.1659261790840738, 0.19053315105946686, 0.1584073820915926, 0.0005126452494873548, 0.0], [0.11304347826086956, 0.0831758034026465, 0.0776937618147448, 0.09187145557655954, 0.09432892249527411, 0.1667296786389414, 0.20170132325141776, 0.17145557655954632, 0.0, 0.0], [0.1569890308946362, 0.09892321626245346, 0.08885981684613062, 0.04910938915165543, 0.11039549159706148, 0.2978766227231559, 0.11582972728187582, 0.08201670524303109, 0.0, 0.0], [0.08420696652012175, 0.05055799797091647, 0.054616165032127156, 0.07592154210348326, 0.1403449442002029, 0.3239770037199865, 0.17534663510314508, 0.0950287453500169, 0.0, 0.0], [0.12520369364475828, 0.05744160782183596, 0.09980988593155894, 0.12656165127648017, 0.14652362846279196, 0.1656708310700706, 0.16132536664856056, 0.11746333514394351, 0.0, 0.0], [0.11197086936731906, 0.0650887573964497, 0.05575785161583978, 0.09126081019572144, 0.12107419208010924, 0.2690031861629495, 0.2091488393263541, 0.07669549385525717, 0.0, 0.0], [0.10784967756281966, 0.0644874360684901, 0.06648877029130532, 0.10406937958639093, 0.13275517011340893, 0.21681120747164775, 0.23882588392261508, 0.06871247498332221, 0.0, 0.0], [0.1212395795578108, 0.09894889452700253, 0.07647698441464298, 0.12196447988401594, 0.14099311344690105, 0.20061616527727438, 0.19246103660746647, 0.04729974628488583, 0.0, 0.0], [0.13355150381798347, 0.10892940626460963, 0.12996727442730246, 0.10628019323671498, 0.1294997662459093, 0.21832632071061245, 0.15910861773414367, 0.014336917562724014, 0.0, 0.0], [0.13460231135282122, 0.1171538635848629, 0.07500566508044414, 0.09789259007477906, 0.15635622025832766, 0.18649444822116473, 0.22615001133016088, 0.0063448900974393836, 0.0, 0.0], [0.1558118498417006, 0.08887381275440977, 0.08322026232473993, 0.1047037539574853, 0.1614654002713704, 0.2130257801899593, 0.19244685662596112, 0.0004522840343735866, 0.0, 0.0], [0.13137335526315788, 0.08018092105263158, 0.10423519736842106, 0.12808388157894737, 0.13630756578947367, 0.20230263157894737, 0.21751644736842105, 0.0, 0.0, 0.0], [0.17389262795429644, 0.0957896384410706, 0.13930192518390985, 0.11973704805133824, 0.14493660979809048, 0.20629206448583504, 0.12005008608545939, 0.0, 0.0, 0.0], [0.1423018786439214, 0.13668754048801554, 0.11034333837184193, 0.11315050744979487, 0.13172101058086808, 0.2293241200604621, 0.1364716044050961, 0.0, 0.0, 0.0], [0.13777325876970006, 0.09481443823080833, 0.08566344687341129, 0.12785968479918658, 0.16344687341128622, 0.26461616675139804, 0.12582613116420946, 0.0, 0.0, 0.0], [0.1171231351592073, 0.10799376530839457, 0.11556446225784903, 0.11779113783121799, 0.18971275885103542, 0.2533956802493877, 0.09841906034290804, 0.0, 0.0, 0.0], [0.09444228527011271, 0.07267780800621843, 0.06840264282938205, 0.13253012048192772, 0.14963078118927323, 0.41352506801399147, 0.06879129420909444, 0.0, 0.0, 0.0], [0.1427061310782241, 0.10280126849894292, 0.10042283298097252, 0.13372093023255813, 0.18340380549682875, 0.266384778012685, 0.07056025369978858, 0.0, 0.0, 0.0], [0.12757087270705947, 0.10672595886603668, 0.10700389105058365, 0.12896053362979434, 0.22234574763757642, 0.2726514730405781, 0.03474152306837132, 0.0, 0.0, 0.0], [0.20919080919080918, 0.14365634365634367, 0.09010989010989011, 0.11708291708291708, 0.2031968031968032, 0.23196803196803198, 0.004795204795204795, 0.0, 0.0, 0.0], [0.13005683717820127, 0.12972250083584086, 0.10414577064526914, 0.15947843530591777, 0.22751588097626213, 0.24908057505850886, 0.0, 0.0, 0.0, 0.0], [0.13179312155421372, 0.10238907849829351, 0.11892885271724862, 0.1730112890522447, 0.2449461800997637, 0.22893147807823575, 0.0, 0.0, 0.0, 0.0], [0.14975956033890542, 0.11747194870620563, 0.1309823677581864, 0.15456835356079687, 0.2770780856423174, 0.17013968399358828, 0.0, 0.0, 0.0, 0.0], [0.1364992541024366, 0.12481352560914968, 0.1496767777225261, 0.18125310790651417, 0.2583291894579811, 0.14942814520139233, 0.0, 0.0, 0.0, 0.0], [0.1833843797856049, 0.16347626339969373, 0.11753445635528331, 0.18434150076569678, 0.24846860643185298, 0.1027947932618683, 0.0, 0.0, 0.0, 0.0], [0.15130803676643884, 0.113834551025218, 0.12231911383455102, 0.20268677822295544, 0.30403016733443317, 0.1058213528164035, 0.0, 0.0, 0.0, 0.0], [0.25448761645080664, 0.13792319927289254, 0.12428993410588503, 0.21313337877755056, 0.21586003181095206, 0.0543058395819132, 0.0, 0.0, 0.0, 0.0], [0.17430939226519337, 0.13674033149171272, 0.13922651933701657, 0.21408839779005526, 0.2828729281767956, 0.05276243093922652, 0.0, 0.0, 0.0, 0.0], [0.14892739273927394, 0.1412953795379538, 0.16274752475247525, 0.26485148514851486, 0.2553630363036304, 0.026815181518151814, 0.0, 0.0, 0.0, 0.0], [0.20306224899598393, 0.1448293172690763, 0.1495983935742972, 0.20607429718875503, 0.27936746987951805, 0.01706827309236948, 0.0, 0.0, 0.0, 0.0], [0.27016696841681753, 0.16777308388654194, 0.17783142224904447, 0.18386642526654598, 0.20016093341380004, 0.0002011667672500503, 0.0, 0.0, 0.0, 0.0], [0.2662047729022325, 0.1798306389530408, 0.1615088529638183, 0.24973056197074672, 0.14272517321016168, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1716898773643733, 0.17293701933070046, 0.16836416545416752, 0.3151112034919975, 0.17189773435876118, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18840919755983107, 0.15391834819333647, 0.19474425152510558, 0.2702956358517128, 0.19263256687001407, 0.0, 0.0, 0.0, 0.0, 0.0], [0.12118451025056948, 0.5707289293849658, 0.11788154897494305, 0.1297266514806378, 0.060478359908883825, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2233502538071066, 0.18189509306260576, 0.19514946418499718, 0.2949802594472645, 0.10462492949802595, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2542864679504089, 0.19203376417831708, 0.21867581113162754, 0.25745185966763384, 0.07755209707201266, 0.0, 0.0, 0.0, 0.0, 0.0], [0.256848454385524, 0.20934908268409147, 0.1982910278964564, 0.28650414677054536, 0.049007288263382756, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2502841554898841, 0.19731757217549442, 0.2605137531257104, 0.26824278245055694, 0.02364173675835417, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25332938739271976, 0.19680378810298904, 0.28410772417875113, 0.26191180822728616, 0.0038472920982539215, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2855656697009103, 0.24889466840052016, 0.2543563068920676, 0.21118335500650195, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.26770981507823616, 0.21223328591749643, 0.26372688477951634, 0.25633001422475105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2350669412976313, 0.2698249227600412, 0.29351184346035014, 0.20159629248197733, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2963615023474178, 0.24882629107981222, 0.27816901408450706, 0.1766431924882629, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4523809523809524, 0.19815162907268172, 0.23856516290726817, 0.11090225563909774, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2622478386167147, 0.29164265129683, 0.3265129682997118, 0.11959654178674352, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4613656783468104, 0.24326145552560646, 0.23360287511230907, 0.061769991015274035, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3060938594443147, 0.3420499649778193, 0.3058603782395517, 0.045995797338314263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3742906489020479, 0.28892178633111276, 0.30816679003207503, 0.028620774734764372, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.26167698368036013, 0.2245357343837929, 0.5028137310073157, 0.010973550928531233, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.29516250944822375, 0.3835978835978836, 0.32086167800453513, 0.0003779289493575208, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3071253071253071, 0.43611793611793614, 0.25675675675675674, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3208609271523179, 0.43874172185430466, 0.24039735099337747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.34242232086239693, 0.43119847812301837, 0.22637920101458464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3924646781789639, 0.4392464678178964, 0.16828885400313973, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5715392962331421, 0.3380871182762362, 0.0903735854906216, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4359071890452644, 0.4682388740966147, 0.09585393685812096, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3274336283185841, 0.5992081974848626, 0.07335817419655333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.40597186333620444, 0.5475165087568188, 0.046511627906976744, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.37902454574434175, 0.6047178833280205, 0.01625757092763787, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45910652920962197, 0.5402061855670103, 0.0006872852233676976, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.40945861854387056, 0.5905413814561294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.39062719236705484, 0.6093728076329451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5741505239758653, 0.42584947602413464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7094408799266728, 0.29055912007332724, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8056891025641025, 0.19431089743589744, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7580707895760405, 0.24192921042395954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8316478286734087, 0.1683521713265913, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9432475039411455, 0.05675249605885444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9818043972706596, 0.01819560272934041, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9969678593086719, 0.0030321406913280777, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] ),
math.inf: ([0.4809035509952379, 0.020085691527630272, 0.012774504094008017, 0.012275600397409983, 0.010806725136181439, 0.009873154270444346, 0.009714704169378875, 0.00885714652780157, 0.008742591387166399, 0.010587250334029942, 0.0064065230052417, 0.007512461886327041, 0.008103437938949604, 0.006079987323991915, 0.005651743807598753, 0.005929031484463325, 0.007176290725958409, 0.00634014526020076, 0.006369051697557299, 0.006415087875569564, 0.006027527493233753, 0.010887020795505156, 0.006490030490938367, 0.00808737880708486, 0.004848787214361575, 0.005007237315427044, 0.006177412723971359, 0.007052100106204392, 0.0051453458494638395, 0.005013660968172942, 0.005648531981225805, 0.007048888279831443, 0.0051603343725376, 0.004523322141902771, 0.004953706875877899, 0.00563568467573401, 0.00416895063208743, 0.004004076878276063, 0.005438692658193155, 0.006563902497516188, 0.004282435163931618, 0.004808104080304224, 0.0044751447463085405, 0.005751310425160163, 0.004868058172599267, 0.005177464113193326, 0.004038336359587516, 0.005343408475795676, 0.004456944396861832, 0.005487940662578368, 0.0071345369831100755, 0.0053327023878858475, 0.004663571893521532, 0.009633337901264175, 0.0039955120079482, 0.004236398985919353, 0.004445167700161019, 0.004938718352804138, 0.0037267892014114907, 0.004333824385898797, 0.003954828873890849, 0.004349883517763541, 0.0037846020761245673, 0.007047817671040461, 0.003934487306862174, 0.005066120798931104, 0.0047684915550378565, 0.004536169447394566, 0.0040244184453047385, 0.0030458820103463633, 0.003787813902497516, 0.0034826903970673886, 0.0036604114563705504, 0.0036507759772517042, 0.007105630545753537, 0.0030137637466168764, 0.004767420946246874, 0.004019065401349824, 0.0035726215355099526, 0.00342059508719038, 0.00427922333755867, 0.008244758299359348, 0.004000865051903114, 0.0026518979752646545, 0.002993422179588201, 0.0029687981773955942, 0.0020084620918839288, 0.002186183151187091, 0.0031101185378053374, 0.0020138151358388436, 0.0014731576963924766, 0.0014763695227654253, 0.0011733872349172633, 0.001200152454691836, 0.0038509798211655076, 0.0007601322415978622, 0.0004774915207783754, 0.00022268662852444415, 0.0002837113296104697, 1.391791428277776e-05] ,
[[0.1294894999187421, 0.07726403480065096, 0.05391963703312874, 0.07717943751711426, 0.07625999861972853, 0.07821018862967985, 0.11584262233769009, 0.11426421512643954, 0.13598790704094285, 0.1415824589758831], [0.07302382602206706, 0.05825915462928415, 0.056713394808379086, 0.07254410745695858, 0.07718138691967379, 0.13400138585363253, 0.14663397473482223, 0.12312776504450722, 0.14897926549757476, 0.10953573903310058], [0.06830372108615487, 0.054056319141803554, 0.048273550117331546, 0.08112638283607107, 0.07609788803218237, 0.08959101575595038, 0.12068387529332886, 0.1376131411330875, 0.20549782098558497, 0.11875628561850486], [0.060788417931275075, 0.06480027908599337, 0.0514564800279086, 0.061660561660561664, 0.06584685156113727, 0.18001046572475143, 0.14207221350078492, 0.11372754229897088, 0.17259724402581544, 0.08703994418280132], [0.0592431147216168, 0.057261739647315235, 0.05646918961759461, 0.05934218347533188, 0.07975034674063801, 0.17832375668714087, 0.13384188626907073, 0.12026946701010502, 0.18040420051515751, 0.07509411531602933], [0.09694209499024073, 0.09260464107568857, 0.0495554109737584, 0.09780958577315116, 0.07720667967902842, 0.08338755150726523, 0.11515940143135979, 0.15256994144437216, 0.186402081977879, 0.04836261114725656], [0.11428256557196385, 0.11351113070310778, 0.050694291381970465, 0.09323341415031959, 0.0588494599955918, 0.09367423407538021, 0.1297112629490853, 0.1543971787524796, 0.16398501212254793, 0.027661450297553448], [0.09488698174785447, 0.07530520971836094, 0.03988879487489423, 0.0872718481808292, 0.06333857125589266, 0.11434787864136348, 0.12317176356823402, 0.17998307748096218, 0.19448809379910553, 0.027317780732503325], [0.14511388684790597, 0.06931178055351457, 0.05241244183198628, 0.05106539309331374, 0.07323046779328925, 0.08474161156012736, 0.12356110702914523, 0.16176830761694833, 0.23059025226549107, 0.008204751408278227], [0.04985337243401759, 0.03913439174840732, 0.033977146324198605, 0.08403276367681262, 0.07098796642734351, 0.27960359995955103, 0.13611083021539083, 0.13166144200626959, 0.17029022145818587, 0.004348265749823036], [0.06366978609625669, 0.05080213903743316, 0.0456216577540107, 0.05815508021390375, 0.08572860962566844, 0.15641711229946523, 0.1467245989304813, 0.1869986631016043, 0.20588235294117646, 0.0], [0.07325067692746187, 0.08108878438078951, 0.06484252529571041, 0.07738349722103463, 0.09248966794926607, 0.14336611087359272, 0.13253527148353997, 0.19695026364543253, 0.13809320222317228, 0.0], [0.05786761791518034, 0.06420927467300833, 0.04610912934337429, 0.05535737878187343, 0.07477870260272163, 0.2732197119830889, 0.16356189721231337, 0.1717532038578412, 0.09314308363059849, 0.0], [0.07078711040676175, 0.047015319598520865, 0.053530551153372075, 0.06356752949462934, 0.10142630744849446, 0.18665257967952104, 0.17010036978341259, 0.20073956682514527, 0.10618066561014262, 0.0], [0.07804508429626823, 0.056260655427164234, 0.05493464671339269, 0.08107596135631748, 0.1399886342110248, 0.10930100397802614, 0.1602576245501042, 0.21784428869103997, 0.10229210077666225, 0.0], [0.0628385698808234, 0.07818707114481763, 0.06229685807150596, 0.11213434452871072, 0.0917298663777537, 0.15167930660888407, 0.158540989526905, 0.21271217045864932, 0.06988082340195016, 0.0], [0.11741011487393704, 0.054751603759510666, 0.058332090108906456, 0.06698493211994629, 0.09145158884081754, 0.2579442040877219, 0.14724750111890197, 0.17246009249589736, 0.033417872594360735, 0.0], [0.15653495440729484, 0.08308004052684904, 0.06484295845997974, 0.08054711246200608, 0.08156028368794327, 0.1534954407294833, 0.16312056737588654, 0.19030732860520094, 0.0265113137453563, 0.0], [0.11363254328458565, 0.07060010085728693, 0.06791057320558078, 0.08959488989746175, 0.0926206085056312, 0.12909732728189613, 0.15985879979828543, 0.23802319717599596, 0.03866195999327618, 0.0], [0.08327770360480641, 0.07843791722296395, 0.07726969292389853, 0.09279038718291055, 0.14152202937249667, 0.1620493991989319, 0.18791722296395194, 0.17473297730307077, 0.0020026702269692926, 0.0], [0.10621669626998224, 0.07815275310834814, 0.07300177619893428, 0.08632326820603908, 0.0886323268206039, 0.15683836589698047, 0.19236234458259324, 0.21847246891651864, 0.0, 0.0], [0.1534074146917101, 0.09666633887304553, 0.08683253023896155, 0.04798898613432983, 0.10787688071590126, 0.29117907365522666, 0.11544891336414594, 0.10059986232667913, 0.0, 0.0], [0.08215110524579347, 0.04932365555922138, 0.05328274496865721, 0.07406796436819532, 0.1369185087429891, 0.3160673045199604, 0.17387000989772353, 0.11431870669745958, 0.0, 0.0], [0.12205454064072015, 0.05599682287529786, 0.09729944400317712, 0.12337834259994705, 0.14283823140058247, 0.16150383902568174, 0.16123907863383638, 0.1356897008207572, 0.0, 0.0], [0.1086332523735924, 0.06314859792448664, 0.05409582689335394, 0.08854051667034665, 0.11746522411128284, 0.2616471627290793, 0.21130492382424376, 0.09516449547361448, 0.0, 0.0], [0.10369895231986316, 0.06200555911909344, 0.06392986957451358, 0.10006414368184734, 0.1276459268762027, 0.20846696600384862, 0.2407526192003421, 0.09343596322428907, 0.0, 0.0], [0.11594454072790294, 0.09462738301559792, 0.0731369150779896, 0.11663778162911612, 0.13483535528596188, 0.19185441941074524, 0.20311958405545927, 0.06984402079722704, 0.0, 0.0], [0.1301047517838166, 0.10611811143160771, 0.12661302565659632, 0.10353727038105359, 0.12615758311826325, 0.21299529376043722, 0.1717018369515713, 0.022772126916654017, 0.0, 0.0], [0.12359550561797752, 0.1075738660008323, 0.0688722430295464, 0.0898876404494382, 0.14357053682896379, 0.17249271743653766, 0.2505201831044528, 0.04348730753225135, 0.0, 0.0], [0.14712790945974802, 0.08392056374119154, 0.07858210548793508, 0.09886824685030964, 0.15246636771300448, 0.2015801836429639, 0.23361093316250267, 0.003843689942344651, 0.0, 0.0], [0.1211144806671721, 0.07391963608794541, 0.09609552691432904, 0.11808188021228203, 0.1256633813495072, 0.18821076573161485, 0.27691432903714935, 0.0, 0.0, 0.0], [0.16874240583232078, 0.09295261239368165, 0.13517618469015796, 0.11619076549210207, 0.14064398541919806, 0.20215674362089914, 0.14413730255164034, 0.0, 0.0, 0.0], [0.13672199170124483, 0.13132780082987552, 0.10601659751037344, 0.1087136929460581, 0.12676348547717842, 0.22302904564315354, 0.16742738589211617, 0.0, 0.0, 0.0], [0.12828402366863906, 0.08828402366863905, 0.07976331360946745, 0.11905325443786982, 0.15266272189349112, 0.2539644970414201, 0.17798816568047338, 0.0, 0.0, 0.0], [0.1136805705640804, 0.10481953749729847, 0.11216771125999568, 0.11432893883725956, 0.18456883509833585, 0.2535119948130538, 0.11692241192997622, 0.0, 0.0, 0.0], [0.09232522796352584, 0.07104863221884498, 0.0668693009118541, 0.1295592705167173, 0.14703647416413373, 0.4101443768996961, 0.08301671732522796, 0.0, 0.0, 0.0], [0.1386748844375963, 0.09989727786337955, 0.09758602978941962, 0.12994350282485875, 0.17847971237801746, 0.2717000513610683, 0.08371854134566, 0.0, 0.0, 0.0], [0.12272727272727273, 0.10267379679144385, 0.10294117647058823, 0.12406417112299466, 0.2144385026737968, 0.2786096256684492, 0.05454545454545454, 0.0, 0.0, 0.0], [0.20610236220472442, 0.14153543307086613, 0.08877952755905512, 0.11535433070866141, 0.2017716535433071, 0.23996062992125985, 0.006496062992125984, 0.0, 0.0, 0.0], [0.12689610177785027, 0.1265698907192954, 0.10161474473984668, 0.15560267493068014, 0.2229652585222639, 0.25803294731691406, 0.008318381993149567, 0.0, 0.0, 0.0], [0.1255, 0.0975, 0.11325, 0.16475, 0.23475, 0.26425, 0.0, 0.0, 0.0, 0.0], [0.14562458249833, 0.11422845691382766, 0.12736584279670451, 0.15030060120240482, 0.27054108216432865, 0.19193943442440436, 0.0, 0.0, 0.0, 0.0], [0.13133971291866028, 0.12009569377990431, 0.14401913875598085, 0.17464114832535885, 0.2521531100478469, 0.1777511961722488, 0.0, 0.0, 0.0, 0.0], [0.17833209233060313, 0.15897244973938943, 0.1142963514519732, 0.17926284437825762, 0.24478778853313476, 0.12434847356664185, 0.0, 0.0, 0.0, 0.0], [0.14119199472179458, 0.10622388387948098, 0.1141411919947218, 0.18935561908950957, 0.29030129755883, 0.15878601275566306, 0.0, 0.0, 0.0, 0.0], [0.23159636062861869, 0.12551695616211744, 0.11311000827129859, 0.19396195202646815, 0.21836228287841192, 0.11745244003308519, 0.0, 0.0, 0.0, 0.0], [0.16728525980911982, 0.1312301166489926, 0.13361611876988336, 0.2059915164369035, 0.28950159066808057, 0.07237539766702014, 0.0, 0.0, 0.0, 0.0], [0.1446603886996594, 0.13724704468042476, 0.1580845521939491, 0.25806451612903225, 0.2660789420957724, 0.03586455620116209, 0.0, 0.0, 0.0, 0.0], [0.19433101128993513, 0.13860196973336536, 0.14316598606773961, 0.19841460485227, 0.29521979341820803, 0.030266634638481865, 0.0, 0.0, 0.0, 0.0], [0.26199765899336713, 0.16269996098322279, 0.17245415528677333, 0.17928209129925868, 0.22259071400702302, 0.0009754194303550526, 0.0, 0.0, 0.0, 0.0], [0.25945378151260506, 0.1752701080432173, 0.15741296518607442, 0.24474789915966386, 0.1631152460984394, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16583015458743225, 0.16703473198152982, 0.16281871110218832, 0.3057618952017667, 0.19855450712708292, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18434343434343434, 0.15059687786960516, 0.19077134986225897, 0.2676767676767677, 0.2066115702479339, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11824849966659258, 0.5569015336741499, 0.11502556123583019, 0.13125138919759946, 0.07857301622582796, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21221864951768488, 0.17282958199356913, 0.18542336548767416, 0.2877813504823151, 0.1417470525187567, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24361890320950214, 0.18397776092999749, 0.20975486479656305, 0.26105635582512005, 0.10159211523881728, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24614643545279383, 0.200626204238921, 0.19051059730250483, 0.2892581888246628, 0.07345857418111754, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2386733145458487, 0.18816388467374812, 0.24907869065683938, 0.2863646217212226, 0.03771948840234121, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2459063487503591, 0.19132433208848032, 0.276644642344154, 0.27635736857224935, 0.009767308244757253, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27124505928853754, 0.23641304347826086, 0.24703557312252963, 0.2349308300395257, 0.010375494071146246, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25473741201949107, 0.20194910665944776, 0.25446670276123445, 0.28884677855982677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.22471080482402167, 0.2579374846172779, 0.28624169333005167, 0.2311100172286488, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2857142857142857, 0.23988684582743988, 0.2727015558698727, 0.2016973125884017, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.43870575725353184, 0.1921616284368829, 0.23803736898070788, 0.13109524532887742, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24761904761904763, 0.27564625850340135, 0.3268027210884354, 0.14993197278911566, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4340659340659341, 0.22950126796280643, 0.2445054945054945, 0.091927303465765, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.29434216434665467, 0.3293668612483161, 0.31544678940278403, 0.06084418500224517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3580363464715601, 0.2773188576823224, 0.322397923058768, 0.04224687278734954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24740622505985635, 0.21335461558925245, 0.5163607342378292, 0.022878425113061984, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27451669595782074, 0.36203866432337434, 0.35711775043936733, 0.00632688927943761, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2826455624646693, 0.4053137365743358, 0.3120407009609949, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.29787888103289273, 0.41254226867506916, 0.2895788502920381, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.31617431997660134, 0.4036267914594911, 0.28019888856390757, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36686217008797656, 0.4252199413489736, 0.20791788856304985, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5562754256441164, 0.33825523579930694, 0.10546933855657677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4088809946714032, 0.4628774422735346, 0.12824156305506218, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.31596676397933976, 0.5984729395912868, 0.08556029642937346, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.37906233351092167, 0.5508790623335109, 0.07005860415556739, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3596044351213665, 0.6119268804315253, 0.02846868444710818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42065727699530514, 0.5683881064162755, 0.010954616588419406, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.33675256442331747, 0.6632474355766825, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3673548889754577, 0.6326451110245422, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5022745517795023, 0.49772544822049775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.64715381509891, 0.35284618490109004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7478540772532188, 0.2521459227467811, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7345834835917778, 0.26541651640822217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7921108742004265, 0.20788912579957355, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9324191968658179, 0.06758080313418217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9514629948364888, 0.048537005163511185, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.98989898989899, 0.010101010101010102, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] ),
},
    'curio_se': {500: ([0.7243125, 0.10115625, 0.03040625, 0.01840625, 0.0096875, 0.00796875, 0.006125, 0.00328125, 0.0028125, 0.0021875, 0.0020625, 0.0016875, 0.000875, 0.0015625, 0.00134375, 0.0006875, 0.0009375, 0.0004375, 0.001125, 0.00140625, 0.001375, 0.03259375, 0.0036875, 0.00221875, 0.0029375, 0.00125, 0.0010625, 0.00421875, 0.00215625, 0.000875, 0.002, 0.00184375, 0.0014375, 0.0010625, 0.002125, 0.001, 0.0013125, 0.0015, 0.00046875, 0.00121875, 0.00040625, 0.00096875, 0.000375, 0.00034375, 0.0005625, 0.00028125, 0.0005, 0.0003125, 0.0005625, 0.00034375, 0.0003125, 0.00040625, 0.0003125, 0.00296875, 0.00021875, 0.001, 0.00034375, 0.00084375, 0.00015625, 0.00040625, 0.000375, 0.00025, 6.25e-05, 0.0005, 0.00021875, 0.00021875, 0.00015625, 0.000125, 0.00034375, 0.0001875, 0.00021875, 6.25e-05, 6.25e-05, 0.000125, 0.000125, 3.125e-05, 0.000125, 3.125e-05, 0.00015625, 0.0, 3.125e-05, 0.0, 0.0, 0.0, 3.125e-05, 0.000125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[[0.2710760203641384, 0.12140823194408491, 0.10069893864871861, 0.08201743032185693, 0.15376650271809475, 0.14824402450599708, 0.08900681680904306, 0.02601604970230391, 0.007765984985762361, 0.0], [0.21501390176088972, 0.17207290701266606, 0.10874266295953043, 0.09731232622798888, 0.1566265060240964, 0.14118010503552672, 0.07476058078467716, 0.025332097621254247, 0.008958912573370404, 0.0], [0.2682425488180884, 0.13463514902363824, 0.11408016443987667, 0.08016443987667009, 0.13052415210688592, 0.1500513874614594, 0.08633093525179857, 0.029804727646454265, 0.006166495375128468, 0.0], [0.266553480475382, 0.12903225806451613, 0.11205432937181664, 0.06791171477079797, 0.1460101867572156, 0.15789473684210525, 0.1035653650254669, 0.013582342954159592, 0.003395585738539898, 0.0], [0.25806451612903225, 0.13870967741935483, 0.06129032258064516, 0.08387096774193549, 0.22258064516129034, 0.14516129032258066, 0.08387096774193549, 0.0064516129032258064, 0.0, 0.0], [0.3333333333333333, 0.1411764705882353, 0.058823529411764705, 0.047058823529411764, 0.20784313725490197, 0.12941176470588237, 0.07450980392156863, 0.00784313725490196, 0.0, 0.0], [0.2602040816326531, 0.22448979591836735, 0.08163265306122448, 0.09693877551020408, 0.14285714285714285, 0.09693877551020408, 0.08673469387755102, 0.01020408163265306, 0.0, 0.0], [0.3238095238095238, 0.14285714285714285, 0.0380952380952381, 0.08571428571428572, 0.23809523809523808, 0.08571428571428572, 0.05714285714285714, 0.02857142857142857, 0.0, 0.0], [0.23333333333333334, 0.14444444444444443, 0.06666666666666667, 0.12222222222222222, 0.17777777777777778, 0.13333333333333333, 0.1111111111111111, 0.011111111111111112, 0.0, 0.0], [0.3, 0.12857142857142856, 0.15714285714285714, 0.08571428571428572, 0.12857142857142856, 0.11428571428571428, 0.04285714285714286, 0.04285714285714286, 0.0, 0.0], [0.3333333333333333, 0.18181818181818182, 0.15151515151515152, 0.06060606060606061, 0.10606060606060606, 0.07575757575757576, 0.06060606060606061, 0.030303030303030304, 0.0, 0.0], [0.5555555555555556, 0.1111111111111111, 0.07407407407407407, 0.05555555555555555, 0.05555555555555555, 0.037037037037037035, 0.1111111111111111, 0.0, 0.0, 0.0], [0.39285714285714285, 0.21428571428571427, 0.14285714285714285, 0.0, 0.07142857142857142, 0.14285714285714285, 0.03571428571428571, 0.0, 0.0, 0.0], [0.28, 0.14, 0.3, 0.12, 0.1, 0.04, 0.02, 0.0, 0.0, 0.0], [0.2558139534883721, 0.3023255813953488, 0.11627906976744186, 0.18604651162790697, 0.06976744186046512, 0.06976744186046512, 0.0, 0.0, 0.0, 0.0], [0.18181818181818182, 0.13636363636363635, 0.13636363636363635, 0.18181818181818182, 0.045454545454545456, 0.18181818181818182, 0.13636363636363635, 0.0, 0.0, 0.0], [0.4, 0.16666666666666666, 0.16666666666666666, 0.1, 0.1, 0.06666666666666667, 0.0, 0.0, 0.0, 0.0], [0.2857142857142857, 0.0, 0.42857142857142855, 0.14285714285714285, 0.0, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0], [0.5277777777777778, 0.2222222222222222, 0.08333333333333333, 0.05555555555555555, 0.08333333333333333, 0.027777777777777776, 0.0, 0.0, 0.0, 0.0], [0.4444444444444444, 0.08888888888888889, 0.13333333333333333, 0.2, 0.08888888888888889, 0.022222222222222223, 0.022222222222222223, 0.0, 0.0, 0.0], [0.4772727272727273, 0.045454545454545456, 0.1590909090909091, 0.045454545454545456, 0.25, 0.022727272727272728, 0.0, 0.0, 0.0, 0.0], [0.6625119846596357, 0.087248322147651, 0.23106423777564716, 0.014381591562799617, 0.004793863854266539, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5338983050847458, 0.1016949152542373, 0.2542372881355932, 0.01694915254237288, 0.05084745762711865, 0.03389830508474576, 0.00847457627118644, 0.0, 0.0, 0.0], [0.5070422535211268, 0.08450704225352113, 0.2676056338028169, 0.056338028169014086, 0.04225352112676056, 0.04225352112676056, 0.0, 0.0, 0.0, 0.0], [0.5425531914893617, 0.10638297872340426, 0.2872340425531915, 0.010638297872340425, 0.031914893617021274, 0.02127659574468085, 0.0, 0.0, 0.0, 0.0], [0.55, 0.125, 0.225, 0.025, 0.075, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5882352941176471, 0.11764705882352941, 0.17647058823529413, 0.08823529411764706, 0.029411764705882353, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6518518518518519, 0.06666666666666667, 0.22962962962962963, 0.014814814814814815, 0.02962962962962963, 0.007407407407407408, 0.0, 0.0, 0.0, 0.0], [0.5942028985507246, 0.08695652173913043, 0.2318840579710145, 0.028985507246376812, 0.043478260869565216, 0.014492753623188406, 0.0, 0.0, 0.0, 0.0], [0.42857142857142855, 0.10714285714285714, 0.10714285714285714, 0.14285714285714285, 0.17857142857142858, 0.03571428571428571, 0.0, 0.0, 0.0, 0.0], [0.59375, 0.109375, 0.171875, 0.078125, 0.03125, 0.015625, 0.0, 0.0, 0.0, 0.0], [0.6610169491525424, 0.05084745762711865, 0.22033898305084745, 0.0, 0.05084745762711865, 0.01694915254237288, 0.0, 0.0, 0.0, 0.0], [0.6521739130434783, 0.10869565217391304, 0.17391304347826086, 0.0, 0.06521739130434782, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5882352941176471, 0.058823529411764705, 0.17647058823529413, 0.029411764705882353, 0.14705882352941177, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5735294117647058, 0.16176470588235295, 0.20588235294117646, 0.04411764705882353, 0.0, 0.014705882352941176, 0.0, 0.0, 0.0, 0.0], [0.375, 0.03125, 0.25, 0.0, 0.125, 0.21875, 0.0, 0.0, 0.0, 0.0], [0.47619047619047616, 0.09523809523809523, 0.30952380952380953, 0.023809523809523808, 0.047619047619047616, 0.047619047619047616, 0.0, 0.0, 0.0, 0.0], [0.5208333333333334, 0.10416666666666667, 0.16666666666666666, 0.020833333333333332, 0.1875, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5333333333333333, 0.06666666666666667, 0.3333333333333333, 0.06666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5641025641025641, 0.07692307692307693, 0.28205128205128205, 0.05128205128205128, 0.0, 0.02564102564102564, 0.0, 0.0, 0.0, 0.0], [0.6153846153846154, 0.07692307692307693, 0.15384615384615385, 0.07692307692307693, 0.07692307692307693, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3548387096774194, 0.45161290322580644, 0.06451612903225806, 0.0967741935483871, 0.03225806451612903, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.08333333333333333, 0.08333333333333333, 0.25, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36363636363636365, 0.18181818181818182, 0.0, 0.45454545454545453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6111111111111112, 0.05555555555555555, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.4444444444444444, 0.2222222222222222, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4375, 0.1875, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2777777777777778, 0.2222222222222222, 0.16666666666666666, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36363636363636365, 0.45454545454545453, 0.09090909090909091, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6923076923076923, 0.07692307692307693, 0.15384615384615385, 0.07692307692307693, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.10526315789473684, 0.8315789473684211, 0.06315789473684211, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42857142857142855, 0.5714285714285714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.40625, 0.09375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8181818181818182, 0.18181818181818182, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.48148148148148145, 0.4444444444444444, 0.07407407407407407, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6923076923076923, 0.15384615384615385, 0.15384615384615385, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8333333333333334, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.375, 0.5, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8125, 0.1875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8571428571428571, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8571428571428571, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
1000: ([0.6234847327411223, 0.07329157299692254, 0.03143374729903747, 0.02487286378418491, 0.017098456904642383, 0.011964991160486282, 0.009468101359757295, 0.005190213239627213, 0.007722024576030731, 0.005085448632603619, 0.004574721173363599, 0.008791496606063251, 0.006259685269659733, 0.006041425671693913, 0.003802082196564594, 0.0039024816116288714, 0.0034354060719820156, 0.0018770325425060566, 0.004063993714123579, 0.0028635659253115654, 0.0025841936399153154, 0.006635091778160945, 0.0036056485583953555, 0.005342994958203287, 0.003518344719209027, 0.0022655346268852174, 0.0027544361263286553, 0.009970098435078683, 0.007791867647379794, 0.007839884758932275, 0.003915577187506821, 0.008023222821223563, 0.0026060195997118975, 0.0026671322871423273, 0.005556889364209791, 0.003924307571425453, 0.0024357771132985572, 0.0031691293624637146, 0.0037016827815003167, 0.00193378003797717, 0.0031298426348298665, 0.002117118100268459, 0.002252439051007268, 0.004365191959316411, 0.002125848484187092, 0.0030512691795621712, 0.0011829670209747474, 0.0023441080821529126, 0.0011218543335443176, 0.0010651068380732043, 0.0028853918851081475, 0.0026365759434271124, 0.0017897287033197284, 0.00159329506515049, 0.003003252068009691, 0.0032258768579348277, 0.0009385162712530283, 0.0008250212803108017, 0.0007289870572058406, 0.0007770041687583211, 0.0006329528341008796, 0.0006504136019381452, 0.0007377174411244735, 0.0009297858873343956, 0.001444878538533732, 0.0005019970753213873, 0.0010651068380732043, 0.0007246218652465242, 0.0008992295436191807, 0.0007289870572058406, 0.0006766047536940437, 0.0005543793788331841, 0.0004758059235654888, 0.00038413689241984417, 0.00031429382107078156, 0.0003361197808673636, 0.00043215400397232466, 0.00029683305323351594, 0.0003972324682977934, 0.00017460767837265643, 0.00022262478992513695, 0.0001178601829015431, 0.0001222253748608595, 0.00010912979898291027, 9.166903114564464e-05, 0.00010912979898291027, 1.7460767837265645e-05, 6.984307134906258e-05, 4.801711155248052e-05, 4.801711155248052e-05, 3.492153567453129e-05, 1.7460767837265645e-05, 4.365191959316411e-06, 4.365191959316411e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[[0.13729512500787644, 0.08029069319685503, 0.09399920185393927, 0.11058523709838901, 0.13847134025526672, 0.15224986172469562, 0.13983659009598756, 0.09546246963194265, 0.049793112139521534, 0.002016368995526181], [0.13567599761762955, 0.08880285884455033, 0.09440142942227517, 0.11048243001786778, 0.13936867182846932, 0.14228707564026205, 0.15008933889219775, 0.09696247766527695, 0.04038117927337701, 0.0015485407980941036], [0.14747951673378698, 0.09609776419941675, 0.09734759061241494, 0.11817803082905153, 0.14164699347312873, 0.14359116789334814, 0.13567560061102624, 0.08123871684488265, 0.0383280099986113, 0.00041660880433273156], [0.19305019305019305, 0.13566163566163567, 0.09740259740259741, 0.08862758862758863, 0.1310986310986311, 0.14566514566514566, 0.1261846261846262, 0.05738855738855739, 0.024745524745524744, 0.0001755001755001755], [0.15011488383967322, 0.11360735256573909, 0.11769211130967577, 0.09599183048251213, 0.1166709216236916, 0.18049527699770232, 0.13224406433495023, 0.07786571355629308, 0.015317845289762573, 0.0], [0.12075884713608172, 0.08391098139365195, 0.11163808828894564, 0.09485589201021526, 0.13863553447646845, 0.11856986501276906, 0.08901860634804816, 0.2032105071141919, 0.039401678219627874, 0.0], [0.08713692946058091, 0.05532503457814661, 0.07238358690640849, 0.09727985246657446, 0.11341632088520055, 0.16551406177962194, 0.25311203319502074, 0.12632549562010142, 0.02950668510834486, 0.0], [0.1118587047939445, 0.13036164844407064, 0.10933557611438183, 0.12867956265769553, 0.13877207737594618, 0.1757779646761985, 0.13204373423044574, 0.061396131202691336, 0.011774600504625737, 0.0], [0.20576596947427925, 0.10740531373657433, 0.0746184284906727, 0.08648954211418881, 0.15036743923120408, 0.1317128321085359, 0.18032786885245902, 0.058790276992651214, 0.0045223289994347085, 0.0], [0.15107296137339055, 0.07982832618025751, 0.09871244635193133, 0.14678111587982834, 0.15965665236051502, 0.13218884120171673, 0.13390557939914163, 0.08583690987124463, 0.01201716738197425, 0.0], [0.15362595419847327, 0.21278625954198474, 0.12595419847328243, 0.13263358778625955, 0.11736641221374046, 0.13740458015267176, 0.07538167938931298, 0.04198473282442748, 0.0028625954198473282, 0.0], [0.1519364448857994, 0.2224428997020854, 0.14895729890764647, 0.12462760675273088, 0.12859980139026814, 0.105759682224429, 0.06752730883813307, 0.04865938430983118, 0.0014895729890764648, 0.0], [0.18828451882845187, 0.14086471408647142, 0.08786610878661087, 0.14435146443514643, 0.15550906555090654, 0.16108786610878661, 0.09693165969316597, 0.024407252440725245, 0.000697350069735007, 0.0], [0.22760115606936415, 0.08598265895953758, 0.15173410404624277, 0.13800578034682082, 0.14234104046242774, 0.15534682080924855, 0.07803468208092486, 0.020953757225433526, 0.0, 0.0], [0.2365097588978186, 0.10677382319173363, 0.11710677382319173, 0.13088404133180254, 0.18828932261768083, 0.10907003444316878, 0.09299655568312284, 0.018369690011481057, 0.0, 0.0], [0.19798657718120805, 0.1174496644295302, 0.19686800894854586, 0.17785234899328858, 0.10961968680089486, 0.09731543624161074, 0.08612975391498881, 0.016778523489932886, 0.0, 0.0], [0.2337992376111817, 0.14104193138500634, 0.21855146124523506, 0.1473951715374841, 0.10038119440914867, 0.08386277001270648, 0.054637865311308764, 0.020330368487928845, 0.0, 0.0], [0.21395348837209302, 0.16046511627906976, 0.16279069767441862, 0.1325581395348837, 0.09767441860465116, 0.14418604651162792, 0.06511627906976744, 0.023255813953488372, 0.0, 0.0], [0.28356605800214824, 0.09989258861439312, 0.10741138560687433, 0.17722878625134264, 0.0966702470461869, 0.12352309344790548, 0.1041890440386681, 0.007518796992481203, 0.0, 0.0], [0.20121951219512196, 0.07469512195121951, 0.125, 0.21341463414634146, 0.12347560975609756, 0.15853658536585366, 0.09298780487804878, 0.010670731707317074, 0.0, 0.0], [0.1858108108108108, 0.11993243243243243, 0.16554054054054054, 0.15371621621621623, 0.21621621621621623, 0.07432432432432433, 0.07939189189189189, 0.005067567567567568, 0.0, 0.0], [0.49407894736842106, 0.08552631578947369, 0.18881578947368421, 0.07236842105263158, 0.05131578947368421, 0.07368421052631578, 0.03355263157894737, 0.0006578947368421052, 0.0, 0.0], [0.22639225181598063, 0.09927360774818401, 0.1513317191283293, 0.14527845036319612, 0.2106537530266344, 0.11138014527845036, 0.05569007263922518, 0.0, 0.0, 0.0], [0.2908496732026144, 0.06290849673202614, 0.08251633986928104, 0.37745098039215685, 0.09477124183006536, 0.061274509803921566, 0.029411764705882353, 0.0008169934640522876, 0.0, 0.0], [0.3535980148883375, 0.1141439205955335, 0.11538461538461539, 0.2518610421836228, 0.07568238213399504, 0.06823821339950373, 0.02109181141439206, 0.0, 0.0, 0.0], [0.24084778420038536, 0.14836223506743737, 0.18497109826589594, 0.18882466281310212, 0.13872832369942195, 0.07707129094412331, 0.02119460500963391, 0.0, 0.0, 0.0], [0.18066561014263074, 0.2329635499207607, 0.23771790808240886, 0.1109350237717908, 0.11251980982567353, 0.10776545166402536, 0.017432646592709985, 0.0, 0.0, 0.0], [0.2535026269702277, 0.11777583187390543, 0.28809106830122594, 0.09588441330998249, 0.16637478108581435, 0.07705779334500876, 0.0013134851138353765, 0.0, 0.0, 0.0], [0.30868347338935576, 0.1619047619047619, 0.16806722689075632, 0.15182072829131651, 0.12549019607843137, 0.08179271708683473, 0.002240896358543417, 0.0, 0.0, 0.0], [0.3730512249443207, 0.22884187082405344, 0.1364142538975501, 0.1319599109131403, 0.08630289532293986, 0.04287305122494432, 0.0005567928730512249, 0.0, 0.0, 0.0], [0.3010033444816054, 0.2129319955406912, 0.1750278706800446, 0.14492753623188406, 0.11259754738015608, 0.05016722408026756, 0.0033444816053511705, 0.0, 0.0, 0.0], [0.2823721436343852, 0.13601741022850924, 0.3400435255712731, 0.176278563656148, 0.04461371055495103, 0.02013057671381937, 0.000544069640914037, 0.0, 0.0, 0.0], [0.25963149078726966, 0.19262981574539365, 0.21608040201005024, 0.17252931323283083, 0.11892797319932999, 0.038525963149078725, 0.0016750418760469012, 0.0, 0.0, 0.0], [0.2078559738134206, 0.1718494271685761, 0.23076923076923078, 0.14238952536824878, 0.2176759410801964, 0.029459901800327332, 0.0, 0.0, 0.0, 0.0], [0.19010212097407697, 0.211311861743912, 0.23723487824037706, 0.17831893165750196, 0.15710919088766692, 0.025923016496465043, 0.0, 0.0, 0.0, 0.0], [0.23136818687430477, 0.20133481646273638, 0.12458286985539488, 0.22246941045606228, 0.19243604004449388, 0.027808676307007785, 0.0, 0.0, 0.0, 0.0], [0.35842293906810035, 0.14516129032258066, 0.2222222222222222, 0.13978494623655913, 0.10931899641577061, 0.025089605734767026, 0.0, 0.0, 0.0, 0.0], [0.3608815426997245, 0.23553719008264462, 0.21212121212121213, 0.09917355371900827, 0.07851239669421488, 0.013774104683195593, 0.0, 0.0, 0.0, 0.0], [0.5908018867924528, 0.12735849056603774, 0.1544811320754717, 0.08136792452830188, 0.04481132075471698, 0.0011792452830188679, 0.0, 0.0, 0.0, 0.0], [0.43115124153498874, 0.17381489841986456, 0.17832957110609482, 0.11286681715575621, 0.09255079006772009, 0.011286681715575621, 0.0, 0.0, 0.0, 0.0], [0.2412831241283124, 0.15481171548117154, 0.15481171548117154, 0.35564853556485354, 0.08647140864714087, 0.00697350069735007, 0.0, 0.0, 0.0, 0.0], [0.4865979381443299, 0.177319587628866, 0.13195876288659794, 0.12577319587628866, 0.07628865979381444, 0.002061855670103093, 0.0, 0.0, 0.0, 0.0], [0.3701550387596899, 0.20155038759689922, 0.2248062015503876, 0.1298449612403101, 0.07364341085271318, 0.0, 0.0, 0.0, 0.0, 0.0], [0.275, 0.405, 0.126, 0.151, 0.043, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2874743326488706, 0.31211498973305957, 0.19096509240246407, 0.17248459958932238, 0.03696098562628337, 0.0, 0.0, 0.0, 0.0, 0.0], [0.13161659513590845, 0.1688125894134478, 0.35765379113018597, 0.2989985693848355, 0.04291845493562232, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2177121771217712, 0.14391143911439114, 0.28044280442804426, 0.2730627306273063, 0.08487084870848709, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2905027932960894, 0.1601489757914339, 0.4376163873370577, 0.09869646182495345, 0.01303538175046555, 0.0, 0.0, 0.0, 0.0, 0.0], [0.22178988326848248, 0.38910505836575876, 0.22568093385214008, 0.14007782101167315, 0.023346303501945526, 0.0, 0.0, 0.0, 0.0, 0.0], [0.35655737704918034, 0.3155737704918033, 0.19672131147540983, 0.11475409836065574, 0.01639344262295082, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5264750378214826, 0.1573373676248109, 0.24508320726172467, 0.0680786686838124, 0.0030257186081694403, 0.0, 0.0, 0.0, 0.0, 0.0], [0.640728476821192, 0.20033112582781457, 0.11589403973509933, 0.041390728476821195, 0.0016556291390728477, 0.0, 0.0, 0.0, 0.0, 0.0], [0.48048780487804876, 0.21707317073170732, 0.2121951219512195, 0.09024390243902439, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4191780821917808, 0.40273972602739727, 0.13424657534246576, 0.043835616438356165, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6875, 0.16569767441860464, 0.11046511627906977, 0.03488372093023256, 0.0014534883720930232, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4871447902571042, 0.2963464140730717, 0.16373477672530445, 0.05277401894451962, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4744186046511628, 0.2744186046511628, 0.20465116279069767, 0.046511627906976744, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42857142857142855, 0.36507936507936506, 0.17989417989417988, 0.026455026455026454, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.437125748502994, 0.3532934131736527, 0.19161676646706588, 0.017964071856287425, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4101123595505618, 0.4157303370786517, 0.16292134831460675, 0.011235955056179775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.496551724137931, 0.3724137931034483, 0.12413793103448276, 0.006896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4161073825503356, 0.44966442953020136, 0.1342281879194631, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5621301775147929, 0.3076923076923077, 0.1242603550295858, 0.005917159763313609, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5211267605633803, 0.41784037558685444, 0.06103286384976526, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7824773413897281, 0.18429003021148035, 0.03323262839879154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5739130434782609, 0.3217391304347826, 0.10434782608695652, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.319672131147541, 0.6229508196721312, 0.05737704918032787, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.42771084337349397, 0.07228915662650602, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4854368932038835, 0.5048543689320388, 0.009708737864077669, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6706586826347305, 0.2215568862275449, 0.10778443113772455, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6193548387096774, 0.38064516129032255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6299212598425197, 0.3543307086614173, 0.015748031496062992, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7247706422018348, 0.27522935779816515, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7045454545454546, 0.29545454545454547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7916666666666666, 0.20833333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8701298701298701, 0.12987012987012986, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9595959595959596, 0.04040404040404041, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7352941176470589, 0.2647058823529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9560439560439561, 0.04395604395604396, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.975, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9803921568627451, 0.0196078431372549, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9629629629629629, 0.037037037037037035, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
1500: ([0.5198436028573956, 0.05666664932389193, 0.026157240001456795, 0.01892530293491777, 0.013438916145949855, 0.010062277904090988, 0.008064390254054307, 0.005509799534866781, 0.007257951228648876, 0.009013907171063927, 0.004682549179773469, 0.0075779254226000636, 0.005515002367288752, 0.005991061533899055, 0.00409983194851277, 0.004656535017663616, 0.004326155158868488, 0.0039489498082756255, 0.004583695363756028, 0.004953096465715936, 0.006482729197775269, 0.006500939111252165, 0.004287133915703709, 0.00714088749915454, 0.0038110747490934065, 0.003426065149867588, 0.007767828806001987, 0.010470700249215674, 0.009052928414228707, 0.007372413541932228, 0.005314693319042887, 0.010215761460539118, 0.006506141943674136, 0.004313148077813561, 0.005647674594049, 0.00439639339656509, 0.005751731242488411, 0.003566541625260792, 0.008332336123785788, 0.005587842021196339, 0.006006670031164966, 0.009581015905058714, 0.004893263892863275, 0.007401029120253066, 0.00389692148405592, 0.0046487307690306606, 0.002905781907670536, 0.0039437469758536545, 0.0026144232920401866, 0.002944803150835315, 0.006230391825309698, 0.0040868248674578445, 0.0033142042527952217, 0.0032491688475205904, 0.0037512421762407456, 0.00549939387002284, 0.005811563815341072, 0.0034520793119774407, 0.002317861843987867, 0.0020134961473025916, 0.0023933029141064393, 0.01193789899221136, 0.004802214325478791, 0.002674255864892848, 0.005023334703412537, 0.004651332185241645, 0.0025545907191875258, 0.0047475845850481, 0.0024713454004359975, 0.0016440950453426846, 0.0018522083422215054, 0.0013241208513914975, 0.0011134061383016914, 0.0011706372949433672, 0.0010015452412293253, 0.00229965193051097, 0.0013267222676024827, 0.000928705587321738, 0.0013059109379146008, 0.0006477526365353298, 0.0004214294261796122, 0.00037460393438187747, 0.00049947191250917, 0.0006373469716913888, 0.0006321441392694182, 0.0007934319443505044, 0.00019510621582389452, 0.00023933029141064396, 0.00017949771855798298, 0.00016128780508108614, 7.804248632955781e-05, 8.064390254054307e-05, 3.9021243164778906e-05, 7.804248632955782e-06, 2.60141621098526e-06, 2.60141621098526e-06, 2.60141621098526e-06, 0.0, 0.0, 0.0] ,
[[0.10712051683672703, 0.06607583407979743, 0.0806431434562205, 0.10229644049221592, 0.13475887124620303, 0.15899935445451407, 0.1552561914818021, 0.1239347248424919, 0.0642542948791729, 0.006660628230855073], [0.11210577055501997, 0.07551760547215719, 0.0829545976219988, 0.10811183032640133, 0.13882385346371023, 0.14869393563788275, 0.1569572602488179, 0.11724739475737961, 0.05563971904696323, 0.003948032869669008], [0.12103431128791646, 0.08741919443063152, 0.08960716061660866, 0.11745400298359025, 0.13615116857284934, 0.14997513674788662, 0.15087021382396817, 0.09646941819990054, 0.04803580308304326, 0.0029835902536051715], [0.16082474226804125, 0.11532646048109965, 0.09786941580756013, 0.09305841924398625, 0.13567010309278352, 0.14900343642611683, 0.13731958762886598, 0.07353951890034365, 0.036151202749140895, 0.0012371134020618556], [0.12524196670538135, 0.09523809523809523, 0.10336817653890824, 0.09659310878823074, 0.13879210220673635, 0.17363530778164923, 0.14305071622144794, 0.10046457607433217, 0.023035230352303523, 0.0005807200929152149], [0.10005170630816959, 0.07109617373319545, 0.10108583247156153, 0.10496380558428128, 0.140382626680455, 0.1266804550155119, 0.1266804550155119, 0.18976215098241986, 0.03929679420889348, 0.0], [0.08451612903225807, 0.0703225806451613, 0.07451612903225806, 0.11709677419354839, 0.11806451612903225, 0.15451612903225806, 0.23709677419354838, 0.11419354838709678, 0.02967741935483871, 0.0], [0.12370160528800755, 0.10717658168083097, 0.09301227573182247, 0.16100094428706327, 0.14069877242681775, 0.1506137865911237, 0.13786591123701605, 0.07034938621340887, 0.015580736543909348, 0.0], [0.14086021505376345, 0.08172043010752689, 0.07741935483870968, 0.11039426523297491, 0.17204301075268819, 0.14265232974910394, 0.1860215053763441, 0.07455197132616488, 0.014336917562724014, 0.0], [0.0911976911976912, 0.050216450216450215, 0.09350649350649351, 0.10505050505050505, 0.14256854256854257, 0.1974025974025974, 0.1950937950937951, 0.09610389610389611, 0.02886002886002886, 0.0], [0.12222222222222222, 0.145, 0.12, 0.12555555555555556, 0.135, 0.1527777777777778, 0.12444444444444444, 0.07, 0.005, 0.0], [0.12530037761757637, 0.1774802608994164, 0.13285272914521112, 0.13868863714383797, 0.1414349467902506, 0.11362856162032269, 0.0971507037418469, 0.06865774116031582, 0.004806041881222108, 0.0], [0.15047169811320754, 0.11273584905660378, 0.09198113207547169, 0.14622641509433962, 0.18962264150943398, 0.15235849056603773, 0.10849056603773585, 0.04716981132075472, 0.0009433962264150943, 0.0], [0.15501519756838905, 0.07381676074685194, 0.11463308727746417, 0.12983065566652194, 0.233608336951802, 0.1519756838905775, 0.09900130264871906, 0.03907946157186279, 0.00303951367781155, 0.0], [0.17258883248730963, 0.0983502538071066, 0.13007614213197968, 0.13769035532994925, 0.1782994923857868, 0.1313451776649746, 0.10723350253807107, 0.04378172588832487, 0.0006345177664974619, 0.0], [0.1324022346368715, 0.0893854748603352, 0.16145251396648044, 0.1893854748603352, 0.14860335195530727, 0.12346368715083798, 0.09217877094972067, 0.06312849162011173, 0.0, 0.0], [0.14792543595911004, 0.1358989777510523, 0.179795550210463, 0.13950691521346964, 0.13650030066145522, 0.13048707155742634, 0.09140108238123873, 0.03848466626578473, 0.0, 0.0], [0.10935441370223979, 0.11725955204216074, 0.12714097496706192, 0.16798418972332016, 0.16534914361001318, 0.15151515151515152, 0.12318840579710146, 0.03820816864295125, 0.0, 0.0], [0.17593643586833144, 0.08002270147559591, 0.12031782065834279, 0.1782065834279228, 0.1282633371169126, 0.1492622020431328, 0.14869466515323496, 0.019296254256526674, 0.0, 0.0], [0.15283613445378152, 0.10399159663865547, 0.12920168067226892, 0.19117647058823528, 0.11292016806722689, 0.148109243697479, 0.12237394957983193, 0.03939075630252101, 0.0, 0.0], [0.10553772070626004, 0.08667736757624397, 0.13202247191011235, 0.11998394863563402, 0.09831460674157304, 0.23715890850722313, 0.17897271268057785, 0.0413322632423756, 0.0, 0.0], [0.33733493397358943, 0.08563425370148059, 0.139655862344938, 0.11844737895158063, 0.09523809523809523, 0.14245698279311725, 0.0740296118447379, 0.007202881152460984, 0.0, 0.0], [0.1474514563106796, 0.07584951456310679, 0.13531553398058252, 0.16990291262135923, 0.21055825242718446, 0.1717233009708738, 0.08434466019417476, 0.0048543689320388345, 0.0, 0.0], [0.21420765027322405, 0.06156648451730419, 0.11329690346083789, 0.2859744990892532, 0.19052823315118397, 0.09398907103825137, 0.03934426229508197, 0.001092896174863388, 0.0, 0.0], [0.24232081911262798, 0.0962457337883959, 0.12081911262798635, 0.20204778156996586, 0.14948805460750852, 0.13447098976109215, 0.05392491467576792, 0.0006825938566552901, 0.0, 0.0], [0.1480637813211845, 0.10250569476082004, 0.19893697798025817, 0.1404707668944571, 0.2201974183750949, 0.12072892938496584, 0.06909643128321943, 0.0, 0.0, 0.0], [0.16778298727394508, 0.09176155391828533, 0.13529805760214333, 0.2173476222371065, 0.2588747488278634, 0.08305425318151373, 0.045880776959142666, 0.0, 0.0, 0.0], [0.186583850931677, 0.11652173913043479, 0.24397515527950311, 0.1560248447204969, 0.19850931677018632, 0.08198757763975155, 0.01639751552795031, 0.0, 0.0, 0.0], [0.18160919540229886, 0.14022988505747128, 0.15344827586206897, 0.13994252873563218, 0.24741379310344827, 0.11954022988505747, 0.017816091954022988, 0.0, 0.0, 0.0], [0.26852505292872264, 0.171136203246295, 0.1337332392378264, 0.16584333098094567, 0.16337332392378265, 0.08256880733944955, 0.014820042342978124, 0.0, 0.0, 0.0], [0.2178169358786099, 0.16250611845325502, 0.1595692608908468, 0.15810083210964268, 0.1977484092021537, 0.07929515418502203, 0.024963289280469897, 0.0, 0.0, 0.0], [0.25286478227654696, 0.15762668703845176, 0.24293353705118412, 0.15508021390374332, 0.14616755793226383, 0.04201680672268908, 0.0033104150751209573, 0.0, 0.0, 0.0], [0.2602958816473411, 0.2459016393442623, 0.17273090763694524, 0.15153938424630148, 0.11395441823270691, 0.0547780887644942, 0.0007996801279488205, 0.0, 0.0, 0.0], [0.18636911942098913, 0.16706875753920386, 0.17852834740651388, 0.19300361881785283, 0.2080820265379976, 0.06393244873341375, 0.0030156815440289505, 0.0, 0.0, 0.0], [0.17779824965453708, 0.18102257024412713, 0.2160294795025334, 0.19760479041916168, 0.17227084292952557, 0.05481345002303086, 0.00046061722708429296, 0.0, 0.0, 0.0], [0.19585798816568048, 0.14142011834319526, 0.13254437869822486, 0.24733727810650888, 0.21242603550295858, 0.06923076923076923, 0.001183431952662722, 0.0, 0.0, 0.0], [0.23111714156490276, 0.14382632293080055, 0.2510176390773406, 0.2094075079149706, 0.12121212121212122, 0.04341926729986431, 0.0, 0.0, 0.0, 0.0], [0.2698760029175784, 0.19839533187454414, 0.20058351568198396, 0.16630196936542668, 0.12545587162654998, 0.03938730853391685, 0.0, 0.0, 0.0, 0.0], [0.2644395878863565, 0.16172338432719324, 0.12394630034342803, 0.10927255697783328, 0.2769278801123946, 0.06369029035279425, 0.0, 0.0, 0.0, 0.0], [0.1908752327746741, 0.26955307262569833, 0.13500931098696461, 0.1941340782122905, 0.15782122905027932, 0.05260707635009311, 0.0, 0.0, 0.0, 0.0], [0.1346903421394543, 0.18319618882633174, 0.1779991338241663, 0.25249025552187093, 0.23040277176266782, 0.02122130792550888, 0.0, 0.0, 0.0, 0.0], [0.15557969046972578, 0.1629106706489275, 0.1968503937007874, 0.2139560141189248, 0.2636437686668477, 0.0070594623947868584, 0.0, 0.0, 0.0, 0.0], [0.1951089845826688, 0.17543859649122806, 0.18341307814992025, 0.2222222222222222, 0.19883040935672514, 0.024986709197235512, 0.0, 0.0, 0.0, 0.0], [0.17082601054481547, 0.3202108963093146, 0.2014059753954306, 0.19402460456942003, 0.11142355008787347, 0.00210896309314587, 0.0, 0.0, 0.0, 0.0], [0.29906542056074764, 0.21895861148197596, 0.2136181575433912, 0.1882510013351135, 0.0774365821094793, 0.0026702269692923898, 0.0, 0.0, 0.0, 0.0], [0.16284275321768327, 0.19306099608282037, 0.2630106323447118, 0.23503077783995524, 0.1454952434247342, 0.0005595970900951316, 0.0, 0.0, 0.0, 0.0], [0.24529991047448524, 0.21128021486123547, 0.18710832587287377, 0.2748433303491495, 0.08146821844225605, 0.0, 0.0, 0.0, 0.0, 0.0], [0.26451187335092347, 0.2579155672823219, 0.2836411609498681, 0.14775725593667546, 0.04617414248021108, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19502487562189055, 0.263681592039801, 0.26965174129353237, 0.208955223880597, 0.0626865671641791, 0.0, 0.0, 0.0, 0.0, 0.0], [0.31537102473498235, 0.24646643109540636, 0.25265017667844525, 0.1687279151943463, 0.01678445229681979, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2647181628392484, 0.31816283924843425, 0.21336116910229644, 0.19039665970772443, 0.013361169102296452, 0.0, 0.0, 0.0, 0.0, 0.0], [0.40420114576702737, 0.24124761298535966, 0.2005092297899427, 0.14385741565881605, 0.010184595798854232, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4050235478806907, 0.21350078492935637, 0.17346938775510204, 0.206436420722135, 0.0015698587127158557, 0.0, 0.0, 0.0, 0.0, 0.0], [0.32826261008807045, 0.23698959167333866, 0.18574859887910328, 0.24499599679743794, 0.0040032025620496394, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5346740638002774, 0.22191400832177532, 0.12968099861303745, 0.11026352288488211, 0.0034674063800277394, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45931882686849573, 0.28098391674550616, 0.1773888363292337, 0.08230842005676443, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5608773500447628, 0.24082363473589974, 0.14458370635631154, 0.05371530886302596, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4423511680482291, 0.3006782215523738, 0.1680482290881688, 0.08892238131122833, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3984287317620651, 0.33557800224466894, 0.1919191919191919, 0.07407407407407407, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36563307493540054, 0.3449612403100775, 0.2596899224806202, 0.029715762273901807, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.29891304347826086, 0.39891304347826084, 0.2565217391304348, 0.04565217391304348, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16670298539986925, 0.22924384397472217, 0.5595990411854435, 0.044454129439965136, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5184182015167931, 0.2600216684723727, 0.20910075839653305, 0.012459371614301192, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.504863813229572, 0.2830739299610895, 0.20817120622568094, 0.0038910505836575876, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.33816675297773174, 0.27550491973070945, 0.37907819782496116, 0.0072501294665976174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5357941834451901, 0.29138702460850113, 0.17281879194630873, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5091649694501018, 0.38289205702647655, 0.1079429735234216, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6778082191780822, 0.2778082191780822, 0.04438356164383562, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6568421052631579, 0.3063157894736842, 0.03684210526315789, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6598101265822784, 0.254746835443038, 0.08544303797468354, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5814606741573034, 0.38202247191011235, 0.03651685393258427, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5776031434184676, 0.381139489194499, 0.0412573673870334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6705607476635514, 0.3177570093457944, 0.011682242990654205, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7022222222222222, 0.2822222222222222, 0.015555555555555555, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5766233766233766, 0.4051948051948052, 0.01818181818181818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8959276018099548, 0.10294117647058823, 0.0011312217194570137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.807843137254902, 0.19215686274509805, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8123249299719888, 0.1876750700280112, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8705179282868526, 0.1294820717131474, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8755020080321285, 0.12449799196787148, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9135802469135802, 0.08641975308641975, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9305555555555556, 0.06944444444444445, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.953125, 0.046875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9959183673469387, 0.004081632653061225, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9917695473251029, 0.00823045267489712, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9967213114754099, 0.003278688524590164, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
2000: ([0.4772083011499751, 0.04734776454720466, 0.022236903496975092, 0.016147877466682874, 0.012834788609247635, 0.008792428121655054, 0.008614031029331618, 0.006161561111786363, 0.006757525024603116, 0.008531693909797725, 0.005220565459970437, 0.007467192578680959, 0.00556559719896961, 0.0060243325792298736, 0.004953950025289258, 0.005263694427345334, 0.005202921791498888, 0.004850048422067916, 0.00556363679136166, 0.005479339264219817, 0.006908476410415253, 0.006479147144274238, 0.006159600704178413, 0.007159408584232834, 0.00492454391117001, 0.005279377688208933, 0.008768903230359656, 0.009749107034334578, 0.009386431626863858, 0.00746327176346506, 0.0077690953503052355, 0.00973342377347098, 0.0074064199428345145, 0.005169594862163741, 0.006298789644342852, 0.005953757905343679, 0.005767519182588444, 0.004069806194103878, 0.008359178040298138, 0.006459543068194739, 0.00669283157354077, 0.009762829887590228, 0.00555579516092986, 0.008670882849962164, 0.0046285223623695835, 0.005199000976282989, 0.0037816262757352508, 0.006026292986837823, 0.004830444345988418, 0.003975706628922286, 0.006398770432348294, 0.005240169536049936, 0.004030598041944881, 0.004218797172308066, 0.004785354971005571, 0.006267423122615654, 0.0062791855682633535, 0.00429133225380221, 0.004626561954761634, 0.005800846111923591, 0.004330540405961207, 0.010268615050441288, 0.0048441671992440665, 0.0038129927974624484, 0.005273496465385083, 0.005338189916447428, 0.0035483377703892194, 0.005967480758599328, 0.004477570976557446, 0.0024544303251532056, 0.0025112821457837515, 0.002083913287250685, 0.0018133770373536067, 0.003550298177997169, 0.0020799924720347855, 0.0032307517379013447, 0.0017388815482515124, 0.0014114934777238884, 0.0017937729612741082, 0.0012174131245368538, 0.000909629130088728, 0.0007625985594924896, 0.000897866684441029, 0.000935114428992076, 0.0010076495104862204, 0.0009919662496226216, 0.0005920430976008532, 0.00046853741830001294, 0.0003215068477037746, 0.0002587738042493795, 0.00011958486408494053, 0.00015291179342008792, 0.000113703641261091, 8.037671192594364e-05, 5.68518206305455e-05, 2.9406114119247675e-05, 1.960407607949845e-06, 0.0, 0.0, 0.0] ,
[[0.08857421032523631, 0.05515912629455721, 0.06845285778254315, 0.08775259527653509, 0.12129502963976288, 0.14853156850420873, 0.16621272435225923, 0.14582845499398167, 0.10200761637150146, 0.01618581645941427], [0.10156508777740973, 0.06889698575687314, 0.07668102020536602, 0.09999171911228884, 0.13174892348459755, 0.1444600861212322, 0.16238820801589932, 0.13332229214971844, 0.07403113613779397, 0.006914541238820801], [0.10808428105439478, 0.07881512827294367, 0.08198889182755885, 0.10773164065943754, 0.13021246583796173, 0.15093008904169972, 0.16671074671603633, 0.11099356431279203, 0.05880278585912016, 0.005730406418055188], [0.14277042612601676, 0.10270729634575695, 0.09008134029379629, 0.08911011290518393, 0.13002306665047955, 0.15163287604710451, 0.15223989316498726, 0.08923151632876047, 0.04795435231273522, 0.00424911982517907], [0.10065678936917673, 0.07667634030853826, 0.08675729341683214, 0.08400794256911563, 0.12967771498396213, 0.16098976630517794, 0.1565602566060791, 0.14541011150145106, 0.05758362608828471, 0.0016801588513823125], [0.08784838350055742, 0.06376811594202898, 0.09119286510590859, 0.09698996655518395, 0.13600891861761427, 0.13110367892976588, 0.15072463768115943, 0.19219620958751393, 0.04972129319955407, 0.0004459308807134894], [0.06053709604005462, 0.05097860719162494, 0.058716431497496585, 0.09968138370505235, 0.11697769685935366, 0.18730086481565772, 0.23236231224396905, 0.15179790623577605, 0.04164770141101502, 0.0], [0.08526885141584474, 0.07349665924276169, 0.07031498568246898, 0.13108495068405981, 0.12090359529112313, 0.16926503340757237, 0.19630925867006044, 0.11963092586700605, 0.03372573973910277, 0.0], [0.11778357992457208, 0.0673049028140412, 0.0675950101537569, 0.10008703220191471, 0.16594139831737742, 0.15752828546562228, 0.2059762111981433, 0.09428488540760081, 0.02349869451697128, 0.0], [0.07421875, 0.04204963235294118, 0.08065257352941177, 0.09650735294117647, 0.13212316176470587, 0.19738051470588236, 0.22633272058823528, 0.11787683823529412, 0.03285845588235294, 0.0], [0.08636875704093128, 0.10063837776943298, 0.09425460007510326, 0.10214044310927525, 0.14194517461509576, 0.172361997746902, 0.17386406308674426, 0.10777318813368382, 0.020653398422831395, 0.0], [0.09713835652402206, 0.1407193489104752, 0.11157784195326857, 0.12260435809923864, 0.152533473352586, 0.1514833289577317, 0.12654239957994223, 0.08611184037805199, 0.011289052244683644, 0.0], [0.11553363860514265, 0.08770693906305037, 0.08030996829869673, 0.1306798168369144, 0.1785840084536809, 0.16731243395561818, 0.14758717858400847, 0.08171891511095457, 0.010567101091933779, 0.0], [0.11812561015294501, 0.06150341685649203, 0.09534656687276277, 0.11454604620891637, 0.21737715587373901, 0.16856492027334852, 0.13472177025707777, 0.08232997071265864, 0.007484542792059876, 0.0], [0.11159477641472101, 0.06925207756232687, 0.10051444400474871, 0.13850415512465375, 0.184804115552038, 0.1669964384645825, 0.14364859517214087, 0.08191531460229522, 0.002770083102493075, 0.0], [0.09608938547486033, 0.06443202979515829, 0.1303538175046555, 0.16238361266294227, 0.15456238361266295, 0.15977653631284916, 0.15083798882681565, 0.08044692737430167, 0.0011173184357541898, 0.0], [0.09683496608892238, 0.09382064807837227, 0.13602110022607386, 0.11567445365486059, 0.15749811605124342, 0.15599095704596835, 0.1680482290881688, 0.07611152976639035, 0.0, 0.0], [0.07154405820533549, 0.08973322554567502, 0.11600646725949879, 0.13217461600646727, 0.17259498787388844, 0.20371867421180276, 0.15804365400161682, 0.05618431689571544, 0.0, 0.0], [0.12403100775193798, 0.06307258632840028, 0.10570824524312897, 0.1522198731501057, 0.16596194503171247, 0.19027484143763213, 0.16807610993657504, 0.0306553911205074, 0.0, 0.0], [0.118783542039356, 0.0851520572450805, 0.11520572450805008, 0.16493738819320214, 0.13595706618962433, 0.17638640429338104, 0.156350626118068, 0.047227191413237925, 0.0, 0.0], [0.08399545970488081, 0.07406356413166856, 0.11889897843359819, 0.11691259931895573, 0.12400681044267878, 0.2437570942111237, 0.19721906923950056, 0.04114642451759364, 0.0, 0.0], [0.2574886535552194, 0.07473524962178517, 0.12012102874432677, 0.11860816944024205, 0.12344931921331316, 0.18517397881996975, 0.10741301059001512, 0.013010590015128594, 0.0, 0.0], [0.08847867600254615, 0.049331635900700194, 0.09707192870782941, 0.14704010184595798, 0.21037555697008276, 0.24220241884150223, 0.1524506683640993, 0.013049013367281986, 0.0, 0.0], [0.17278203723986857, 0.05366922234392114, 0.0988499452354874, 0.24288061336254108, 0.19304490690032858, 0.14074479737130338, 0.09309967141292443, 0.004928806133625411, 0.0, 0.0], [0.1480891719745223, 0.06687898089171974, 0.10270700636942676, 0.15963375796178345, 0.17794585987261147, 0.2089968152866242, 0.1305732484076433, 0.0051751592356687895, 0.0, 0.0], [0.08057927961381359, 0.06832528778314148, 0.1344225770516153, 0.1422205718529521, 0.2350538432974378, 0.20794652803564798, 0.13070924619383587, 0.0007426661715558856, 0.0, 0.0], [0.12117147328414934, 0.07936507936507936, 0.12854907221104403, 0.19517102615694165, 0.29085624860272746, 0.12206572769953052, 0.0628213726805276, 0.0, 0.0, 0.0], [0.15785240297607078, 0.10898853810577117, 0.22159662175749045, 0.15724914538507942, 0.2079227830283531, 0.11421677056102957, 0.03217373818620551, 0.0, 0.0, 0.0], [0.13909774436090225, 0.11131996658312447, 0.13032581453634084, 0.13993316624895571, 0.2627401837928154, 0.1683375104427736, 0.04824561403508772, 0.0, 0.0, 0.0], [0.21040189125295508, 0.1415812976096664, 0.12188074599422118, 0.1667980036774363, 0.1728395061728395, 0.136327817178881, 0.050170738114000524, 0.0, 0.0, 0.0], [0.12137269745142569, 0.10219530658591976, 0.11178400201867272, 0.13676507696189755, 0.1806712086802927, 0.186222558667676, 0.16098914963411556, 0.0, 0.0, 0.0], [0.20926485397784492, 0.13554884189325278, 0.21309164149043303, 0.1540785498489426, 0.15891238670694863, 0.08821752265861027, 0.040886203423967774, 0.0, 0.0, 0.0], [0.1850185283218634, 0.19640021175224986, 0.15034409740603494, 0.16437268395976706, 0.14690312334568556, 0.1069348861831657, 0.05002646903123346, 0.0, 0.0, 0.0], [0.13120970800151688, 0.13765642775881684, 0.1626848691695108, 0.2006067500948047, 0.2328403488813045, 0.11149032992036405, 0.023511566173682216, 0.0, 0.0, 0.0], [0.14565826330532214, 0.15717398070339247, 0.20292561469032058, 0.20510426392779335, 0.1870525988173047, 0.0915032679738562, 0.010582010582010581, 0.0, 0.0, 0.0], [0.13730655251893314, 0.11787948633519921, 0.14027000329272307, 0.26440566348370104, 0.22522225880803423, 0.11261112940401712, 0.0023049061573921633, 0.0, 0.0, 0.0], [0.18728755948334466, 0.12440516655336506, 0.22501699524133242, 0.23113528212100612, 0.1583956492182189, 0.07375934738273283, 0.0, 0.0, 0.0, 0.0], [0.1960500963391137, 0.16907514450867053, 0.20134874759152216, 0.18882466281310212, 0.15558766859344894, 0.08911368015414259, 0.0, 0.0, 0.0, 0.0], [0.21904315196998123, 0.14118198874296436, 0.13602251407129456, 0.14681050656660413, 0.2849437148217636, 0.07199812382739212, 0.0, 0.0, 0.0, 0.0], [0.1526555386949924, 0.21122913505311078, 0.165402124430956, 0.2094081942336874, 0.2006069802731411, 0.06069802731411229, 0.0, 0.0, 0.0, 0.0], [0.10984182776801406, 0.15377855887521968, 0.1751611013473931, 0.2521968365553603, 0.2680140597539543, 0.041007615700058585, 0.0, 0.0, 0.0, 0.0], [0.13112449799196788, 0.14216867469879518, 0.19236947791164657, 0.22791164658634538, 0.2863453815261044, 0.020080321285140562, 0.0, 0.0, 0.0, 0.0], [0.15949188426252647, 0.16125617501764292, 0.20889202540578689, 0.21383203952011293, 0.2268877911079746, 0.029640084685956247, 0.0, 0.0, 0.0, 0.0], [0.13429798779109203, 0.23626497852136558, 0.1962468912502826, 0.24869997739091115, 0.1770291657246213, 0.007460999321727334, 0.0, 0.0, 0.0, 0.0], [0.22744599745870395, 0.18297331639135958, 0.19229140194832697, 0.22193985599322322, 0.16857263871241, 0.006776789495976281, 0.0, 0.0, 0.0, 0.0], [0.14969834087481146, 0.19004524886877827, 0.24509803921568626, 0.23868778280542988, 0.17496229260935142, 0.0015082956259426848, 0.0, 0.0, 0.0, 0.0], [0.1824779678589943, 0.18921721099015035, 0.2089165370658372, 0.2939346811819596, 0.12441679626749612, 0.0010368066355624676, 0.0, 0.0, 0.0, 0.0], [0.1977878985035784, 0.22218607677293428, 0.3109954456733897, 0.18542615484710476, 0.08360442420299284, 0.0, 0.0, 0.0, 0.0, 0.0], [0.14488636363636365, 0.2780032467532468, 0.2780032467532468, 0.21266233766233766, 0.0864448051948052, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2465483234714004, 0.2465483234714004, 0.28303747534516766, 0.1814595660749507, 0.04240631163708087, 0.0, 0.0, 0.0, 0.0, 0.0], [0.22977941176470587, 0.27328431372549017, 0.25796568627450983, 0.21323529411764705, 0.025735294117647058, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3363262252151141, 0.24354657687991021, 0.23718668163112608, 0.16049382716049382, 0.02244668911335578, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3132295719844358, 0.22081712062256809, 0.25, 0.20525291828793774, 0.010700389105058366, 0.0, 0.0, 0.0, 0.0, 0.0], [0.266728624535316, 0.25743494423791824, 0.2342007434944238, 0.22769516728624536, 0.013940520446096654, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4219582138467841, 0.20647275706677592, 0.20606308889799263, 0.15731257681278166, 0.00819336337566571, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36315295589615265, 0.2696277760400375, 0.24585548952142633, 0.11510791366906475, 0.006255864873318737, 0.0, 0.0, 0.0, 0.0, 0.0], [0.46425226350296595, 0.2450827349359975, 0.2169840774274118, 0.07336871682797377, 0.00031220730565095225, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3663773412517131, 0.2987665600730927, 0.2206486980356327, 0.11420740063956145, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25508474576271184, 0.3008474576271186, 0.36610169491525424, 0.07796610169491526, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.30348090571138897, 0.31463332206826633, 0.32342007434944237, 0.05846569787090233, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3091896785875962, 0.34268899954730647, 0.2820280669986419, 0.06609325486645541, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18136693394425354, 0.2367315769377625, 0.5353188239786177, 0.046582665139366174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45366248482395793, 0.2962363415621206, 0.23269931201942534, 0.017401861594496155, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4318766066838046, 0.289974293059126, 0.2652956298200514, 0.012853470437017995, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36654275092936806, 0.2721189591078067, 0.3513011152416357, 0.010037174721189592, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47337495409474845, 0.3257436650752846, 0.19867792875504958, 0.0022034520749173708, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4878453038674033, 0.35027624309392263, 0.16077348066298341, 0.0011049723756906078, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6034822601839684, 0.30223390275952694, 0.09395532194480946, 0.000328515111695138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6335376532399299, 0.27189141856392296, 0.09457092819614711, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6214057507987221, 0.26757188498402557, 0.1110223642172524, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5964090554254489, 0.33489461358313816, 0.06869633099141297, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6020696142991533, 0.34807149576669805, 0.04985888993414864, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6432432432432432, 0.3318918918918919, 0.024864864864864864, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6449475427940364, 0.33959138597459965, 0.015461071231363888, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6880301602262017, 0.29783223374175305, 0.01413760603204524, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7773058252427184, 0.220873786407767, 0.0018203883495145632, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7576099210822999, 0.24239007891770012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6541666666666667, 0.3458333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8316939890710382, 0.16830601092896175, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.821256038647343, 0.178743961352657, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8663793103448276, 0.1336206896551724, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8817480719794345, 0.11825192802056556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9235807860262009, 0.07641921397379912, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.960167714884696, 0.039832285115303984, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9902723735408561, 0.009727626459143969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9901185770750988, 0.009881422924901186, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9966887417218543, 0.0033112582781456954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
3000: ([0.39251050350976174, 0.03653245276642543, 0.017871002294018085, 0.013065306484825399, 0.010864366538839359, 0.008112117627853561, 0.007449114898516191, 0.005879674312438962, 0.006723105430321706, 0.008054838774347093, 0.005527409363374183, 0.006894941990841111, 0.005540297105413138, 0.006439575105464689, 0.0053441170321534845, 0.005292566063997663, 0.005282542264634031, 0.0054515148824781125, 0.00598707216276359, 0.005553184847452094, 0.007058186723334546, 0.006522629443049068, 0.006366544567243942, 0.007255798767931861, 0.005435763197763834, 0.0053713244875690565, 0.008558892685204014, 0.00900996365656745, 0.009551248822203574, 0.008314025586463861, 0.008216651535502866, 0.010284418147086367, 0.008050542860334107, 0.006166068579971303, 0.0071741764016851435, 0.006880622277464494, 0.006766064570451558, 0.005556048790127417, 0.008335505156528786, 0.008222379420853512, 0.008122141427217193, 0.009745996924125568, 0.007712597624645945, 0.008727865303048095, 0.006715945573633398, 0.007215703570477333, 0.00525676678055612, 0.008746480930437697, 0.0069063977615424045, 0.0056906540958676175, 0.00782572336032122, 0.006867734535425539, 0.00637370442393225, 0.005972752449386973, 0.006009983704166178, 0.0075994718889706705, 0.007062482637347531, 0.006675850376178871, 0.006562724640503595, 0.007895889955866644, 0.006627163350698372, 0.01274597687652684, 0.007308781707425344, 0.005610463700958562, 0.007046730952633252, 0.0068004318825554385, 0.006241963060867373, 0.007953168809373112, 0.006262010659594638, 0.005258198751893782, 0.004661066704088851, 0.006420959478075087, 0.0042099957327254136, 0.005229559325140548, 0.006430983277438719, 0.004639587134023925, 0.0032262314237518222, 0.004283026270946161, 0.004462022688153874, 0.003347948987453067, 0.0023942560765703716, 0.003115969630751871, 0.0027049938568429613, 0.0023656166498171372, 0.0027436570829598274, 0.0022052358599990263, 0.0015050018758824522, 0.0027536808823234596, 0.0011770804395579218, 0.000986628251648915, 0.0007460570669217485, 0.0005369892516231395, 0.00035226494906477954, 0.00025918681211676864, 0.00013317333440253859, 7.73264522337321e-05, 1.1455770701293643e-05, 5.727885350646822e-06, 0.0, 0.0] ,
[[0.07870706481093012, 0.049083380456394446, 0.06103500483391401, 0.07866328596705642, 0.1096331697707083, 0.13744732857846445, 0.16138705970339834, 0.1641597198153992, 0.12934459422484085, 0.030539391838893856], [0.0961508309814989, 0.06522420821574161, 0.0726716839134525, 0.0950533082470994, 0.1260975227343995, 0.14016933207902163, 0.16098306679209784, 0.14173722169959235, 0.0905456255879586, 0.011367199749137661], [0.09823717948717949, 0.07179487179487179, 0.07483974358974359, 0.09895833333333333, 0.12043269230769231, 0.1453525641025641, 0.1657852564102564, 0.13084935897435898, 0.08517628205128205, 0.00857371794871795], [0.12889083735203857, 0.09294169224024551, 0.08187198597106532, 0.08241999123191583, 0.12121876370013152, 0.14346777729066199, 0.16100394563787812, 0.10960105217010083, 0.06959666812801403, 0.00898728627794827], [0.08751812310531172, 0.06695663635165415, 0.07657835771714776, 0.07512850929221036, 0.12086463687887175, 0.15144325820482404, 0.1609331751680506, 0.17371820218795309, 0.08316857783049954, 0.003690523263477], [0.06990291262135923, 0.05101500441306266, 0.07378640776699029, 0.07837599293909973, 0.12127096204766108, 0.12727272727272726, 0.16398940864960282, 0.22683142100617829, 0.0854368932038835, 0.002118270079435128], [0.051134179161860825, 0.04325259515570934, 0.049980776624375244, 0.08650519031141868, 0.10553633217993079, 0.17493271818531334, 0.2368319876970396, 0.18166089965397925, 0.06978085351787774, 0.00038446751249519417], [0.06600097418412079, 0.05674622503653191, 0.05674622503653191, 0.10594252313687287, 0.10496833901607404, 0.1602532878714077, 0.2109108621529469, 0.1797369702873843, 0.05820750121773015, 0.0004870920603994155], [0.08690095846645367, 0.05005324813631523, 0.05175718849840256, 0.07965921192758253, 0.13525026624068157, 0.15484558040468582, 0.2159744408945687, 0.16741214057507986, 0.058146964856230034, 0.0], [0.05795555555555556, 0.03306666666666667, 0.06435555555555555, 0.07875555555555555, 0.11964444444444444, 0.17724444444444445, 0.23413333333333333, 0.17457777777777778, 0.06026666666666667, 0.0], [0.05984455958549223, 0.06994818652849741, 0.06683937823834196, 0.07564766839378238, 0.11606217616580311, 0.16036269430051814, 0.23730569948186528, 0.15699481865284975, 0.05699481865284974, 0.0], [0.077466251298027, 0.11194184839044652, 0.08909657320872275, 0.102803738317757, 0.13727933541017653, 0.1536863966770509, 0.16510903426791276, 0.13374870197300104, 0.028868120456905504, 0.0], [0.08529335745670716, 0.06513310933057638, 0.061773067976221244, 0.10157663478935125, 0.14913414318945464, 0.16515895580253295, 0.19048849831997933, 0.14629103127423107, 0.03515120186094598, 0.0], [0.08094285078941517, 0.044029352901934625, 0.06826773404491883, 0.08561263064265065, 0.17078052034689792, 0.19012675116744496, 0.1959083833666889, 0.149210584834334, 0.01512119190571492, 0.0], [0.07556270096463022, 0.04796355841371919, 0.07234726688102894, 0.10637727759914255, 0.15380493033226153, 0.18837084673097534, 0.22588424437299034, 0.12111468381564845, 0.00857449088960343, 0.0], [0.07034632034632035, 0.04843073593073593, 0.09767316017316018, 0.13041125541125542, 0.1396103896103896, 0.1680194805194805, 0.21022727272727273, 0.1314935064935065, 0.003787878787878788, 0.0], [0.07020872865275142, 0.06993765248034697, 0.10355109785849824, 0.09867172675521822, 0.1461100569259962, 0.16671184602873407, 0.22716183247492544, 0.11656275413391162, 0.0010843046896177825, 0.0], [0.047543997898607825, 0.059627002889414235, 0.079852902547938, 0.09929078014184398, 0.14657210401891252, 0.20961386918833727, 0.24192277383766744, 0.11531389545573943, 0.0002626740215392698, 0.0], [0.08490791676632385, 0.04448696484094714, 0.07653671370485529, 0.11911026070318105, 0.1478115283425018, 0.20808419038507533, 0.23558957187275772, 0.08347285338435781, 0.0, 0.0], [0.08664259927797834, 0.06446621970087674, 0.08767405879319237, 0.13073749355337802, 0.1333161423414131, 0.22021660649819494, 0.2145435791645178, 0.062403300670448685, 0.0, 0.0], [0.06147291539866099, 0.05356055995130858, 0.08926759991884764, 0.10428078717792656, 0.12010549807263136, 0.2692229661188882, 0.24305132886995334, 0.05903834449178332, 0.0, 0.0], [0.18836443468715697, 0.05773874862788145, 0.09506037321624589, 0.10230515916575192, 0.13238199780461032, 0.2199780461031833, 0.17189901207464325, 0.0322722283205269, 0.0, 0.0], [0.0636527215474584, 0.03733693207377418, 0.07669815564552407, 0.11740890688259109, 0.19163292847503374, 0.25483580746738643, 0.2197480881691408, 0.038686459739091315, 0.0, 0.0], [0.1259127688967831, 0.04085257548845471, 0.07973159660548648, 0.1902506414051707, 0.1760410499309256, 0.2054470100651273, 0.1529504637852773, 0.028813893822774817, 0.0, 0.0], [0.10036880927291886, 0.04926238145416228, 0.0798208640674394, 0.12276080084299262, 0.18361433087460485, 0.2555321390937829, 0.19204425711275025, 0.016596417281348787, 0.0, 0.0], [0.05998400426552919, 0.052252732604638766, 0.11223673687016796, 0.12903225806451613, 0.23060517195414557, 0.24366835510530524, 0.16502266062383364, 0.007198080511863503, 0.0, 0.0], [0.09201940772963024, 0.0642462773966873, 0.10506943282583235, 0.16647147398360382, 0.2912832524677932, 0.17801572695332107, 0.10222519658691652, 0.0006692320562154927, 0.0, 0.0], [0.12619198982835347, 0.08836617927527018, 0.1853146853146853, 0.14351557533375714, 0.21090273363000636, 0.15416401780038144, 0.0912269548633185, 0.0003178639542275906, 0.0, 0.0], [0.10134932533733133, 0.08245877061469266, 0.10434782608695652, 0.12308845577211394, 0.24407796101949025, 0.20599700149925038, 0.1386806596701649, 0.0, 0.0, 0.0], [0.1407165001722356, 0.09541853255253187, 0.09455735446090252, 0.13658284533241474, 0.1846365828453324, 0.19583189803651396, 0.1522562866000689, 0.0, 0.0, 0.0], [0.08539560822586267, 0.07668177065179504, 0.09341233879400487, 0.12635064482398048, 0.19553851516207738, 0.2574067619379575, 0.16521436040432205, 0.0, 0.0, 0.0], [0.14731272626009467, 0.09927596769702032, 0.1642996379838485, 0.1464773043720412, 0.19896964633806738, 0.1743247006404901, 0.06934001670843776, 0.0, 0.0, 0.0], [0.12735681252223408, 0.13980789754535752, 0.1200640341515475, 0.14834578441835647, 0.2248310209889719, 0.17395944503735325, 0.06563500533617929, 0.0, 0.0, 0.0], [0.08406874129122155, 0.09126799814212726, 0.11820715281003251, 0.18114259173246633, 0.2742684626103112, 0.20181142591732465, 0.04923362749651649, 0.0, 0.0, 0.0], [0.10159680638722556, 0.12215568862275449, 0.16127744510978043, 0.20778443113772455, 0.22734530938123754, 0.16387225548902196, 0.015968063872255488, 0.0, 0.0, 0.0], [0.0890738813735692, 0.08553590010405827, 0.12154006243496358, 0.24432882414151924, 0.2651404786680541, 0.1870967741935484, 0.007284079084287201, 0.0, 0.0, 0.0], [0.12253968253968255, 0.09523809523809523, 0.17354497354497356, 0.22306878306878306, 0.21587301587301588, 0.16783068783068783, 0.0019047619047619048, 0.0, 0.0, 0.0], [0.11134020618556702, 0.10592783505154639, 0.14587628865979382, 0.19871134020618555, 0.277319587628866, 0.16056701030927836, 0.0002577319587628866, 0.0, 0.0, 0.0], [0.16406115787665349, 0.11166466242913589, 0.13021817557120768, 0.16869953616217145, 0.3169558495103934, 0.10840061845043807, 0.0, 0.0, 0.0, 0.0], [0.0943921978404737, 0.13723441309648207, 0.13688610240334378, 0.22378962034134448, 0.2878787878787879, 0.1198188784395681, 0.0, 0.0, 0.0, 0.0], [0.07246121297602257, 0.11177715091678421, 0.15267983074753175, 0.26533850493653033, 0.3191114245416079, 0.07863187588152327, 0.0, 0.0, 0.0, 0.0], [0.10343814281516309, 0.12180429033205994, 0.17572729944166912, 0.25374669409344697, 0.29694387305318837, 0.04833970026447253, 0.0, 0.0, 0.0, 0.0], [0.09673226884515411, 0.10230226513182325, 0.18975120683252877, 0.26735982176011885, 0.29576680282213147, 0.04808763460824359, 0.0, 0.0, 0.0, 0.0], [0.10484003281378179, 0.1893355209187859, 0.19950779327317472, 0.27071369975389664, 0.21443806398687448, 0.021164889253486464, 0.0, 0.0, 0.0, 0.0], [0.12729211087420042, 0.11961620469083156, 0.17803837953091683, 0.2929637526652452, 0.2656716417910448, 0.016417910447761194, 0.0, 0.0, 0.0, 0.0], [0.0952569954356023, 0.13474895812661242, 0.2196864457233578, 0.3155387973804326, 0.22861678904544552, 0.0061520142885493154, 0.0, 0.0, 0.0, 0.0], [0.10923454099700354, 0.12721329338055026, 0.20975211114137837, 0.37047126123672025, 0.18196676654862434, 0.001362026695723236, 0.0, 0.0, 0.0, 0.0], [0.11394891944990176, 0.14227242960052391, 0.2683366077275704, 0.2722658808120498, 0.20317616240995415, 0.0, 0.0, 0.0, 0.0, 0.0], [0.09662036077130416, 0.19552145967240306, 0.2695417789757412, 0.2865436450342111, 0.15177275554634045, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1532460996477101, 0.18193256165072974, 0.2745344740815299, 0.28459989934574736, 0.10568696527428284, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16505032021957913, 0.2093321134492223, 0.26422689844464775, 0.2860018298261665, 0.07538883806038427, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21538782318598831, 0.2062135112593828, 0.2427022518765638, 0.2737698081734779, 0.06192660550458716, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16805212311840037, 0.18782296113232982, 0.3190294315884071, 0.2761177263536284, 0.04897775780723433, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1666267082234476, 0.23543514744665547, 0.2759530088707744, 0.272596499640374, 0.0493886358187485, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27543483440552774, 0.20181081725041697, 0.2832975935191804, 0.21920419347152728, 0.020252561353347628, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25513472771810813, 0.2328999434708875, 0.2833992839645751, 0.21914452609760693, 0.00942151874882231, 0.0, 0.0, 0.0, 0.0, 0.0], [0.34387672343876724, 0.23012976480129765, 0.24817518248175183, 0.17538523925385238, 0.0024330900243309003, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2681252681252681, 0.2524667524667525, 0.29386529386529386, 0.18511368511368512, 0.000429000429000429, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2142701287366354, 0.27252891119354133, 0.3423521710669867, 0.17084878900283657, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24791439970982954, 0.28545520493289805, 0.3340587595212187, 0.1325716358360537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2536732929991357, 0.3256266205704408, 0.30293863439930857, 0.11776145203111495, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19537130659476462, 0.27075609482080665, 0.4592742388495675, 0.07459835973486126, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.33405172413793105, 0.3083855799373041, 0.2966300940438871, 0.06093260188087774, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.33103624298111284, 0.315211842776927, 0.31368044920877997, 0.040071465033180195, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.29384271489534647, 0.3182280024385288, 0.3564316195895143, 0.03149766307661044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.39650452726889873, 0.3404927353126974, 0.24847336281322382, 0.014529374605180037, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3608625831612755, 0.4239504473503097, 0.2112869924294563, 0.0038999770589584768, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47443284119553475, 0.34929780338494776, 0.17554915376305366, 0.0007202016564638099, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45094900525954723, 0.383032243311228, 0.1660187514292248, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4406318082788671, 0.4234749455337691, 0.13589324618736384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5198156682027649, 0.3634408602150538, 0.11674347158218126, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4132471008028546, 0.4745762711864407, 0.11217662801070473, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5942176870748299, 0.33877551020408164, 0.06700680272108843, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5829682365826944, 0.36829134720700984, 0.04874041621029573, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6154531284791805, 0.3520374081496326, 0.03250946337118682, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6669753086419753, 0.3237654320987654, 0.009259259259259259, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6085219707057257, 0.3857079449622725, 0.005770084332001775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.645937813440321, 0.352390504847877, 0.001671681711802073, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7432605905006419, 0.25673940949935814, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7532078699743371, 0.24679213002566297, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8175837320574163, 0.18241626794258373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8556985294117647, 0.14430147058823528, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8094229751191107, 0.19057702488088937, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9176755447941889, 0.08232445520581114, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9608559498956158, 0.03914405010438413, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9532467532467532, 0.046753246753246755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9866793529971456, 0.013320647002854425, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.999479979199168, 0.0005200208008320333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
5000: ([0.3167225781764659, 0.028847512680643457, 0.01465475820355558, 0.010856544038195207, 0.008959611090698398, 0.00696810326272467, 0.006423482399288623, 0.005177702939213355, 0.0059940907005715804, 0.007069200548751721, 0.00524944940026481, 0.0061571508393248875, 0.005315760523357822, 0.005692972977673806, 0.00491898085239144, 0.005016816935643424, 0.00529184503634067, 0.005169006398479845, 0.005533174041695565, 0.005328805334458086, 0.006511534874215409, 0.006304992031794553, 0.0060256156607305525, 0.0067289483925531525, 0.005149439181829448, 0.005378810443675767, 0.0075616421677867085, 0.008230188736675269, 0.008759590653827674, 0.007794274632408094, 0.007563816302970086, 0.009454226844916763, 0.007525768937260981, 0.00611258106806565, 0.00712029272556109, 0.006712642378677821, 0.006481096981648125, 0.005783199587783969, 0.007712744563031441, 0.009142238446102102, 0.008052996719230009, 0.009391176924598817, 0.008007339880379083, 0.009099842810026243, 0.007242044295830226, 0.00819540257374123, 0.006216939556867767, 0.00880416042508691, 0.007982337325770241, 0.0074898957067352535, 0.008700889003876482, 0.008425860903179238, 0.007876891769376436, 0.007557293897419954, 0.007692090278789355, 0.008066041530330272, 0.008010601083154148, 0.007862759890684483, 0.008148658667298615, 0.00878785441121158, 0.007923635675819052, 0.01336767017499614, 0.00875415531586923, 0.007649694642713495, 0.008816118168595488, 0.00935530369407309, 0.008147571599706926, 0.010367363621935285, 0.008420425565220794, 0.007307268351331549, 0.007575774046478662, 0.007900807256393588, 0.007299658878189728, 0.00790841672953541, 0.008580224501199036, 0.008834598317654195, 0.008375855793961558, 0.0068463516924555335, 0.007618169682554522, 0.007742095388007035, 0.006901792139631658, 0.007601863668679191, 0.006656114863910008, 0.005700582450815627, 0.005649490274006257, 0.006074533702356545, 0.004345009163979798, 0.00463090794059393, 0.003955838966155238, 0.003153583083488965, 0.002170873980602366, 0.0016273401847580078, 0.0014447128293543037, 0.0010153211306372608, 0.0004826580107097899, 0.00045982959128432687, 0.00028915997938919847, 3.261202775066148e-05, 4.348270366754864e-06, 0.0] ,
[[0.07404712464176005, 0.04617734379022155, 0.05742135882342846, 0.07402309896861217, 0.10320056288719946, 0.12975922843266804, 0.1537608759073982, 0.16257143347462719, 0.14706114533816134, 0.05197782773592353], [0.09243697478991597, 0.06270490258883822, 0.06986471718732336, 0.09145721068696537, 0.12130233259222972, 0.13513207973772468, 0.15702603911519764, 0.14515582017560388, 0.10298828051399932, 0.021931642612201832], [0.0909428083970032, 0.06646391217268749, 0.06928269416215414, 0.09161041465766634, 0.11163860247756101, 0.1351531785475855, 0.15696164972924856, 0.13938135153178546, 0.11853720050441362, 0.020028187819894665], [0.11775307900270351, 0.08491038349854811, 0.07479723640732953, 0.07529788725342945, 0.11094422749574447, 0.13207169320116152, 0.15269850806047863, 0.11414839291078402, 0.11775307900270351, 0.01962551316711725], [0.08056297015287552, 0.061635525357922834, 0.07049259888376608, 0.06915797136617326, 0.11162339238049017, 0.13965057024993935, 0.15190487745692793, 0.18163067216694978, 0.1161125940305751, 0.017228827954380004], [0.061778471138845555, 0.045085803432137285, 0.06521060842433697, 0.06957878315132605, 0.10764430577223089, 0.11497659906396256, 0.15756630265210608, 0.23057722308892356, 0.1372854914196568, 0.01029641185647426], [0.04501607717041801, 0.03807750888475207, 0.04400067693349129, 0.07649348451514638, 0.09358605517007954, 0.15552546962260957, 0.22084955153156202, 0.19800304620071077, 0.1240480622778812, 0.004400067693349129], [0.05689691370984674, 0.04891874868780181, 0.04891874868780181, 0.09216880117572958, 0.09153894604241025, 0.14150745328574427, 0.1960948981734201, 0.20575267688431662, 0.11673315137518371, 0.0014696619777451185], [0.07399347116430903, 0.042618788538266235, 0.044069640914037, 0.06800870511425462, 0.11606819006166122, 0.13565469713456657, 0.20021762785636563, 0.2025752629669931, 0.11661225970257526, 0.00018135654697134566], [0.05013070890358296, 0.02860218360756574, 0.055666615408273105, 0.06827618022451176, 0.10425957250499769, 0.16069506381669998, 0.2200522835614332, 0.20421343995079194, 0.10810395202214362, 0.0], [0.04783599088838269, 0.05591219714226548, 0.053427210602609236, 0.060882170221577964, 0.09463657071857527, 0.1377096707392835, 0.2265479395319942, 0.22385587078069993, 0.09919237937461173, 0.0], [0.0658545197740113, 0.09516242937853107, 0.0759180790960452, 0.08792372881355932, 0.11793785310734463, 0.13788841807909605, 0.16348870056497175, 0.1942090395480226, 0.06161723163841808, 0.0], [0.06748466257668712, 0.051533742331288344, 0.04887525562372188, 0.08077709611451943, 0.1196319018404908, 0.1394683026584867, 0.18752556237218815, 0.2310838445807771, 0.0736196319018405, 0.0], [0.06950544204697345, 0.03780790528928776, 0.058621348100057286, 0.07427916746228756, 0.14970402902425053, 0.17147221691808287, 0.20565209089173192, 0.19744128317739162, 0.035516517089936986, 0.0], [0.06232044198895027, 0.04022099447513812, 0.05966850828729282, 0.08839779005524862, 0.12839779005524862, 0.16397790055248618, 0.22983425414364642, 0.19889502762430938, 0.028287292817679558, 0.0], [0.056338028169014086, 0.03878656554712893, 0.07822318526543878, 0.10465872156013001, 0.11289274106175515, 0.14821235102925243, 0.22253521126760564, 0.2249187432286024, 0.013434452871072589, 0.0], [0.053204601479046834, 0.052999178307313065, 0.07867707477403452, 0.07539030402629417, 0.11400986031224322, 0.14564502875924404, 0.22103533278553822, 0.2456861133935908, 0.013352506162695153, 0.0], [0.0380651945320715, 0.04794952681388013, 0.06414300736067298, 0.08033648790746582, 0.12197686645636173, 0.18843322818086225, 0.2618296529968454, 0.19621451104100945, 0.0010515247108307045, 0.0], [0.06974459724950884, 0.03654223968565815, 0.06286836935166994, 0.09882121807465619, 0.1280943025540275, 0.1988212180746562, 0.2597249508840864, 0.1449901768172888, 0.0003929273084479371, 0.0], [0.06854345165238677, 0.050999592003263976, 0.06956344349245205, 0.10465116279069768, 0.11403508771929824, 0.20379436964504283, 0.24949000407996735, 0.13892288861689106, 0.0, 0.0], [0.05058430717863105, 0.04407345575959933, 0.07362270450751252, 0.08681135225375626, 0.10567612687813022, 0.25242070116861437, 0.27479131886477465, 0.11202003338898163, 0.0, 0.0], [0.14810344827586208, 0.0453448275862069, 0.075, 0.08206896551724138, 0.11362068965517241, 0.21948275862068967, 0.22482758620689655, 0.09155172413793103, 0.0, 0.0], [0.051055385170485294, 0.030488904925130796, 0.06187984845751398, 0.09561609236875339, 0.16092368753382644, 0.246797762944254, 0.267544650911059, 0.08569366768897708, 0.0, 0.0], [0.10306946688206785, 0.03360258481421648, 0.06591276252019386, 0.15735056542810985, 0.15282714054927302, 0.20678513731825526, 0.22697899838449112, 0.05347334410339257, 0.0, 0.0], [0.08043065231158962, 0.03947646189571459, 0.06438674266413341, 0.10111885159383577, 0.16022799240025332, 0.24699176694110198, 0.2634578847371754, 0.043909647456195904, 0.0, 0.0], [0.04547291835084883, 0.039611964430072755, 0.0854890864995958, 0.10044462409054164, 0.18977364591754245, 0.24191592562651576, 0.2698059822150364, 0.0274858528698464, 0.0, 0.0], [0.07906843013225992, 0.05534790109258195, 0.09100057504312824, 0.1466359976998275, 0.26682001150086254, 0.20040253018976423, 0.15583668775158138, 0.00488786658999425, 0.0, 0.0], [0.10487386078457271, 0.07383436798309338, 0.15440496631884823, 0.12389380530973451, 0.19297318716153744, 0.19323735305772025, 0.15480121516312245, 0.001981244221371021, 0.0, 0.0], [0.08401588483494664, 0.06837925043435096, 0.08711839166046165, 0.10585753288657235, 0.22238768925291635, 0.2489451476793249, 0.18279970215934474, 0.0004964010920824026, 0.0, 0.0], [0.11394700139470014, 0.07768479776847978, 0.07782426778242678, 0.11520223152022316, 0.17712691771269176, 0.23430962343096234, 0.20390516039051604, 0.0, 0.0, 0.0], [0.07042253521126761, 0.06338028169014084, 0.07832710549008336, 0.11195745903995401, 0.18899108939350387, 0.29936763437769476, 0.18755389479735557, 0.0, 0.0, 0.0], [0.12165114407266873, 0.08244222145567437, 0.13717373807059904, 0.12958491433827757, 0.2021386685063815, 0.23145912383580544, 0.09555018972059331, 0.0, 0.0, 0.0], [0.10342337137079301, 0.11382348692763253, 0.09952332803697819, 0.13130145890509895, 0.21869131879243103, 0.24035822620251335, 0.09287880976455294, 0.0, 0.0, 0.0], [0.0643784456695714, 0.07060288102436423, 0.0951449404232616, 0.15721145296105282, 0.2665836741952694, 0.2768984527832118, 0.06918015294326872, 0.0, 0.0, 0.0], [0.07816793893129771, 0.09404580152671756, 0.12564885496183206, 0.17587786259541985, 0.24916030534351144, 0.24458015267175573, 0.03251908396946565, 0.0, 0.0, 0.0], [0.06947368421052631, 0.06753036437246963, 0.09813765182186235, 0.20793522267206477, 0.27174089068825913, 0.26105263157894737, 0.024129554655870446, 0.0, 0.0, 0.0], [0.09728279100972828, 0.07598121435759812, 0.14089231801408922, 0.19540422676954042, 0.24052331432405233, 0.24035558537403556, 0.009560550150956054, 0.0, 0.0, 0.0], [0.08139097744360903, 0.07838345864661654, 0.11165413533834587, 0.18778195488721805, 0.2926691729323308, 0.24454887218045113, 0.0035714285714285713, 0.0, 0.0, 0.0], [0.1350246652572234, 0.09217758985200845, 0.11106412966878083, 0.15771670190274842, 0.3274136715997181, 0.1766032417195208, 0.0, 0.0, 0.0, 0.0], [0.06456599286563615, 0.09524375743162901, 0.10297265160523186, 0.1929845422116528, 0.3513674197384067, 0.1928656361474435, 0.0, 0.0, 0.0, 0.0], [0.05588552915766739, 0.08693304535637149, 0.12526997840172785, 0.24176565874730022, 0.3490820734341253, 0.14106371490280778, 0.0, 0.0, 0.0, 0.0], [0.08160666743836092, 0.0973492302349809, 0.1464289848362079, 0.24354670679476792, 0.3318671142493344, 0.09920129644634795, 0.0, 0.0, 0.0, 0.0], [0.07113765951669834, 0.07711105077382568, 0.14852022807493892, 0.2379853380396416, 0.3624762421938637, 0.10276948140103177, 0.0, 0.0, 0.0, 0.0], [0.07693226615697049, 0.1390514872775057, 0.15864293393859755, 0.2643650698841238, 0.3027117429219926, 0.05829649982080994, 0.0, 0.0, 0.0, 0.0], [0.09066346442509757, 0.08946262383668568, 0.13989792854998498, 0.2775442809966977, 0.35980186130291203, 0.04262984088862203, 0.0, 0.0, 0.0, 0.0], [0.06446478312773578, 0.0939116593712694, 0.1663350576999602, 0.305080249369943, 0.3426183844011142, 0.027589866029977452, 0.0, 0.0, 0.0, 0.0], [0.07308970099667775, 0.0890015737016961, 0.16156670746634028, 0.36422451477531037, 0.299702745235181, 0.012414757824794544, 0.0, 0.0, 0.0, 0.0], [0.0880355599456723, 0.1116187183602914, 0.21891591554512904, 0.2915174712927522, 0.2871959501172984, 0.002716384738856649, 0.0, 0.0, 0.0, 0.0], [0.0653683780471197, 0.1357755685687049, 0.20672749557401607, 0.3117254528122021, 0.28040310499795723, 0.0, 0.0, 0.0, 0.0, 0.0], [0.09114658925979681, 0.11204644412191582, 0.19854862119013061, 0.32249637155297534, 0.2757619738751814, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11444277861069466, 0.14992503748125938, 0.21239380309845077, 0.37306346826586706, 0.15017491254372814, 0.0, 0.0, 0.0, 0.0, 0.0], [0.13701457876403045, 0.1448845310282544, 0.20990839891626886, 0.36034060121274675, 0.14785189007869953, 0.0, 0.0, 0.0, 0.0, 0.0], [0.10557548992547612, 0.12613855920507866, 0.2717361302787745, 0.3600607231576042, 0.13648909743306653, 0.0, 0.0, 0.0, 0.0, 0.0], [0.10299194476409666, 0.1609608745684695, 0.24309551208285385, 0.37097238204833144, 0.12197928653624857, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16633691351045787, 0.14330130016958734, 0.25593555681175806, 0.3671565856416054, 0.06726964386659129, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18557951482479784, 0.1871967654986523, 0.27991913746630726, 0.3157681940700809, 0.03153638814016173, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2346315646627765, 0.18116433708780025, 0.259601031347537, 0.31415388790880716, 0.010449178993079115, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18609152495506706, 0.19397207244573483, 0.30734135213604313, 0.3056822895064289, 0.006912760956726116, 0.0, 0.0, 0.0, 0.0, 0.0], [0.13754002134471718, 0.20424226254002134, 0.3160352187833511, 0.3415154749199573, 0.0006670224119530416, 0.0, 0.0, 0.0, 0.0, 0.0], [0.17998515586343394, 0.2346610588817417, 0.36516575952498764, 0.22018802572983673, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1688846206612704, 0.25737412539442994, 0.3520373165043216, 0.22170393743997804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.15141904529560055, 0.2400585508660649, 0.48019842237944216, 0.1283239814588924, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2298522289829877, 0.2651185893455855, 0.3863156587607103, 0.1187135229107165, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2026431718061674, 0.24683814125337503, 0.4483444649708683, 0.10217422196958931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2033292231812577, 0.30924784217016027, 0.42022194821208386, 0.06720098643649815, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23716012084592145, 0.2864280734371369, 0.4312107831745294, 0.04520102254241227, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2581721147431621, 0.3886591060707138, 0.34142761841227487, 0.011741160773849233, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3198070672119115, 0.3506343713956171, 0.32620320855614976, 0.0033553528363216944, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.30338239091143815, 0.3919442292796282, 0.30467337980893366, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3119607259744124, 0.44183278786075575, 0.2462064861648319, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.359305495766968, 0.4311952934423877, 0.2094992107906443, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.31370390753990096, 0.50357732526142, 0.18271876719867913, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.37259865971705136, 0.48979895755770664, 0.137602382725242, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42515463917525775, 0.4762886597938144, 0.09855670103092784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47700494108703917, 0.44875205878626634, 0.07424300012669453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45158114925556786, 0.5058447151470408, 0.04257413559739141, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3861129136924075, 0.5970149253731343, 0.016872160934458143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5500158780565259, 0.44537948555096857, 0.004604636392505557, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6110159817351598, 0.3888413242009132, 0.00014269406392694063, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.643920247121595, 0.35607975287840493, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6331705780437864, 0.3668294219562136, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7148577148577149, 0.2851422851422851, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7788665686754859, 0.22113343132451413, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8464912280701754, 0.15350877192982457, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8801231479699827, 0.11987685203001731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9312813171080888, 0.06871868289191124, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9657242932199149, 0.03427570678008506, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9884976525821596, 0.011502347417840376, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9953283869194833, 0.004671613080516625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
10000: ([0.2668256754418846, 0.02408379227819267, 0.012705984544851648, 0.009349499726074796, 0.007729376315561835, 0.006096637928548773, 0.005618169891294945, 0.004615279259536922, 0.00527756552579222, 0.006157910671549263, 0.004734220466537873, 0.005569512124794556, 0.00483874455753871, 0.005037880972290303, 0.00455130330728641, 0.004505348750036043, 0.004845953115538768, 0.004718001211037744, 0.005372177849542977, 0.004935159020789481, 0.006032661976298261, 0.005674036215795392, 0.005619070961044952, 0.006326410714800611, 0.004741429024537932, 0.005223501340791788, 0.007148186326807185, 0.007325697067558605, 0.008060969983564488, 0.007221172976557769, 0.007267127533808137, 0.00920622963582365, 0.007228381534557827, 0.005890292955797122, 0.006903095354805225, 0.007040959026556328, 0.006564293128802514, 0.005920929327297367, 0.0073121810213084976, 0.008593502205818749, 0.007630258643061042, 0.00875749690032006, 0.0077221677575617775, 0.008615127879818921, 0.007429320088809434, 0.008070881750814567, 0.007092320002306738, 0.008984566477321876, 0.008650269600069203, 0.0075987212018107896, 0.008401574349067213, 0.008250194631066002, 0.008107825610564863, 0.007807769383812462, 0.008051959286064416, 0.008264611747066117, 0.008874635967820997, 0.009030521034572244, 0.009107111963322857, 0.009705422277327643, 0.009479253770075835, 0.014033260286612266, 0.009246777774573974, 0.008328587699316629, 0.00919631786857357, 0.010101892967330816, 0.008837692108070701, 0.011032698019088261, 0.010210922407081687, 0.008532229462818258, 0.00892870015282143, 0.00975858539257807, 0.009168384706323347, 0.010279403708082235, 0.010206417058331651, 0.010282106917332256, 0.010580361004584643, 0.009499077304575993, 0.009849593437328797, 0.010254173755082033, 0.00934859865632479, 0.010934481416337476, 0.008693520948069548, 0.00803483896081428, 0.008491681324067933, 0.00983247311207866, 0.00738877195005911, 0.006151603183299213, 0.006951753121305614, 0.006402100573801217, 0.0038511721115308093, 0.004004353969032035, 0.003901632017531213, 0.003823238949280586, 0.0026338268792710706, 0.0015074896917620598, 0.0008776419365070211, 0.00021715780975173726, 0.00012524869525100198, 9.010697500072086e-07] ,
[[0.07285535304824717, 0.04543412996714181, 0.05649717514124294, 0.07283171406283243, 0.10153957335008325, 0.12767416022504313, 0.15133003062937112, 0.16022842013906477, 0.14849672937751798, 0.06311271405945543], [0.09177641424723136, 0.0622568093385214, 0.06936545944328046, 0.09080365160131697, 0.1204354983537863, 0.1341664172403472, 0.15594133492966178, 0.14475456450164623, 0.10505836575875487, 0.025441484585453457], [0.08694418835543578, 0.06354159279483725, 0.06623643713211828, 0.08758244096163392, 0.10673001914757818, 0.12921069427700163, 0.15027303028154032, 0.134387632082831, 0.12835969080207077, 0.04673427416495284], [0.1133384734001542, 0.08172706245181187, 0.07199306090979182, 0.07247494217424827, 0.10678488820354665, 0.12712027756360833, 0.14716653816499614, 0.11189282960678489, 0.13483037779491133, 0.03267154973014649], [0.07740732105385871, 0.059221263697831664, 0.06773140592212637, 0.06644905572394498, 0.10725110748426206, 0.13418046164607134, 0.1460713453019352, 0.1763814408953136, 0.1318489158311961, 0.033457682443460014], [0.05852793378657996, 0.04271356783919598, 0.06177948566361218, 0.06591782441619864, 0.10198049068873781, 0.10892698788057936, 0.1495713863434821, 0.2230268992018918, 0.16095181791309487, 0.026603606266627253], [0.042662389735364875, 0.03608660785886127, 0.04170008019246191, 0.07249398556535686, 0.08869286287089014, 0.1475541299117883, 0.2094627105052125, 0.1926222935044106, 0.156214915797915, 0.012510024057738572], [0.05290901991409606, 0.045490042951971885, 0.045490042951971885, 0.0857087075361187, 0.08512299882858258, 0.13158922295978134, 0.18352206169465052, 0.19875048809058962, 0.1671222178836392, 0.004295197188598204], [0.06966023561550282, 0.04012292982755677, 0.041488816800409764, 0.0640259518524842, 0.10927095782823971, 0.12771043196175516, 0.18883387399692675, 0.20010244152296397, 0.15758920949291447, 0.001195151101246372], [0.04770266315481417, 0.027216856892010536, 0.05297044190810653, 0.0649692712906058, 0.09920983318700614, 0.15320456540825286, 0.2099795141937372, 0.20280948200175591, 0.14193737196371087, 0.0], [0.04396650171298059, 0.051389417586600684, 0.049105443471640656, 0.05595736581652075, 0.08698134754472783, 0.12657023220403502, 0.2125999238675295, 0.23277502854967644, 0.14065473924628855, 0.0], [0.060346222294127165, 0.08720271800679502, 0.0695680310629348, 0.08056948713800356, 0.10807312732567546, 0.12635495874453972, 0.1522407377447015, 0.20643908752629025, 0.10920563015693253, 0.0], [0.061452513966480445, 0.04692737430167598, 0.044506517690875234, 0.07355679702048418, 0.10893854748603352, 0.12737430167597766, 0.17355679702048418, 0.24376163873370577, 0.11992551210428305, 0.0], [0.06510463244500089, 0.03541405830799499, 0.05490967626542658, 0.06957610445358613, 0.1402253621892327, 0.16079413342872473, 0.19513503845465927, 0.20783401895904133, 0.07100697549633339, 0.0], [0.05583052860819639, 0.03603246881805583, 0.05345476143337953, 0.07919223916056227, 0.11502672738071669, 0.14808948723025145, 0.2136210651356167, 0.22589586220550387, 0.07285686002771728, 0.0], [0.052, 0.0358, 0.0722, 0.0968, 0.1044, 0.1378, 0.2154, 0.2548, 0.0308, 0.0], [0.048159166976571216, 0.04797322424693194, 0.07121606545184084, 0.0682409817776125, 0.10319821494979546, 0.13220528077352176, 0.2097433990330978, 0.29509111193752324, 0.024172554853105245, 0.0], [0.034568372803666925, 0.04354469060351413, 0.05825057295645531, 0.07295645530939648, 0.11077158135981666, 0.1715049656226127, 0.25458365164247515, 0.24732620320855614, 0.006493506493506494, 0.0], [0.059543777255954375, 0.031197584703119757, 0.05367326400536732, 0.08436766185843676, 0.10935927541093593, 0.17108352901710835, 0.2388460248238846, 0.24723247232472326, 0.004696410600469641, 0.0], [0.06134745298521088, 0.045645426328281904, 0.06226036151177652, 0.09366441482563446, 0.10224575497535147, 0.18422494066094577, 0.2499543545736717, 0.20065729413912725, 0.0, 0.0], [0.04525765496639283, 0.039432412247946226, 0.06587005227781927, 0.07766990291262135, 0.0948469006721434, 0.2270351008215086, 0.27064973861090363, 0.17923823749066467, 0.0, 0.0], [0.13641416547562332, 0.04176592027949817, 0.06908051453072891, 0.07590916309353661, 0.10465300936954106, 0.20406542798157853, 0.23773225345402574, 0.13037954581546768, 0.0, 0.0], [0.04538165490699166, 0.02710070558050032, 0.055003207184092365, 0.0849903784477229, 0.14352148813341886, 0.22209749839640797, 0.27726106478511864, 0.14464400256574728, 0.0, 0.0], [0.0908702464036462, 0.02962540948582823, 0.05811138014527845, 0.13872667711152256, 0.13530836063238855, 0.1861558182595072, 0.2624982196268338, 0.09870388833499502, 0.0, 0.0], [0.07240592930444698, 0.035537818320030404, 0.057962751805397186, 0.09103002660585328, 0.14481185860889395, 0.22633979475484606, 0.29228430254656024, 0.07962751805397188, 0.0, 0.0], [0.038813179230636534, 0.033810591685354496, 0.07296877695359669, 0.08573400034500604, 0.16249784371226497, 0.2197688459548042, 0.3303432810074176, 0.05606348111091944, 0.0, 0.0], [0.06933064414471196, 0.04853145090129837, 0.07979326862473213, 0.12857683095928402, 0.2347157443590067, 0.18366317912517333, 0.2247573427454935, 0.030631539140300014, 0.0, 0.0], [0.0976629766297663, 0.06875768757687577, 0.14378843788437884, 0.11549815498154982, 0.18031980319803198, 0.18942189421894218, 0.1980319803198032, 0.006519065190651907, 0.0, 0.0], [0.07567627990163202, 0.061591772859378495, 0.07858260675162083, 0.09546165884194054, 0.20165437066845518, 0.24267829197406662, 0.24133691035099486, 0.0030181086519114686, 0.0, 0.0], [0.10194659346144247, 0.06950336910406787, 0.06962815073621163, 0.10319440978287996, 0.16109308709757925, 0.2267282256051909, 0.2677813825804842, 0.00012478163214374845, 0.0, 0.0], [0.06075635461872288, 0.05468071915685059, 0.0675759454432734, 0.09659020458772474, 0.16404215747055176, 0.2921264724116553, 0.2642281463112213, 0.0, 0.0, 0.0], [0.10355290202603504, 0.07017715572085739, 0.11676617402368601, 0.1107957326025252, 0.17549182734657923, 0.25320544191054123, 0.17001076636977586, 0.0, 0.0, 0.0], [0.08925454998753428, 0.09822986786337572, 0.08588880578409375, 0.11393667414609823, 0.1924707055597108, 0.25978558962852155, 0.16043380703066568, 0.0, 0.0, 0.0], [0.055377084289429404, 0.060731222273214015, 0.08184182346642191, 0.13599510478812912, 0.2354290959155576, 0.3178828208658406, 0.11274284840140737, 0.0, 0.0, 0.0], [0.0668320062655006, 0.0804072575381804, 0.1074272288213027, 0.15089413914632555, 0.2194230518209111, 0.2983944654744811, 0.07662185093329853, 0.0, 0.0, 0.0], [0.05490145891988738, 0.053365753775275145, 0.07755310980291784, 0.16444842590222677, 0.2263885334015869, 0.31238802149987205, 0.11095469669823393, 0.0, 0.0, 0.0], [0.07961564859299931, 0.06218256691832533, 0.1153054221002059, 0.16032944406314345, 0.20988332189430337, 0.3177762525737817, 0.05490734385724091, 0.0, 0.0, 0.0], [0.06604778572515599, 0.06361284431593364, 0.0903971998173794, 0.15294475726677828, 0.2579516055394917, 0.3399786942626693, 0.02906711307259169, 0.0, 0.0, 0.0], [0.11805298829328405, 0.08059149722735674, 0.0973505853357979, 0.13826247689463955, 0.3070856438693777, 0.2568083795440542, 0.0018484288354898336, 0.0, 0.0, 0.0], [0.056936143441333756, 0.08398867568417741, 0.09090909090909091, 0.17154241375694662, 0.33889063646849116, 0.25773303973996015, 0.0, 0.0, 0.0, 0.0], [0.048889938592347665, 0.07605101558809636, 0.1095890410958904, 0.21362777515351913, 0.3410486537553141, 0.21079357581483232, 0.0, 0.0, 0.0, 0.0], [0.07253832698837329, 0.08653153616627225, 0.13046609733511677, 0.2200843708200432, 0.336557258977261, 0.15382240971293343, 0.0, 0.0, 0.0, 0.0], [0.06114352392065344, 0.06627771295215869, 0.1277712952158693, 0.20851808634772462, 0.3628938156359393, 0.17339556592765462, 0.0, 0.0, 0.0, 0.0], [0.06735697102813513, 0.12174458738625667, 0.13910678799288778, 0.23700449743750654, 0.3228741763413869, 0.111912979813827, 0.0, 0.0, 0.0, 0.0], [0.07325651910248636, 0.07252880533656762, 0.11315949060036386, 0.23104912067919953, 0.4037598544572468, 0.10624620982413584, 0.0, 0.0, 0.0, 0.0], [0.05425923858434744, 0.07904432287596293, 0.14111867812883777, 0.2716311264932455, 0.38852294294964834, 0.06542369096795803, 0.0, 0.0, 0.0, 0.0], [0.05310633972811587, 0.06466776775505019, 0.11802820480243933, 0.28331851098970906, 0.436158048532588, 0.04472112819209757, 0.0, 0.0, 0.0, 0.0], [0.07150737137699328, 0.09086350416207001, 0.1787182830207602, 0.25403670644870124, 0.3890281817270083, 0.015845953264466955, 0.0, 0.0, 0.0, 0.0], [0.05, 0.10427083333333333, 0.15947916666666667, 0.2625, 0.41052083333333333, 0.013229166666666667, 0.0, 0.0, 0.0, 0.0], [0.07446934661449069, 0.09154512036048856, 0.16494723111585438, 0.30653385509308667, 0.3623858650539547, 0.00011858176212498517, 0.0, 0.0, 0.0, 0.0], [0.09824109824109824, 0.12902187902187903, 0.18447018447018448, 0.35242385242385244, 0.23584298584298585, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11598951507208388, 0.1226518130187855, 0.1811926605504587, 0.3567059851463521, 0.22346002621231978, 0.0, 0.0, 0.0, 0.0, 0.0], [0.08501889308735275, 0.10180040008890864, 0.2237163814180929, 0.37386085796843743, 0.21560346743720826, 0.0, 0.0, 0.0, 0.0, 0.0], [0.08263127524523947, 0.12960184650894402, 0.20323139065204848, 0.39538372763993074, 0.1891517599538373, 0.0, 0.0, 0.0, 0.0, 0.0], [0.13171441360787825, 0.11425693822739481, 0.21195165622202328, 0.41450313339301703, 0.12757385854968667, 0.0, 0.0, 0.0, 0.0, 0.0], [0.15013083296990842, 0.15165721761883993, 0.2386611426079372, 0.37788922808547754, 0.0816615787178369, 0.0, 0.0, 0.0, 0.0, 0.0], [0.17555081734186212, 0.13635902122042848, 0.21230581784952787, 0.4296882932277389, 0.04609605036044268, 0.0, 0.0, 0.0, 0.0, 0.0], [0.13430453003392537, 0.14158850528836558, 0.2518459389343444, 0.44522051486729197, 0.027040510876072642, 0.0, 0.0, 0.0, 0.0, 0.0], [0.10200850895419017, 0.15385376471752252, 0.2653606411398041, 0.474423666765608, 0.004353418422875235, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1351777922198496, 0.1780707455203788, 0.30934917834927117, 0.3774022839105004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11730038022813688, 0.18127376425855513, 0.3112167300380228, 0.3902091254752852, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11981507640940028, 0.1926929497881084, 0.43720303069217925, 0.2502889431103121, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18066653673747807, 0.21126486065094524, 0.38579224322744105, 0.22227635938413565, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.15471167369901548, 0.1974467164340582, 0.4409823650329979, 0.20685924483392837, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1620615324319028, 0.25553595923966294, 0.4477758181461885, 0.13462669018224574, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18285612345018284, 0.23851574346623852, 0.46454375167246453, 0.11408438141111409, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19789967373572595, 0.3178017944535073, 0.43433931484502447, 0.04995921696574225, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2512250898399216, 0.2941032342371774, 0.434171839268213, 0.02049983665468801, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.20870102364984117, 0.30480056477232614, 0.4828803388633957, 0.0036180727144369926, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.22462773260111943, 0.3637131692892597, 0.4114478825641567, 0.00021121554546414616, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25754364718942374, 0.38298516500151375, 0.35947118780906245, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21532779316712836, 0.4409048938134811, 0.34376731301939056, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25405405405405407, 0.4406879606879607, 0.30525798525798525, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2812061711079944, 0.4704593267882188, 0.24833450210378682, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.34360377858214886, 0.4864483093493423, 0.16994791206850887, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3433529050915783, 0.5331697484883008, 0.12347734642012094, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2869187531936638, 0.6402657128257537, 0.07281553398058252, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36330867008157847, 0.6052931132612408, 0.0313982166571808, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4306101911993413, 0.5630774860488519, 0.006312322751806788, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47627416520210897, 0.5225834797891037, 0.0011423550087873461, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4673734939759036, 0.5326265060240963, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5047383601153688, 0.49526163988463123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6329809286898839, 0.3670190713101161, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7066277896153415, 0.2933722103846585, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.778544142614601, 0.22145585738539897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7309384164222874, 0.2690615835777126, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8621951219512195, 0.1378048780487805, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9641130804159953, 0.03588691958400469, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.986649384316267, 0.013350615683732988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] ),
math.inf: ([0.2506254077827773, 0.022628774843056452, 0.011940217777131711, 0.008802191969764069, 0.0073029035824033915, 0.005798544321128303, 0.005293992366644467, 0.004376164188136184, 0.005029461777442725, 0.005828969564614765, 0.004502936035996444, 0.005312585570997306, 0.004605198659937053, 0.004799582159989452, 0.004447156422937929, 0.004392221955531816, 0.004633933612118712, 0.004562941377316967, 0.005177362266613028, 0.004767466625198186, 0.005808686068957124, 0.005478234118868046, 0.005381042368841846, 0.006090964716859304, 0.004632243320813909, 0.005048900127447965, 0.007080630275821735, 0.0071566933845378915, 0.00818946137177281, 0.006939490951870645, 0.007090772023650556, 0.008796275950197257, 0.006948787554047065, 0.005850943351577211, 0.006692708421369339, 0.0068127191040103854, 0.00644000987130122, 0.005800234612433106, 0.006998651147538767, 0.00827651137397019, 0.007279239504136142, 0.008460753126193769, 0.007519260869418235, 0.008385535163130014, 0.0072741686302217326, 0.007920705054309059, 0.007002031730148374, 0.008639924004502936, 0.0085249841957763, 0.007447423488964088, 0.008295104578323028, 0.00806691525217456, 0.007973104084757968, 0.0076756128151125565, 0.008341587589205123, 0.008494558952289838, 0.008775147308887213, 0.009415767713407729, 0.0092070167372645, 0.00971410412870554, 0.009282234700328254, 0.01353331733190898, 0.009045593917655768, 0.008305246326151849, 0.009347310915563189, 0.010095264817938723, 0.00887571964152302, 0.011169444942141329, 0.010332750746263612, 0.008727819152352717, 0.009092922074190267, 0.010007369670088943, 0.00942083858732214, 0.010587139587636534, 0.010189921131007718, 0.010500089585439154, 0.01058544929633173, 0.009615222087374538, 0.00985439830700423, 0.011415382326990233, 0.014287187253851329, 0.015996071763007637, 0.012411809051171879, 0.008520758467514291, 0.008715987113219092, 0.01016710219839287, 0.007766888545571944, 0.0063909914234619196, 0.007490525917236576, 0.006925123475779816, 0.004661823418647969, 0.004492794288167623, 0.004451382151199937, 0.005052280710057571, 0.007281774941093348, 0.002608119483311754, 0.0012584218764261834, 0.000503706808831434, 0.0002476276761537083, 3.127038913886419e-05] ,
[[0.07275069381919223, 0.045368862271410605, 0.05641601499930871, 0.07272708879199587, 0.10139370824860815, 0.12749075188755915, 0.15111601196437663, 0.15999824648369398, 0.14831375802149407, 0.0644248635123606], [0.09161531279178338, 0.06214752567693744, 0.0692436974789916, 0.09064425770308124, 0.12022408963585435, 0.13393090569561159, 0.15566760037348273, 0.14450046685340803, 0.10491129785247433, 0.02711484593837535], [0.08677802944507361, 0.06342015855039637, 0.06610985277463194, 0.08741506228765572, 0.10652604756511891, 0.12896375990939976, 0.1499858437146093, 0.1341308040770102, 0.1282559456398641, 0.04841449603624009], [0.11291406625060009, 0.0814210273643783, 0.07172347575612098, 0.07220355256841095, 0.10638502160345656, 0.12664426308209314, 0.14661545847335575, 0.1114738358137302, 0.13490158425348056, 0.0357177148343735], [0.07684295799097327, 0.05878949195694943, 0.0672375882421016, 0.06596458743201018, 0.10646915866219188, 0.1332021756741118, 0.14500636500405045, 0.17509547506075684, 0.13169772017127648, 0.03969447980557806], [0.057717533887188456, 0.042122139629791576, 0.060924063547587816, 0.06500510129718699, 0.10056843025797989, 0.10741874362337851, 0.1475003643783705, 0.219938784433756, 0.16163824515376768, 0.03716659379099257], [0.04246487867177522, 0.035919540229885055, 0.041507024265644954, 0.07215836526181353, 0.08828224776500639, 0.14687100893997446, 0.20849297573435505, 0.19173052362707535, 0.15788633461047255, 0.014687100893997445], [0.05233680957898803, 0.04499806875241406, 0.04499806875241406, 0.08478176902278872, 0.08420239474700657, 0.13016608729239088, 0.18153727307840864, 0.1966010042487447, 0.17303978370027037, 0.007338740826573967], [0.06855990589816838, 0.03948916148546463, 0.04083347336582087, 0.06301461939169888, 0.10754495042849942, 0.1256931608133087, 0.18585111745925054, 0.19710972945723407, 0.16938329692488657, 0.002520584775667955], [0.04726692764970277, 0.026968247063940843, 0.052486588371755835, 0.06437581557198782, 0.09830361026533276, 0.15180513266637669, 0.20806147600405975, 0.20110192837465565, 0.14919530230535016, 0.00043497172683775554], [0.04335585585585586, 0.05067567567567568, 0.04842342342342342, 0.05518018018018018, 0.08577327327327328, 0.12481231231231231, 0.20964714714714713, 0.23085585585585586, 0.15127627627627627, 0.0], [0.05933821189945911, 0.08574610244988864, 0.06840598154629335, 0.07922367165128857, 0.10626789691377665, 0.12424435252943047, 0.1496977410117722, 0.20442252624880689, 0.12265351574928413, 0.0], [0.06056157093044595, 0.04624701780143146, 0.043861258946595705, 0.07249036520462471, 0.10735914846760873, 0.1255276197467425, 0.17104055790053221, 0.24206276380987338, 0.13084969719214534, 0.0], [0.06409579151259025, 0.03486529318541997, 0.05405881317133298, 0.06849797499559782, 0.13805247402711746, 0.1583025180489523, 0.19211128719845044, 0.20707871104067618, 0.08293713681986266, 0.0], [0.053591790193842644, 0.03458760927404029, 0.05131128848346636, 0.07601672367920942, 0.11041429114405168, 0.14215127328012162, 0.20505511212466743, 0.22690992018244013, 0.0999619916381604, 0.0], [0.05002886280546469, 0.03444294785453146, 0.06946315181835674, 0.0931306522994035, 0.10044256301712526, 0.13257648643448144, 0.20761978064267847, 0.2568789686357514, 0.05541658649220704, 0.0], [0.04723691409812147, 0.047054532190406714, 0.06985227065475105, 0.06693416013131498, 0.10122195878168885, 0.12967353638519058, 0.2057267919022433, 0.29655298194419116, 0.03574685391209192, 0.0], [0.033524726801259494, 0.04223004260048157, 0.056491942952398594, 0.07075384330431561, 0.10742730135210224, 0.16632709761066863, 0.24689757362474532, 0.2589368401555844, 0.017410631598444155, 0.0], [0.05794972249428665, 0.030362389813907934, 0.05223636957231472, 0.0821090434214822, 0.10643160300359125, 0.1665034280117532, 0.232615083251714, 0.26101860920666015, 0.010773751224289911, 0.0], [0.05956390710866868, 0.04431838326537848, 0.06045027477397624, 0.09094132246055664, 0.0992731785144478, 0.17886899485906754, 0.24428292855876618, 0.22230101045913844, 0.0, 0.0], [0.044085552160628545, 0.03841117415975556, 0.06416412047140986, 0.07565837334497308, 0.09239051360395752, 0.221155245162229, 0.26436781609195403, 0.1997672050050924, 0.0, 0.0], [0.13252082690527614, 0.040573896945387225, 0.06710891700092564, 0.07374267201481025, 0.10166615242209194, 0.1982412835544585, 0.23218142548596113, 0.15396482567108918, 0.0, 0.0], [0.04444793466310664, 0.026543112926024817, 0.05387152505104445, 0.0832417150934506, 0.14056855662007225, 0.21752787812156432, 0.27406942044919114, 0.1597298570755458, 0.0, 0.0], [0.08852504509504648, 0.02886082974885528, 0.05661162758429305, 0.13514638545858193, 0.1318162897183294, 0.1813514638545858, 0.2579436658803941, 0.11974469265991397, 0.0, 0.0], [0.06951286261631089, 0.0341178617040686, 0.05564677978471082, 0.08739281153074256, 0.13902572523262177, 0.21729611384783798, 0.2895457033388068, 0.10746214194490057, 0.0, 0.0], [0.03766320723133579, 0.032808838299296955, 0.07080682959491129, 0.08319383997321728, 0.1576832942751925, 0.2132574489454302, 0.33093404753933714, 0.07365249414127888, 0.0, 0.0], [0.06564812604440201, 0.045953688231081403, 0.07555502506564812, 0.12174743375507281, 0.2222487467175937, 0.1739078539030795, 0.2368106946765338, 0.05812843160658868, 0.0, 0.0], [0.09376476145488899, 0.06601322626358054, 0.13804912612187056, 0.11088804912612187, 0.17312234293811998, 0.18209730751062825, 0.21551724137931033, 0.02054794520547945, 0.0, 0.0], [0.06986584107327141, 0.056862745098039215, 0.07254901960784314, 0.08813209494324045, 0.18617131062951497, 0.22456140350877193, 0.27275541795665637, 0.02910216718266254, 0.0, 0.0], [0.09950066983315065, 0.06783582998416758, 0.0679576178297406, 0.10071854828888077, 0.15722810863475825, 0.22201924247960053, 0.2843746194129826, 0.00036536353671903543, 0.0, 0.0], [0.058402860548271755, 0.05256257449344458, 0.06495828367103695, 0.09284862932061978, 0.15768772348033372, 0.28164481525625745, 0.29189511323003575, 0.0, 0.0, 0.0], [0.10165257494235204, 0.06888931591083781, 0.11462336664104535, 0.10876249039200615, 0.1723674096848578, 0.2495196003074558, 0.18418524212144505, 0.0, 0.0, 0.0], [0.08708343468742398, 0.09584042811967891, 0.08379956215032838, 0.11116516662612504, 0.18803210897591827, 0.2554123084407687, 0.17866699099975675, 0.0, 0.0, 0.0], [0.0522894698829987, 0.057345081612017915, 0.07727863642929365, 0.12841253791708795, 0.22230247002744474, 0.3114256825075834, 0.1509461216235736, 0.0, 0.0, 0.0], [0.0646546281096098, 0.07778759944437429, 0.10392726354337668, 0.14597802752872838, 0.2124005556257103, 0.2956181336027276, 0.09963379214547291, 0.0, 0.0, 0.0], [0.05321920357275772, 0.05173055452177149, 0.07517677707480462, 0.1594095025431088, 0.21945168093288675, 0.31137575983128646, 0.1296365215233842, 0.0, 0.0, 0.0], [0.07611548556430446, 0.0594488188976378, 0.11023622047244094, 0.15328083989501312, 0.20065616797900263, 0.32860892388451446, 0.07165354330708662, 0.0, 0.0, 0.0], [0.06323765117295643, 0.06090630919422993, 0.08655107096022148, 0.14643741803875857, 0.24755937636602068, 0.3578609937345184, 0.037447180533294475, 0.0, 0.0, 0.0], [0.1156865112909069, 0.0789759690858592, 0.09539910638811738, 0.13549088274363, 0.3016543895664775, 0.26929114841202756, 0.003501992512981524, 0.0, 0.0, 0.0], [0.05544776881445931, 0.08179311753293168, 0.08853262534463392, 0.16705810272643726, 0.3309506790564689, 0.27560502399673237, 0.0006126825283365669, 0.0, 0.0, 0.0], [0.04806687565308255, 0.07477069546035063, 0.10774410774410774, 0.21003134796238246, 0.33588761174968074, 0.22349936143039592, 0.0, 0.0, 0.0, 0.0], [0.07042253521126761, 0.08400759164918589, 0.12666067325941463, 0.21366496853461192, 0.32823893716911395, 0.17700529417640595, 0.0, 0.0, 0.0, 0.0], [0.058896257165336634, 0.0638417444082275, 0.12307519388557941, 0.20085422052377205, 0.35393953017871194, 0.1993930538383725, 0.0, 0.0, 0.0, 0.0], [0.06490626889739971, 0.11731505744809514, 0.13404555533158638, 0.22838137472283815, 0.3155613787542834, 0.1397903648457972, 0.0, 0.0, 0.0, 0.0], [0.07017543859649122, 0.06947833159056582, 0.10840013942140118, 0.22133147438131753, 0.3963053328685953, 0.1343092831416289, 0.0, 0.0, 0.0, 0.0], [0.051856594110115235, 0.07554417413572344, 0.13486982501067007, 0.25970977379428084, 0.38583013230900554, 0.09218950064020487, 0.0, 0.0, 0.0, 0.0], [0.050452625226312615, 0.06143633071816536, 0.11213035606517803, 0.2691611345805673, 0.44212432106216054, 0.06469523234761618, 0.0, 0.0, 0.0, 0.0], [0.06974469333855032, 0.08862369167563337, 0.17431282402425902, 0.2478724444879194, 0.3977306074537807, 0.021715739019857183, 0.0, 0.0, 0.0, 0.0], [0.047586001784475064, 0.09923664122137404, 0.15177951819173194, 0.25022305938336475, 0.4349162288093586, 0.016258550609695648, 0.0, 0.0, 0.0, 0.0], [0.07126645483431684, 0.0876078075351793, 0.15785292782569224, 0.2939173853835679, 0.38856105310939626, 0.0007943713118474807, 0.0, 0.0, 0.0, 0.0], [0.09332654100866021, 0.12256749872643913, 0.17524197656647988, 0.3365257259296994, 0.27233825776872134, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11126244106862232, 0.11765322158198009, 0.17391304347826086, 0.3451021477213201, 0.2520691461498167, 0.0, 0.0, 0.0, 0.0, 0.0], [0.08108967564129743, 0.09709561161755353, 0.21337714649141404, 0.36092855628577486, 0.24750900996396014, 0.0, 0.0, 0.0, 0.0, 0.0], [0.078837260515305, 0.12365117815459149, 0.19390002202158116, 0.38416648315349045, 0.21944505615503193, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11925025329280649, 0.10344478216818642, 0.19189463019250252, 0.3894630192502533, 0.19594731509625127, 0.0, 0.0, 0.0, 0.0, 0.0], [0.13700129340364142, 0.13839418963287234, 0.21788876728683712, 0.373694159785096, 0.13302158989155308, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16652219974959068, 0.129346046422036, 0.2013868824039295, 0.431089280554753, 0.07165559086969084, 0.0, 0.0, 0.0, 0.0, 0.0], [0.12081500762947671, 0.12736738174311102, 0.2269993716901535, 0.48056727403285165, 0.04425096490440714, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0946392509638333, 0.14273912245272627, 0.24646594455663667, 0.5047732696897375, 0.011382412337066276, 0.0, 0.0, 0.0, 0.0, 0.0], [0.12667478684531058, 0.1668696711327649, 0.2904993909866017, 0.4151731338089438, 0.0007830172263789803, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11235545843576436, 0.1736319766912501, 0.3013748520440681, 0.4126377128289174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11653031911571848, 0.1874102291887841, 0.4264659963779429, 0.26959345531755446, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.17322246099224517, 0.2026534616462674, 0.37428758292067643, 0.249836494440811, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1455174519181846, 0.18632339472880838, 0.42281469420983003, 0.24534445914317696, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1495479204339964, 0.23625678119349006, 0.4365280289330922, 0.17766726943942135, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.17161992465466722, 0.2240267894516534, 0.4632063624947677, 0.1411469233989117, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18482193867834698, 0.2970862692820415, 0.44029708626928205, 0.07779470577032946, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23274818401937045, 0.27338075060532685, 0.4584594430992736, 0.03541162227602906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19344020938982495, 0.28324881400294455, 0.514886307868477, 0.008424668738753477, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.20596494625738357, 0.33543139343468575, 0.45676382298828316, 0.0018398373196475259, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2371967654986523, 0.35384329398642994, 0.40895994051491774, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1969428257748501, 0.4065535005489401, 0.39650367367620976, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23190095989952453, 0.4068359199784695, 0.3612631201220059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2561666799712621, 0.43585854554163006, 0.30797477448710786, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3228000331757485, 0.4686903873268641, 0.2085095794973874, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3153573728267869, 0.5256761107533805, 0.1589665164198326, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.26922155688622756, 0.6334530938123752, 0.0973253493013972, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3375230728663092, 0.6158038147138964, 0.04667311241979432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4044596912521441, 0.5861921097770154, 0.00934819897084048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.40230991337824834, 0.5961353372325461, 0.001554749389205597, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2888494528246081, 0.711150547175392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.32836688328842395, 0.6716331167115761, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42925234917608607, 0.5707476508239139, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6377702836738742, 0.3622297163261258, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7352855619121497, 0.2647144380878503, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6937655860349127, 0.30623441396508727, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8205658324265506, 0.1794341675734494, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9395662523142025, 0.06043374768579741, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9647974726390612, 0.035202527360938735, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9985355137905785, 0.001464486209421528, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] ),
},
    'curio_le':

{500: ([0.7016661427224143, 0.054856963219113486, 0.021848475322225714, 0.02059100911662999, 0.008645080163470606, 0.008802263439170073, 0.002514932411191449, 0.005501414649481296, 0.0014146494812951901, 0.0017290160326941214, 0.002672115686890915, 0.00015718327569946557, 0.002672115686890915, 0.0011002829298962591, 0.0007859163784973279, 0.004872681546683433, 0.0014146494812951901, 0.0014146494812951901, 0.0012574662055957245, 0.0011002829298962591, 0.0015718327569946558, 0.009116629990569003, 0.0014146494812951901, 0.0011002829298962591, 0.0014146494812951901, 0.002514932411191449, 0.0006287331027978623, 0.0007859163784973279, 0.0007859163784973279, 0.0007859163784973279, 0.0006287331027978623, 0.0045583149952845015, 0.003300848789688777, 0.00047154982709839675, 0.0020433825840930524, 0.0009430996541967935, 0.007387613957874882, 0.014303678088651368, 0.0017290160326941214, 0.0012574662055957245, 0.011317195850361521, 0.004715498270983967, 0.0022005658597925182, 0.0015718327569946558, 0.0023577491354919836, 0.0014146494812951901, 0.0015718327569946558, 0.00031436655139893113, 0.0007859163784973279, 0.0006287331027978623, 0.0014146494812951901, 0.0007859163784973279, 0.00047154982709839675, 0.009430996541967935, 0.0022005658597925182, 0.0009430996541967935, 0.00675888085507702, 0.0009430996541967935, 0.0009430996541967935, 0.004715498270983967, 0.01603269412134549, 0.0009430996541967935, 0.002514932411191449, 0.0022005658597925182, 0.0015718327569946558, 0.00031436655139893113, 0.0011002829298962591, 0.0007859163784973279, 0.0006287331027978623, 0.0007859163784973279, 0.0009430996541967935, 0.00047154982709839675, 0.0007859163784973279, 0.0028292989625903803, 0.0006287331027978623, 0.00031436655139893113, 0.00015718327569946557, 0.0028292989625903803, 0.00015718327569946557, 0.00031436655139893113, 0.00015718327569946557, 0.00047154982709839675, 0.0006287331027978623, 0.00015718327569946557, 0.0006287331027978623, 0.002672115686890915, 0.0015718327569946558, 0.00015718327569946557, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[[0.40412186379928317, 0.1496415770609319, 0.09901433691756273, 0.06496415770609319, 0.09520609318996416, 0.09274193548387097, 0.0642921146953405, 0.026657706093189962, 0.003360215053763441, 0.0], [0.3467048710601719, 0.1318051575931232, 0.10601719197707736, 0.07163323782234957, 0.12320916905444126, 0.11174785100286533, 0.0659025787965616, 0.022922636103151862, 0.02005730659025788, 0.0], [0.2517985611510791, 0.07913669064748201, 0.04316546762589928, 0.08633093525179857, 0.16546762589928057, 0.2302158273381295, 0.11510791366906475, 0.007194244604316547, 0.02158273381294964, 0.0], [0.26717557251908397, 0.12213740458015267, 0.05343511450381679, 0.11450381679389313, 0.12213740458015267, 0.21374045801526717, 0.10687022900763359, 0.0, 0.0, 0.0], [0.4727272727272727, 0.10909090909090909, 0.12727272727272726, 0.03636363636363636, 0.12727272727272726, 0.09090909090909091, 0.01818181818181818, 0.01818181818181818, 0.0, 0.0], [0.23214285714285715, 0.19642857142857142, 0.03571428571428571, 0.03571428571428571, 0.125, 0.10714285714285714, 0.25, 0.017857142857142856, 0.0, 0.0], [0.3125, 0.0625, 0.25, 0.0625, 0.0, 0.1875, 0.125, 0.0, 0.0, 0.0], [0.4857142857142857, 0.08571428571428572, 0.05714285714285714, 0.02857142857142857, 0.11428571428571428, 0.08571428571428572, 0.05714285714285714, 0.08571428571428572, 0.0, 0.0], [0.2222222222222222, 0.1111111111111111, 0.1111111111111111, 0.0, 0.2222222222222222, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0], [0.36363636363636365, 0.09090909090909091, 0.09090909090909091, 0.2727272727272727, 0.09090909090909091, 0.0, 0.09090909090909091, 0.0, 0.0, 0.0], [0.47058823529411764, 0.17647058823529413, 0.058823529411764705, 0.0, 0.11764705882352941, 0.17647058823529413, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47058823529411764, 0.058823529411764705, 0.0, 0.17647058823529413, 0.11764705882352941, 0.11764705882352941, 0.058823529411764705, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.06451612903225806, 0.0, 0.03225806451612903, 0.0, 0.0, 0.8709677419354839, 0.03225806451612903, 0.0, 0.0, 0.0], [0.1111111111111111, 0.0, 0.1111111111111111, 0.0, 0.0, 0.0, 0.7777777777777778, 0.0, 0.0, 0.0], [0.6666666666666666, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.125, 0.25, 0.125, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0], [0.5714285714285714, 0.0, 0.0, 0.0, 0.0, 0.14285714285714285, 0.2857142857142857, 0.0, 0.0, 0.0], [0.4, 0.3, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], [0.7758620689655172, 0.1206896551724138, 0.10344827586206896, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5555555555555556, 0.3333333333333333, 0.0, 0.0, 0.1111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4444444444444444, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.2222222222222222, 0.0, 0.0, 0.0], [0.0, 0.125, 0.6875, 0.0, 0.0625, 0.0, 0.125, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.2, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8275862068965517, 0.034482758620689655, 0.13793103448275862, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.09523809523809523, 0.8095238095238095, 0.09523809523809523, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6666666666666666, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8461538461538461, 0.15384615384615385, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.16666666666666666, 0.0, 0.0, 0.6666666666666666, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9230769230769231, 0.03296703296703297, 0.0, 0.0, 0.04395604395604396, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36363636363636365, 0.5454545454545454, 0.0, 0.0, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0], [0.875, 0.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19444444444444445, 0.0, 0.013888888888888888, 0.7916666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.5, 0.06666666666666667, 0.0, 0.03333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7142857142857143, 0.14285714285714285, 0.0, 0.07142857142857142, 0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7333333333333333, 0.0, 0.26666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7777777777777778, 0.0, 0.0, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2222222222222222, 0.0, 0.7777777777777778, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6666666666666666, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9767441860465116, 0.0, 0.0, 0.023255813953488372, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8333333333333334, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9666666666666667, 0.03333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8725490196078431, 0.12745098039215685, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6666666666666666, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1875, 0.8125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2857142857142857, 0.0, 0.7142857142857143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
1000: ([0.620247171567107, 0.049483290614337254, 0.02017860349697158, 0.01758281227041941, 0.012978956132760844, 0.013583008179191223, 0.007444533328979805, 0.004914045026366056, 0.01039949063719328, 0.004571204675689354, 0.00749351052193362, 0.009534226895009225, 0.005958891809380765, 0.004407947365843306, 0.005697680113627088, 0.0175664865394348, 0.005779308768550112, 0.0019101105251987658, 0.004701810523566193, 0.0026937456124597977, 0.0020896935660294188, 0.004620181868643169, 0.0020896935660294188, 0.004652833330612378, 0.003591660816613064, 0.0021386707589832335, 0.004016129822212789, 0.0076241163698104585, 0.005403816955904201, 0.009142409351378708, 0.004701810523566193, 0.014317666073498441, 0.005387491224919595, 0.002187647951937048, 0.0030039345011672897, 0.0028406771913212416, 0.0051752567221197325, 0.004685484792581588, 0.0034610549687362252, 0.0017142017533835078, 0.007150670171256918, 0.00648131520088812, 0.0018121561392911367, 0.0030039345011672897, 0.002203973682921653, 0.0031182146180595235, 0.0013550356717222014, 0.0017795046773219272, 0.0013060584787683868, 0.0013550356717222014, 0.001191778361876153, 0.001893784794214161, 0.001208104092860758, 0.004309992979935676, 0.0018284818702757415, 0.0007673093562764273, 0.0023345795307984914, 0.000799960818245637, 0.000897915204153266, 0.0011264754379377336, 0.0023019280688292816, 0.001093823975968524, 0.0007673093562764273, 0.0015182929815682497, 0.0013550356717222014, 0.0007673093562764273, 0.011134148531500498, 0.001191778361876153, 0.000995869590060895, 0.001975413449137185, 0.00047344619855354023, 0.00031018888870749185, 0.0008489380111994515, 0.0006856807013534031, 0.00040814327461512087, 0.0004407947365843306, 0.00024488596476907254, 0.0008326122802148466, 0.0015672701745220643, 0.0003754918126459112, 0.0006203777774149837, 0.0005387491224919595, 0.0005060976605227499, 0.0008489380111994515, 0.0005060976605227499, 0.00035916608166130636, 0.00031018888870749185, 0.00019590877181525804, 0.00016325730984604835, 0.0004244690055997257, 0.00013060584787683867, 1.6325730984604834e-05, 0.00011428011689223384, 1.6325730984604834e-05, 6.530292393841934e-05, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[[0.4042429985260055, 0.15674352495262162, 0.12965887555274794, 0.10196883554432512, 0.07096230785428512, 0.05445883343861865, 0.04403558643925037, 0.027163613392293114, 0.010160033691303432, 0.0006053906085491682], [0.30649950511382384, 0.18277796106895414, 0.1329594193335533, 0.10755526228967338, 0.08116133289343451, 0.06994391290003299, 0.06103596172880237, 0.037611349389640385, 0.020125371164632134, 0.0003299241174529858], [0.24271844660194175, 0.16181229773462782, 0.13754045307443366, 0.17637540453074432, 0.09466019417475728, 0.08090614886731391, 0.05177993527508091, 0.03802588996763754, 0.015372168284789644, 0.0008090614886731392], [0.26369545032497677, 0.21448467966573817, 0.11420612813370473, 0.1392757660167131, 0.08449396471680594, 0.08449396471680594, 0.0584958217270195, 0.035283194057567316, 0.005571030640668524, 0.0], [0.24150943396226415, 0.17358490566037735, 0.21635220125786164, 0.11320754716981132, 0.09056603773584905, 0.0830188679245283, 0.045283018867924525, 0.033962264150943396, 0.0025157232704402514, 0.0], [0.2932692307692308, 0.20432692307692307, 0.15985576923076922, 0.15625, 0.04326923076923077, 0.030048076923076924, 0.037259615384615384, 0.039663461538461536, 0.036057692307692304, 0.0], [0.20175438596491227, 0.05043859649122807, 0.11842105263157894, 0.17543859649122806, 0.125, 0.15789473684210525, 0.06578947368421052, 0.08991228070175439, 0.015350877192982455, 0.0], [0.2591362126245847, 0.132890365448505, 0.16943521594684385, 0.13953488372093023, 0.06976744186046512, 0.059800664451827246, 0.05647840531561462, 0.10299003322259136, 0.009966777408637873, 0.0], [0.41130298273155413, 0.27472527472527475, 0.0847723704866562, 0.07221350078492936, 0.04552590266875981, 0.0141287284144427, 0.03767660910518053, 0.04866562009419152, 0.01098901098901099, 0.0], [0.25, 0.125, 0.09285714285714286, 0.15714285714285714, 0.225, 0.07857142857142857, 0.039285714285714285, 0.03214285714285714, 0.0, 0.0], [0.09586056644880174, 0.19825708061002179, 0.18736383442265794, 0.3202614379084967, 0.06318082788671024, 0.08932461873638345, 0.0196078431372549, 0.026143790849673203, 0.0, 0.0], [0.3082191780821918, 0.3082191780821918, 0.1541095890410959, 0.07534246575342465, 0.06506849315068493, 0.053082191780821915, 0.0273972602739726, 0.008561643835616438, 0.0, 0.0], [0.4438356164383562, 0.07945205479452055, 0.06575342465753424, 0.12876712328767123, 0.09315068493150686, 0.12602739726027398, 0.03561643835616438, 0.0273972602739726, 0.0, 0.0], [0.3333333333333333, 0.09259259259259259, 0.17037037037037037, 0.10740740740740741, 0.05925925925925926, 0.06296296296296296, 0.15925925925925927, 0.014814814814814815, 0.0, 0.0], [0.2148997134670487, 0.09455587392550144, 0.5243553008595988, 0.06017191977077364, 0.04297994269340974, 0.05157593123209169, 0.008595988538681949, 0.0028653295128939827, 0.0, 0.0], [0.04553903345724907, 0.7574349442379182, 0.09293680297397769, 0.046468401486988845, 0.012081784386617101, 0.03345724907063197, 0.011152416356877323, 0.0009293680297397769, 0.0, 0.0], [0.2937853107344633, 0.10451977401129943, 0.1016949152542373, 0.3983050847457627, 0.025423728813559324, 0.011299435028248588, 0.05649717514124294, 0.00847457627118644, 0.0, 0.0], [0.26495726495726496, 0.17094017094017094, 0.1282051282051282, 0.1794871794871795, 0.20512820512820512, 0.008547008547008548, 0.02564102564102564, 0.017094017094017096, 0.0, 0.0], [0.4479166666666667, 0.17708333333333334, 0.125, 0.06597222222222222, 0.06944444444444445, 0.03819444444444445, 0.06944444444444445, 0.006944444444444444, 0.0, 0.0], [0.23030303030303031, 0.06666666666666667, 0.07878787878787878, 0.05454545454545454, 0.048484848484848485, 0.09090909090909091, 0.42424242424242425, 0.006060606060606061, 0.0, 0.0], [0.2421875, 0.0859375, 0.046875, 0.3359375, 0.078125, 0.1640625, 0.0390625, 0.0078125, 0.0, 0.0], [0.23674911660777384, 0.053003533568904596, 0.13074204946996468, 0.04240282685512368, 0.29328621908127206, 0.07420494699646643, 0.1625441696113074, 0.007067137809187279, 0.0, 0.0], [0.359375, 0.1484375, 0.125, 0.109375, 0.109375, 0.0859375, 0.0625, 0.0, 0.0, 0.0], [0.28421052631578947, 0.02456140350877193, 0.3684210526315789, 0.2596491228070175, 0.02456140350877193, 0.028070175438596492, 0.010526315789473684, 0.0, 0.0, 0.0], [0.3, 0.02727272727272727, 0.18181818181818182, 0.38181818181818183, 0.05909090909090909, 0.01818181818181818, 0.031818181818181815, 0.0, 0.0, 0.0], [0.25190839694656486, 0.08396946564885496, 0.2366412213740458, 0.061068702290076333, 0.29770992366412213, 0.04580152671755725, 0.022900763358778626, 0.0, 0.0, 0.0], [0.4105691056910569, 0.10569105691056911, 0.08943089430894309, 0.10569105691056911, 0.22764227642276422, 0.04878048780487805, 0.012195121951219513, 0.0, 0.0, 0.0], [0.42398286937901497, 0.15417558886509636, 0.16274089935760172, 0.12419700214132762, 0.09207708779443255, 0.03640256959314775, 0.006423982869379015, 0.0, 0.0, 0.0], [0.459214501510574, 0.1782477341389728, 0.09365558912386707, 0.11178247734138973, 0.08157099697885196, 0.06948640483383686, 0.006042296072507553, 0.0, 0.0, 0.0], [0.45535714285714285, 0.125, 0.13392857142857142, 0.07321428571428572, 0.2017857142857143, 0.008928571428571428, 0.0017857142857142857, 0.0, 0.0, 0.0], [0.4027777777777778, 0.1388888888888889, 0.06944444444444445, 0.2743055555555556, 0.027777777777777776, 0.04861111111111111, 0.03819444444444445, 0.0, 0.0, 0.0], [0.22234891676168758, 0.09806157354618016, 0.5427594070695553, 0.09578107183580388, 0.037628278221208664, 0.0034207525655644243, 0.0, 0.0, 0.0, 0.0], [0.2727272727272727, 0.20909090909090908, 0.08484848484848485, 0.3787878787878788, 0.03333333333333333, 0.01818181818181818, 0.0030303030303030303, 0.0, 0.0, 0.0], [0.20149253731343283, 0.17164179104477612, 0.13432835820895522, 0.23134328358208955, 0.12686567164179105, 0.13432835820895522, 0.0, 0.0, 0.0, 0.0], [0.2826086956521739, 0.21195652173913043, 0.11413043478260869, 0.2826086956521739, 0.07065217391304347, 0.03804347826086957, 0.0, 0.0, 0.0, 0.0], [0.21839080459770116, 0.08620689655172414, 0.06896551724137931, 0.3735632183908046, 0.15517241379310345, 0.09770114942528736, 0.0, 0.0, 0.0, 0.0], [0.48264984227129337, 0.06624605678233439, 0.056782334384858045, 0.056782334384858045, 0.05362776025236593, 0.28391167192429023, 0.0, 0.0, 0.0, 0.0], [0.5017421602787456, 0.18815331010452963, 0.03832752613240418, 0.07317073170731707, 0.1951219512195122, 0.003484320557491289, 0.0, 0.0, 0.0, 0.0], [0.46226415094339623, 0.27358490566037735, 0.1320754716981132, 0.07075471698113207, 0.05188679245283019, 0.009433962264150943, 0.0, 0.0, 0.0, 0.0], [0.6190476190476191, 0.08571428571428572, 0.08571428571428572, 0.13333333333333333, 0.0380952380952381, 0.0380952380952381, 0.0, 0.0, 0.0, 0.0], [0.0410958904109589, 0.2808219178082192, 0.08904109589041095, 0.2945205479452055, 0.2899543378995434, 0.0045662100456621, 0.0, 0.0, 0.0, 0.0], [0.3425692695214106, 0.08816120906801007, 0.2141057934508816, 0.055415617128463476, 0.2972292191435768, 0.0025188916876574307, 0.0, 0.0, 0.0, 0.0], [0.3153153153153153, 0.10810810810810811, 0.3963963963963964, 0.14414414414414414, 0.02702702702702703, 0.009009009009009009, 0.0, 0.0, 0.0, 0.0], [0.23369565217391305, 0.30978260869565216, 0.16847826086956522, 0.25, 0.03804347826086957, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4888888888888889, 0.17037037037037037, 0.21481481481481482, 0.08888888888888889, 0.037037037037037035, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3403141361256545, 0.1518324607329843, 0.2513089005235602, 0.21465968586387435, 0.041884816753926704, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1927710843373494, 0.37349397590361444, 0.24096385542168675, 0.1566265060240964, 0.03614457831325301, 0.0, 0.0, 0.0, 0.0, 0.0], [0.22935779816513763, 0.28440366972477066, 0.3853211009174312, 0.045871559633027525, 0.05504587155963303, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.275, 0.2, 0.2375, 0.0375, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4819277108433735, 0.03614457831325301, 0.1686746987951807, 0.27710843373493976, 0.03614457831325301, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4657534246575342, 0.1506849315068493, 0.0821917808219178, 0.3013698630136986, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.49137931034482757, 0.13793103448275862, 0.3017241379310345, 0.06896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21621621621621623, 0.08108108108108109, 0.14864864864864866, 0.5540540540540541, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.14015151515151514, 0.14393939393939395, 0.03409090909090909, 0.6818181818181818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5892857142857143, 0.26785714285714285, 0.07142857142857142, 0.0625, 0.008928571428571428, 0.0, 0.0, 0.0, 0.0, 0.0], [0.46808510638297873, 0.2553191489361702, 0.2127659574468085, 0.06382978723404255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6363636363636364, 0.06293706293706294, 0.16783216783216784, 0.13286713286713286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.40816326530612246, 0.4897959183673469, 0.10204081632653061, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8363636363636363, 0.07272727272727272, 0.07272727272727272, 0.01818181818181818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7391304347826086, 0.14492753623188406, 0.08695652173913043, 0.028985507246376812, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8085106382978723, 0.12056737588652482, 0.07092198581560284, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3880597014925373, 0.3880597014925373, 0.22388059701492538, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3829787234042553, 0.5319148936170213, 0.06382978723404255, 0.02127659574468085, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.26881720430107525, 0.3978494623655914, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25301204819277107, 0.6385542168674698, 0.10843373493975904, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8297872340425532, 0.06382978723404255, 0.10638297872340426, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.08211143695014662, 0.9090909090909091, 0.008797653958944282, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3013698630136986, 0.6712328767123288, 0.0273972602739726, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6557377049180327, 0.3442622950819672, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.371900826446281, 0.5867768595041323, 0.04132231404958678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7241379310344828, 0.2413793103448276, 0.034482758620689655, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8421052631578947, 0.10526315789473684, 0.05263157894736842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5192307692307693, 0.4807692307692308, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9285714285714286, 0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7777777777777778, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.6666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9803921568627451, 0.0196078431372549, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9166666666666666, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9130434782608695, 0.08695652173913043, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2894736842105263, 0.7105263157894737, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9393939393939394, 0.06060606060606061, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
1500: ([0.5522341592875188, 0.036935718478066704, 0.029785388011330287, 0.017189374027100825, 0.012795059586087223, 0.01397912573047184, 0.026794600250082935, 0.028187919463087248, 0.009538877689029526, 0.020307755122872383, 0.006236762191543114, 0.007849541939929058, 0.003960497103631306, 0.0038329037691071017, 0.0039094597698216245, 0.008247633143644576, 0.004557633909204583, 0.0037359328348687063, 0.0035675096332967566, 0.00680838033021155, 0.005420164850588206, 0.005302778982825937, 0.004542322709061679, 0.010340163829841528, 0.0028478832265802436, 0.00372572536810677, 0.008186388343072958, 0.006042820323066323, 0.006430704060019905, 0.005272156582540128, 0.0056753515196366145, 0.0146272998698548, 0.007579044070737745, 0.0025416592237221527, 0.003404190165105775, 0.0025263480235792482, 0.005200704315206574, 0.0031387960292954295, 0.0061040651236379416, 0.0036032357669635337, 0.00425651363972746, 0.0045627376425855515, 0.0026488376247224845, 0.0028376757598183072, 0.0032204557633909205, 0.0025365554903411846, 0.0019343149513869396, 0.004618878709776201, 0.0011687549442417127, 0.000944190675479113, 0.0053487125832546506, 0.007859749406690994, 0.00126062214509914, 0.002118049353101794, 0.0009339832087171766, 0.0018424477505295123, 0.006517467527496363, 0.0016229872151478806, 0.0007247301400974813, 0.0013984229463852808, 0.0016229872151478806, 0.0025365554903411846, 0.001434149080052058, 0.00126062214509914, 0.0013218669456707582, 0.0017250618827672442, 0.004144231505346161, 0.0028019496261515297, 0.000944190675479113, 0.0016485058820527216, 0.0006839002730497359, 0.00047975093781100873, 0.0011840661443846174, 0.00047975093781100873, 0.00041340240385842245, 0.0004746472044300406, 0.0006736928062877995, 0.000658381606144895, 0.0008165973409549085, 0.00019904560185775896, 0.00038788373695358155, 0.0003776762701916452, 0.00026539413581034524, 0.0004440248041442315, 0.0005001658713348814, 0.00022966800214356802, 0.00022456426876259984, 0.00010207466761936356, 0.00012248960114323628, 0.00018373440171485443, 7.655600071452267e-05, 0.00010717840100033175, 5.103733380968178e-05, 1.0207466761936357e-05, 8.165973409549086e-05, 1.0207466761936357e-05, 0.0, 0.0, 0.0, 0.0] ,
[[0.3961756714293636, 0.1884900463947062, 0.13647622040258037, 0.10902755956451822, 0.06768821278719432, 0.04726345169220532, 0.02813256686567716, 0.017753830797951977, 0.007550692223803626, 0.0014417478419992238], [0.3632720740638386, 0.1815669476302335, 0.12339367141080558, 0.11151029432085117, 0.08539450048362582, 0.050020726820505734, 0.040210031781124776, 0.025424899820367556, 0.018101423241674727, 0.0011054304269725024], [0.36960246744345443, 0.17032213845099384, 0.15798492117888965, 0.137251542152159, 0.05397532556545579, 0.06151473612063057, 0.023303632625085675, 0.017649074708704592, 0.00822481151473612, 0.00017135023989033586], [0.38271971496437057, 0.22951306413301661, 0.1342042755344418, 0.0917458432304038, 0.05166270783847981, 0.0418646080760095, 0.03147268408551069, 0.024643705463182897, 0.006235154394299287, 0.0059382422802850355], [0.4036697247706422, 0.18627842042281612, 0.172317510969286, 0.07339449541284404, 0.07658556043079377, 0.040686078978859196, 0.028320702034303948, 0.015955325089748704, 0.002792181890706023, 0.0], [0.29061701350857977, 0.2792990142387733, 0.19021540708287696, 0.09748083242059145, 0.044906900328587074, 0.03504928806133625, 0.03066812705366922, 0.018619934282584884, 0.013143483023001095, 0.0], [0.6687619047619048, 0.17447619047619048, 0.05523809523809524, 0.040952380952380955, 0.018285714285714287, 0.019238095238095238, 0.010476190476190476, 0.010285714285714285, 0.002285714285714286, 0.0], [0.7224334600760456, 0.1461162411732754, 0.045446315408292595, 0.044903132355603836, 0.011768966141589716, 0.008690928843020097, 0.00832880680789426, 0.011044722071338041, 0.0012674271229404308, 0.0], [0.3707865168539326, 0.256286784376672, 0.12038523274478331, 0.08239700374531835, 0.08667736757624397, 0.03317281968967362, 0.01819154628143392, 0.027822364901016586, 0.004280363830925628, 0.0], [0.5639607941693893, 0.10731339532545865, 0.14702186479014828, 0.0585574264890676, 0.05529027393817542, 0.03593867805981402, 0.017089721035436038, 0.014073887911535563, 0.0007539582809751194, 0.0], [0.40916530278232405, 0.12847790507364976, 0.14402618657937807, 0.17921440261865793, 0.040098199672667756, 0.046644844517184945, 0.02618657937806874, 0.023731587561374796, 0.0024549918166939444, 0.0], [0.3621586475942783, 0.27828348504551365, 0.1378413524057217, 0.0812743823146944, 0.055266579973992196, 0.046163849154746424, 0.027308192457737322, 0.010403120936280884, 0.0013003901170351106, 0.0], [0.43943298969072164, 0.13530927835051546, 0.0979381443298969, 0.10438144329896908, 0.08762886597938144, 0.08376288659793814, 0.028350515463917526, 0.023195876288659795, 0.0, 0.0], [0.35019973368841545, 0.16511318242343542, 0.19307589880159787, 0.10252996005326231, 0.05725699067909454, 0.045272969374167776, 0.07723035952063914, 0.007989347536617843, 0.0013315579227696406, 0.0], [0.3563968668407311, 0.17362924281984335, 0.26370757180156656, 0.0587467362924282, 0.06266318537859007, 0.037859007832898174, 0.020887728459530026, 0.02349869451697128, 0.0026109660574412533, 0.0], [0.20915841584158415, 0.5631188118811881, 0.08044554455445545, 0.06992574257425743, 0.023514851485148515, 0.03032178217821782, 0.01485148514851485, 0.008663366336633664, 0.0, 0.0], [0.36954087346024633, 0.12989921612541994, 0.1019036954087346, 0.20044792833146696, 0.12206047032474804, 0.019036954087346025, 0.03471444568868981, 0.022396416573348264, 0.0, 0.0], [0.46311475409836067, 0.17759562841530055, 0.08196721311475409, 0.15437158469945356, 0.06420765027322405, 0.01639344262295082, 0.015027322404371584, 0.0273224043715847, 0.0, 0.0], [0.43919885550786836, 0.1473533619456366, 0.12732474964234622, 0.09012875536480687, 0.09728183118741059, 0.05150214592274678, 0.04291845493562232, 0.004291845493562232, 0.0, 0.0], [0.5637181409295352, 0.17841079460269865, 0.06221889055472264, 0.07421289355322339, 0.034482758620689655, 0.023238380809595203, 0.06221889055472264, 0.0014992503748125937, 0.0, 0.0], [0.5216572504708098, 0.1704331450094162, 0.09416195856873823, 0.1224105461393597, 0.01977401129943503, 0.04048964218455744, 0.021657250470809793, 0.009416195856873822, 0.0, 0.0], [0.3474494706448508, 0.21174205967276227, 0.17035611164581327, 0.04523580365736285, 0.1039461020211742, 0.06544754571703561, 0.05101058710298364, 0.004812319538017324, 0.0, 0.0], [0.3921348314606742, 0.16179775280898875, 0.20898876404494382, 0.10449438202247191, 0.05056179775280899, 0.06292134831460675, 0.019101123595505618, 0.0, 0.0, 0.0], [0.6011846001974334, 0.12487660414610069, 0.11599210266535044, 0.07847976307996052, 0.035538005923000986, 0.024679170779861797, 0.0192497532082922, 0.0, 0.0, 0.0], [0.3387096774193548, 0.0913978494623656, 0.1039426523297491, 0.1881720430107527, 0.06810035842293907, 0.18100358422939067, 0.02867383512544803, 0.0, 0.0, 0.0], [0.2876712328767123, 0.3356164383561644, 0.1410958904109589, 0.045205479452054796, 0.1178082191780822, 0.049315068493150684, 0.023287671232876714, 0.0, 0.0, 0.0], [0.529925187032419, 0.12655860349127182, 0.07169576059850374, 0.07668329177057356, 0.15336658354114713, 0.029301745635910224, 0.012468827930174564, 0.0, 0.0, 0.0], [0.4222972972972973, 0.17060810810810811, 0.16047297297297297, 0.12077702702702703, 0.0785472972972973, 0.022804054054054054, 0.024493243243243243, 0.0, 0.0, 0.0], [0.5142857142857142, 0.13412698412698412, 0.0873015873015873, 0.11507936507936507, 0.0746031746031746, 0.05634920634920635, 0.018253968253968255, 0.0, 0.0, 0.0], [0.34172313649564373, 0.10164569215876089, 0.12003872216844143, 0.22265246853823814, 0.1393998063891578, 0.04549854791868345, 0.02904162633107454, 0.0, 0.0, 0.0], [0.5449640287769785, 0.12949640287769784, 0.10251798561151079, 0.1366906474820144, 0.044964028776978415, 0.022482014388489208, 0.018884892086330936, 0.0, 0.0, 0.0], [0.6015352407536636, 0.09525471039776692, 0.23307745987438938, 0.04361479413817167, 0.015352407536636426, 0.010816468946266573, 0.0003489183531053733, 0.0, 0.0, 0.0], [0.4909090909090909, 0.21212121212121213, 0.10909090909090909, 0.13063973063973064, 0.03569023569023569, 0.01616161616161616, 0.0053872053872053875, 0.0, 0.0, 0.0], [0.26305220883534136, 0.20281124497991967, 0.08634538152610442, 0.10843373493975904, 0.0783132530120482, 0.26104417670682734, 0.0, 0.0, 0.0, 0.0], [0.4917541229385307, 0.13493253373313344, 0.1679160419790105, 0.11244377811094453, 0.06146926536731634, 0.022488755622188907, 0.008995502248875561, 0.0, 0.0, 0.0], [0.3373737373737374, 0.08484848484848485, 0.13131313131313133, 0.17373737373737375, 0.15757575757575756, 0.11515151515151516, 0.0, 0.0, 0.0, 0.0], [0.5201177625122669, 0.07850834151128558, 0.18940137389597644, 0.050049067713444556, 0.05789990186457311, 0.10402355250245339, 0.0, 0.0, 0.0, 0.0], [0.3154471544715447, 0.25365853658536586, 0.2016260162601626, 0.05365853658536585, 0.15772357723577235, 0.01788617886178862, 0.0, 0.0, 0.0, 0.0], [0.6078595317725752, 0.22742474916387959, 0.06354515050167224, 0.03260869565217391, 0.0451505016722408, 0.023411371237458192, 0.0, 0.0, 0.0, 0.0], [0.49008498583569404, 0.2790368271954674, 0.09206798866855524, 0.09490084985835694, 0.0339943342776204, 0.009915014164305949, 0.0, 0.0, 0.0, 0.0], [0.19184652278177458, 0.23860911270983212, 0.12709832134292565, 0.2182254196642686, 0.22062350119904076, 0.0035971223021582736, 0.0, 0.0, 0.0, 0.0], [0.25615212527964204, 0.17002237136465326, 0.22930648769574943, 0.14988814317673377, 0.18680089485458612, 0.007829977628635347, 0.0, 0.0, 0.0, 0.0], [0.2658959537572254, 0.26396917148362237, 0.17533718689788053, 0.2658959537572254, 0.017341040462427744, 0.011560693641618497, 0.0, 0.0, 0.0, 0.0], [0.43345323741007197, 0.1474820143884892, 0.10251798561151079, 0.10431654676258993, 0.210431654676259, 0.0017985611510791368, 0.0, 0.0, 0.0, 0.0], [0.4960380348652932, 0.07923930269413629, 0.08399366085578447, 0.312202852614897, 0.01901743264659271, 0.009508716323296355, 0.0, 0.0, 0.0, 0.0], [0.3963782696177062, 0.1267605633802817, 0.2112676056338028, 0.14084507042253522, 0.1227364185110664, 0.002012072434607646, 0.0, 0.0, 0.0, 0.0], [0.1424802110817942, 0.12401055408970976, 0.37994722955145116, 0.31926121372031663, 0.03430079155672823, 0.0, 0.0, 0.0, 0.0, 0.0], [0.10386740331491713, 0.6552486187845303, 0.16353591160220995, 0.06298342541436464, 0.014364640883977901, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1572052401746725, 0.33624454148471616, 0.14410480349344978, 0.32751091703056767, 0.034934497816593885, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4540540540540541, 0.08108108108108109, 0.1891891891891892, 0.1945945945945946, 0.08108108108108109, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2824427480916031, 0.5820610687022901, 0.04580152671755725, 0.08301526717557252, 0.006679389312977099, 0.0, 0.0, 0.0, 0.0, 0.0], [0.052597402597402594, 0.775974025974026, 0.09025974025974026, 0.07077922077922078, 0.01038961038961039, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3967611336032389, 0.24696356275303644, 0.16194331983805668, 0.1902834008097166, 0.004048582995951417, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25783132530120484, 0.22409638554216868, 0.04578313253012048, 0.4674698795180723, 0.004819277108433735, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4918032786885246, 0.2677595628415301, 0.12021857923497267, 0.11475409836065574, 0.00546448087431694, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5263157894736842, 0.15789473684210525, 0.11080332409972299, 0.20498614958448755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5489428347689899, 0.37196554424432265, 0.05090054815974941, 0.028191072826938137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5283018867924528, 0.2578616352201258, 0.12264150943396226, 0.09119496855345911, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6126760563380281, 0.2112676056338028, 0.09859154929577464, 0.07746478873239436, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3905109489051095, 0.4635036496350365, 0.08394160583941605, 0.06204379562043796, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5723270440251572, 0.24842767295597484, 0.09433962264150944, 0.08490566037735849, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.34004024144869216, 0.20724346076458752, 0.4124748490945674, 0.04024144869215292, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4306049822064057, 0.3096085409252669, 0.1601423487544484, 0.099644128113879, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5546558704453441, 0.2388663967611336, 0.19838056680161945, 0.008097165991902834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.416988416988417, 0.35135135135135137, 0.22393822393822393, 0.007722007722007722, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5680473372781065, 0.34615384615384615, 0.08579881656804733, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.14655172413793102, 0.8165024630541872, 0.03694581280788178, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7941712204007286, 0.17122040072859745, 0.03460837887067395, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.572972972972973, 0.3027027027027027, 0.12432432432432433, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5634674922600619, 0.38390092879256965, 0.05263157894736842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5373134328358209, 0.2462686567164179, 0.21641791044776118, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6063829787234043, 0.30851063829787234, 0.0851063829787234, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7456896551724138, 0.2543103448275862, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7659574468085106, 0.20212765957446807, 0.031914893617021274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7654320987654321, 0.2345679012345679, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8494623655913979, 0.13978494623655913, 0.010752688172043012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6893939393939394, 0.3106060606060606, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9689922480620154, 0.031007751937984496, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.81875, 0.18125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9487179487179487, 0.05128205128205128, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5394736842105263, 0.4605263157894737, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8243243243243243, 0.17567567567567569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9615384615384616, 0.038461538461538464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9489795918367347, 0.05102040816326531, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9555555555555556, 0.044444444444444446, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
2000: ([0.5248436473227437, 0.032842559651945345, 0.02388910289818948, 0.017295098683465137, 0.013830980489905055, 0.012106002583218147, 0.02277310733951191, 0.019578074369490835, 0.009454805012349595, 0.01603181437085042, 0.007202986562733679, 0.008120708798803562, 0.006367406131744125, 0.00626826947044028, 0.00702737304842401, 0.010497156194058598, 0.006574176882463575, 0.007013210668237747, 0.00810937889465455, 0.009593596338174979, 0.007211483990845437, 0.006460877840973466, 0.007123677233690603, 0.008542747728354219, 0.004999320205751059, 0.0054610137998232534, 0.011672633749518478, 0.006783780109220276, 0.005495003512270286, 0.006823434773741814, 0.0071293421857651085, 0.01011477192902948, 0.008619224581360042, 0.003192200493983821, 0.006290929278738302, 0.006511862409644014, 0.004483809566971063, 0.004067435589494913, 0.010542475810654641, 0.004741564886361061, 0.0035859146631619498, 0.004410165190002493, 0.0027928213727311866, 0.006038838911422809, 0.0025860506220117376, 0.002758831660284154, 0.003880492171036233, 0.00485486392785117, 0.00220366635698262, 0.0018496068523260294, 0.0037077111327638167, 0.006390065940042147, 0.0015720242006752622, 0.0024557567242981123, 0.0021186920758650384, 0.0025350660533411886, 0.00450080442319458, 0.001699485622351635, 0.0010111939452992229, 0.0022178287371688835, 0.001492714871632186, 0.0022064988330198726, 0.001277446692800979, 0.0013652534499558134, 0.0015465319163399877, 0.0017023180983888877, 0.003141215925313272, 0.002699349663501847, 0.0014842174435204278, 0.0014728875393714169, 0.0008100881466542793, 0.0013850807822165824, 0.0013114364052480116, 0.0008270830028777957, 0.0004928508304819741, 0.0004900183544447213, 0.0006684643447916431, 0.0006089823480093359, 0.0006627993927171376, 0.00031157236409779974, 0.001031021277559992, 0.0005608302553760396, 0.00028891255579977795, 0.0003653894088056015, 0.00040504407332713963, 0.0006684643447916431, 0.0002379279871292289, 0.0006514694885681267, 9.630418526659264e-05, 0.00016428361016065804, 0.00014445627789988897, 0.00015012122997439442, 0.00010763408941560355, 0.00015295370601164715, 5.3817044707801775e-05, 4.815209263329632e-05, 5.948199678230722e-05, 0.0, 0.0, 0.0] ,
[[0.3044280741520278, 0.17695026849078496, 0.16322081005963462, 0.14042472813621523, 0.09609541541865674, 0.05688226881459295, 0.03341158692895113, 0.017836423001160312, 0.009012655495291292, 0.001737769502684908], [0.2944372574385511, 0.18180250107805088, 0.15929279862009488, 0.13445450625269512, 0.09279862009486847, 0.04941785252263907, 0.03648124191461837, 0.02630444156964209, 0.02285467874083657, 0.0021561017680034496], [0.2895423286696704, 0.19373962532606118, 0.17453165757647618, 0.1588807208916291, 0.06805786103865306, 0.05371116907754328, 0.027507706900640267, 0.025254920559639554, 0.008536874555371117, 0.00023713540431586437], [0.28922371437929906, 0.23157549950867998, 0.1912872584343269, 0.10301342941369145, 0.05977726826072716, 0.041107107762856206, 0.03930560104814936, 0.034719947592531934, 0.006223386832623649, 0.0037667867671143137], [0.36084374360024574, 0.1775547818963752, 0.1736637313127176, 0.11386442760597994, 0.07659225885725988, 0.03972967438050379, 0.0358386237968462, 0.01679295515052222, 0.005119803399549457, 0.0], [0.2767898923724848, 0.2477772578380908, 0.18671034160037436, 0.10645765091249415, 0.06855404773046327, 0.03532990173139916, 0.033926064576509124, 0.026204960224613945, 0.018249883013570427, 0.0], [0.48308457711442787, 0.17114427860696518, 0.11579601990049751, 0.11890547263681592, 0.04664179104477612, 0.03072139303482587, 0.015298507462686567, 0.012562189054726369, 0.005845771144278607, 0.0], [0.6187789351851852, 0.16174768518518517, 0.06901041666666667, 0.07465277777777778, 0.026475694444444444, 0.019675925925925927, 0.01461226851851852, 0.012297453703703705, 0.0027488425925925927, 0.0], [0.37177950868783705, 0.21420011983223486, 0.14619532654284004, 0.09257040143798682, 0.07938885560215699, 0.03505092869982025, 0.0329538645895746, 0.025464349910125823, 0.002396644697423607, 0.0], [0.4742049469964664, 0.12243816254416962, 0.16395759717314487, 0.07632508833922262, 0.0666077738515901, 0.03657243816254417, 0.027738515901060072, 0.0196113074204947, 0.01254416961130742, 0.0], [0.2874557609123083, 0.15650806134486828, 0.16515926071569012, 0.19937082186394023, 0.048368069209594966, 0.06449075894612662, 0.03814392449862367, 0.030279197797876523, 0.010224144710971293, 0.0], [0.2807813044994768, 0.24276246948029298, 0.15660969654691315, 0.11126613184513429, 0.09103592605510986, 0.05266829438437391, 0.024066968957098013, 0.0355772584583188, 0.005231949773282177, 0.0], [0.3429715302491103, 0.17126334519572953, 0.16637010676156583, 0.13745551601423486, 0.0947508896797153, 0.04448398576512456, 0.02402135231316726, 0.018683274021352312, 0.0, 0.0], [0.254405784003615, 0.22729326705829192, 0.25259828287392677, 0.09986443741527339, 0.051965657478535925, 0.05061003163126977, 0.04609127880704925, 0.012200632625395391, 0.004970628106642567, 0.0], [0.2361950826279726, 0.25957275292220877, 0.2543329302700524, 0.12293430068520758, 0.04877065699314793, 0.0354695687222894, 0.026199113260781944, 0.014510278113663845, 0.002015316404675534, 0.0], [0.36859147328656233, 0.3248785752833243, 0.11683756071235833, 0.0744738262277388, 0.03291958985429034, 0.04749055585536967, 0.024015110631408525, 0.010523475445223961, 0.0002698327037236913, 0.0], [0.32399827660491165, 0.18828091339939682, 0.1934510986643688, 0.14691943127962084, 0.08013787160706592, 0.023265833692373977, 0.025420077552778975, 0.01852649719948298, 0.0, 0.0], [0.3501615508885299, 0.14176090468497576, 0.2701938610662359, 0.0981421647819063, 0.04361873990306947, 0.049273021001615507, 0.026252019386106624, 0.02059773828756058, 0.0, 0.0], [0.318546978693678, 0.2315752706950751, 0.2553265805099546, 0.08278030038421237, 0.06147397834439399, 0.028641285365001747, 0.018861334264757248, 0.0027942717429269995, 0.0, 0.0], [0.5231768526719811, 0.22763507528786536, 0.08591674047829938, 0.07056392087392974, 0.030410392677886033, 0.020962503690581637, 0.03690581635665781, 0.0044286979627989375, 0.0, 0.0], [0.4662215239591516, 0.17871170463472114, 0.11468970934799685, 0.08523173605655932, 0.03652788688138256, 0.05420267085624509, 0.05498821681068342, 0.009426551453260016, 0.0, 0.0], [0.24024550635686104, 0.23805348531345902, 0.18719859710653222, 0.055677334502411226, 0.17360806663743972, 0.05918456817185445, 0.03857957036387549, 0.007452871547566856, 0.0, 0.0], [0.32445328031809145, 0.258051689860835, 0.22584493041749504, 0.07833001988071571, 0.04294234592445328, 0.04850894632206759, 0.018290258449304174, 0.0035785288270377734, 0.0, 0.0], [0.4986737400530504, 0.16047745358090185, 0.1289787798408488, 0.08885941644562334, 0.06465517241379311, 0.03149867374005305, 0.020225464190981434, 0.006631299734748011, 0.0, 0.0], [0.3059490084985836, 0.24475920679886687, 0.11444759206798867, 0.09971671388101983, 0.04475920679886686, 0.14390934844192635, 0.0339943342776204, 0.012464589235127478, 0.0, 0.0], [0.4118257261410788, 0.2474066390041494, 0.10840248962655602, 0.09906639004149377, 0.0700207468879668, 0.038381742738589214, 0.02437759336099585, 0.0005186721991701245, 0.0, 0.0], [0.5192914341179325, 0.19388497937393837, 0.09439456442611016, 0.07231254549866538, 0.08565882067459354, 0.026449890803203105, 0.008007765105556904, 0.0, 0.0, 0.0], [0.4254697286012526, 0.20167014613778705, 0.15615866388308977, 0.0709812108559499, 0.056784968684759914, 0.0651356993736952, 0.023799582463465554, 0.0, 0.0, 0.0], [0.45824742268041235, 0.13659793814432988, 0.0865979381443299, 0.11185567010309279, 0.08195876288659794, 0.08969072164948454, 0.03505154639175258, 0.0, 0.0, 0.0], [0.43129929431299296, 0.1087588210875882, 0.1523453715234537, 0.16770444167704443, 0.07513491075134911, 0.04234122042341221, 0.0224159402241594, 0.0, 0.0, 0.0], [0.4600715137067938, 0.21771950735001985, 0.11402463249900675, 0.08820023837902265, 0.04648390941597139, 0.03853794199443782, 0.03496225665474772, 0.0, 0.0, 0.0], [0.56482777933352, 0.10865303836460376, 0.20498459815177822, 0.053486418370204424, 0.03668440212825539, 0.024642957154858584, 0.006720806496779613, 0.0, 0.0, 0.0], [0.42195202103187646, 0.27571475517581334, 0.06638186000657247, 0.08971409792967466, 0.09168583634571147, 0.040749260598093986, 0.013802168912257641, 0.0, 0.0, 0.0], [0.30168589174800353, 0.21384205856255545, 0.08251996450754215, 0.17125110913930788, 0.06299911268855368, 0.15527950310559005, 0.012422360248447204, 0.0, 0.0, 0.0], [0.6042323277802791, 0.13462404322377308, 0.11166141377757767, 0.06348491670418731, 0.042323277802791534, 0.03872129671319226, 0.00495272399819901, 0.0, 0.0, 0.0], [0.40843845150065244, 0.13266637668551545, 0.09525880817746847, 0.08699434536755112, 0.19791213571117877, 0.07872988255763376, 0.0, 0.0, 0.0, 0.0], [0.4649399873657612, 0.07833228048010107, 0.15540113708149084, 0.13644977890082122, 0.07959570435881239, 0.08401768793430196, 0.0012634238787113076, 0.0, 0.0, 0.0], [0.28133704735376047, 0.18245125348189414, 0.2903899721448468, 0.04387186629526462, 0.12534818941504178, 0.0766016713091922, 0.0, 0.0, 0.0, 0.0], [0.2775389575497045, 0.14535196131112305, 0.06179473401397098, 0.18027941966684577, 0.3167651800107469, 0.018269747447608814, 0.0, 0.0, 0.0, 0.0], [0.4217443249701314, 0.1917562724014337, 0.1816009557945042, 0.06750298685782556, 0.12485065710872162, 0.012544802867383513, 0.0, 0.0, 0.0, 0.0], [0.23459715639810427, 0.20695102685624012, 0.12243285939968404, 0.17772511848341233, 0.21642969984202212, 0.04186413902053712, 0.0, 0.0, 0.0, 0.0], [0.3230571612074502, 0.2556197816313423, 0.15350032113037893, 0.12010276172125883, 0.1252408477842004, 0.0224791265253693, 0.0, 0.0, 0.0, 0.0], [0.22920892494929007, 0.18052738336713997, 0.1206896551724138, 0.21602434077079108, 0.2363083164300203, 0.017241379310344827, 0.0, 0.0, 0.0, 0.0], [0.37476547842401503, 0.099906191369606, 0.35412757973733583, 0.053939962476547844, 0.1125703564727955, 0.004690431519699813, 0.0, 0.0, 0.0, 0.0], [0.4819277108433735, 0.10405257393209201, 0.09200438116100766, 0.2497261774370208, 0.056955093099671415, 0.01533406352683461, 0.0, 0.0, 0.0, 0.0], [0.5061601642710473, 0.13655030800821355, 0.13039014373716631, 0.11293634496919917, 0.11293634496919917, 0.001026694045174538, 0.0, 0.0, 0.0, 0.0], [0.2759124087591241, 0.0708029197080292, 0.25547445255474455, 0.23357664233576642, 0.16423357664233576, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2969661610268378, 0.4352392065344224, 0.14294049008168028, 0.07526254375729288, 0.049591598599766626, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3444730077120823, 0.21079691516709512, 0.21465295629820053, 0.17352185089974292, 0.056555269922879174, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44104134762633995, 0.21745788667687596, 0.20367534456355282, 0.10872894333843798, 0.02909647779479326, 0.0, 0.0, 0.0, 0.0, 0.0], [0.26585179526355995, 0.5553857906799083, 0.05653170359052712, 0.11077158135981666, 0.01145912910618793, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1972517730496454, 0.5660460992907801, 0.08687943262411348, 0.13652482269503546, 0.013297872340425532, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2756756756756757, 0.42342342342342343, 0.13513513513513514, 0.15495495495495495, 0.010810810810810811, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23990772779700115, 0.33679354094579006, 0.1245674740484429, 0.28027681660899656, 0.01845444059976932, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3195187165775401, 0.42914438502673796, 0.12967914438502673, 0.11764705882352941, 0.004010695187165776, 0.0, 0.0, 0.0, 0.0, 0.0], [0.32849162011173183, 0.1329608938547486, 0.4223463687150838, 0.11173184357541899, 0.004469273743016759, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5047199496538703, 0.34675896790434235, 0.11390811831340465, 0.034612964128382634, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42, 0.21333333333333335, 0.20333333333333334, 0.16333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4369747899159664, 0.30532212885154064, 0.15966386554621848, 0.09803921568627451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5644955300127714, 0.2669220945083014, 0.11749680715197956, 0.05108556832694764, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4781783681214421, 0.23719165085388993, 0.1555977229601518, 0.12903225806451613, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27471116816431324, 0.3080872913992298, 0.3594351732991014, 0.057766367137355584, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.37250554323725055, 0.28824833702882485, 0.2660753880266075, 0.07317073170731707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.487551867219917, 0.22406639004149378, 0.24066390041493776, 0.04771784232365145, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2967032967032967, 0.24725274725274726, 0.43772893772893773, 0.018315018315018316, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5174708818635607, 0.3560732113144759, 0.12312811980033278, 0.0033277870216306157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18394950405770966, 0.6988277727682597, 0.11722272317403065, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6988457502623295, 0.2518363064008394, 0.04931794333683106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5076335877862596, 0.3148854961832061, 0.17748091603053434, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.49230769230769234, 0.40576923076923077, 0.10192307692307692, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6538461538461539, 0.15734265734265734, 0.1888111888111888, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4703476482617587, 0.45807770961145194, 0.07157464212678936, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7429805615550756, 0.23758099352051837, 0.019438444924406047, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6883561643835616, 0.2568493150684932, 0.0547945205479452, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6896551724137931, 0.27586206896551724, 0.034482758620689655, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7109826589595376, 0.2658959537572254, 0.023121387283236993, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4745762711864407, 0.5211864406779662, 0.00423728813559322, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.772093023255814, 0.22790697674418606, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7606837606837606, 0.23931623931623933, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8363636363636363, 0.16363636363636364, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6236263736263736, 0.37637362637362637, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8484848484848485, 0.15151515151515152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8921568627450981, 0.10784313725490197, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8604651162790697, 0.13953488372093023, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9370629370629371, 0.06293706293706294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9703389830508474, 0.029661016949152543, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9880952380952381, 0.011904761904761904, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
3000: ([0.4272652488773212, 0.024265707388777258, 0.01901642549863121, 0.015159535015441048, 0.012954634337113805, 0.011021131983871186, 0.016636211616522595, 0.015105592491200626, 0.011708899167936564, 0.014584709991504052, 0.007277183660809408, 0.008045864631235418, 0.007268755141396841, 0.00896288754332259, 0.007853694388628915, 0.009107858077218723, 0.007875608539101587, 0.009707968659393416, 0.010164794411554488, 0.008907259315199655, 0.006151133467290602, 0.007307526330694645, 0.006702358636872413, 0.00878757433954122, 0.006786643830998072, 0.007162555796798511, 0.012383180720941836, 0.008295348805847369, 0.007723895189675401, 0.008108235674888406, 0.008743746038595877, 0.010830647445147196, 0.008150378271951236, 0.005313338637681551, 0.00896288754332259, 0.010439564144404138, 0.006230361549768721, 0.005957277520801586, 0.008713403368710638, 0.00802563618464526, 0.00822117783501679, 0.006230361549768721, 0.005808935579140426, 0.007405297155880409, 0.010458106887111782, 0.009146629266516527, 0.007565439024719162, 0.010392364435693769, 0.004507572181840249, 0.004396315725594379, 0.00501665475435923, 0.008630803878467493, 0.005072282982482165, 0.004128288808274783, 0.0036326918668159076, 0.005193653662023115, 0.004795827545750003, 0.0030308955807587016, 0.0026785834693134466, 0.0037995765511847127, 0.0037506911385918303, 0.0037608053618869096, 0.0026229552411905113, 0.0025774412363626558, 0.002813439779914501, 0.002995495799225925, 0.004263145118875837, 0.0032213801194826914, 0.0033444365029061535, 0.0026701549499008807, 0.0021543295618518467, 0.0035416638571601957, 0.0028842393429800548, 0.0011968497565843594, 0.0030089814302860303, 0.0010232222566855017, 0.0017784175960514073, 0.0017548177416962228, 0.002075101479373727, 0.0009675940285625666, 0.0016536755087454317, 0.0017227893679284722, 0.0013249632516553612, 0.0014395911156662577, 0.0007703666743085242, 0.001296306285652637, 0.0017025609213383141, 0.0020481302172535164, 0.0003995118201556242, 0.00033714077650263643, 0.00018374172319393685, 0.0002747697328496487, 0.00019722735425404232, 0.00019385594648901595, 0.00011125645624587003, 6.911385918304047e-05, 6.57424514180141e-05, 2.6971262120210916e-05, 0.0, 0.0] ,
[[0.23590726888236593, 0.14775273806142095, 0.14634425401634946, 0.1454249913202664, 0.12290897326642047, 0.0986333364896001, 0.05692327115487801, 0.028461635577439005, 0.013994097781144462, 0.003649433450115204], [0.2551580409864536, 0.15811045501910387, 0.14685654741229592, 0.144633553317124, 0.11427579020493227, 0.06425842306356373, 0.04432094477249045, 0.03098298020145884, 0.03556790552275096, 0.005835359499826329], [0.23021008775817747, 0.15530538072865882, 0.1527346866412552, 0.1709954791241911, 0.10734863930502615, 0.08155305380728659, 0.04325857636734332, 0.035103270986614664, 0.02171793280737523, 0.0017728924740714476], [0.2196152563104637, 0.19882130545980206, 0.1754698098521072, 0.13343711775825642, 0.09118203046814188, 0.06360502613143557, 0.04436784165462026, 0.043144668075169576, 0.027354609140442566, 0.0030023351495607697], [0.25256994144437217, 0.13819128171763176, 0.16955107351984386, 0.13012361743656473, 0.1342875731945348, 0.059726740403383216, 0.04202992843201041, 0.019388418998048145, 0.052309694209499026, 0.0018217306441119063], [0.19516671765065768, 0.1983787090853472, 0.1644233710614867, 0.12144386662587947, 0.1119608442948914, 0.06974609972468644, 0.069593147751606, 0.03456714591618232, 0.0333435301315387, 0.0013765677577240747], [0.40247238828655385, 0.16384638767858953, 0.11612118755699666, 0.13354949842942548, 0.08795217347248961, 0.042861485459519705, 0.021785388590536022, 0.01793494781639477, 0.013476542709494376, 0.0], [0.49179779042517574, 0.14418033701595803, 0.0899453186028345, 0.11605847561656066, 0.06026113157013726, 0.046200200870438565, 0.02254212699475505, 0.018078339471041177, 0.010936279433098984, 0.0], [0.21336020731356176, 0.16167578462424417, 0.16282752663403396, 0.16815433342931183, 0.11359055571551972, 0.07443132738266628, 0.036855744313273826, 0.034408292542470485, 0.03469622804491794, 0.0], [0.3287101248266297, 0.1183541377716135, 0.15661118816458622, 0.130374479889043, 0.12968099861303745, 0.05605640314378178, 0.0345584835876098, 0.01872399445214979, 0.026930189551548776, 0.0], [0.20639332870048646, 0.12230715774843641, 0.13666898309010886, 0.1899467222608293, 0.10794533240676396, 0.12138058837155433, 0.0433171183692379, 0.04146397961547371, 0.030576789437109102, 0.0], [0.1881416300020951, 0.1655143515608632, 0.12968782736224596, 0.10601298973391997, 0.11062225015713388, 0.0680913471611146, 0.04211187932118165, 0.18185627487953068, 0.007961449821914938, 0.0], [0.23794063079777367, 0.1913265306122449, 0.18390538033395176, 0.1461038961038961, 0.11201298701298701, 0.055658627087198514, 0.03316326530612245, 0.03130797773654916, 0.008580705009276438, 0.0], [0.16513071280797442, 0.1781079556140681, 0.2294526988903517, 0.14594696257287945, 0.08971224374647357, 0.1098363738950536, 0.040812488245251084, 0.0372390445740079, 0.003761519653940192, 0.0], [0.20090148100450742, 0.1845889675896115, 0.20798454603992272, 0.18308649924876583, 0.0884309937754883, 0.0489375402446877, 0.0326250268297918, 0.04614724189740287, 0.00729770336982185, 0.0], [0.2739218952433833, 0.26947991856376086, 0.14029243013140846, 0.12826207662409772, 0.06662965019433648, 0.05682028502683694, 0.03849713122339441, 0.024430871737923375, 0.001665741254858412, 0.0], [0.22538527397260275, 0.14554794520547945, 0.2399400684931507, 0.1701626712328767, 0.12778253424657535, 0.04323630136986301, 0.027183219178082193, 0.020333904109589043, 0.0004280821917808219, 0.0], [0.2048966834519882, 0.10557388435492272, 0.26254558083000523, 0.11894426115645078, 0.08838339989581524, 0.0557388435492273, 0.14186490710192742, 0.02170515714533773, 0.0003472825143254037, 0.0], [0.309452736318408, 0.2162520729684909, 0.20315091210613598, 0.10016583747927031, 0.0625207296849088, 0.05107794361525705, 0.0406301824212272, 0.016749585406301823, 0.0, 0.0], [0.36241483724451173, 0.22369417108251324, 0.13758516275548827, 0.10957607872823619, 0.07759273277819834, 0.03349735049205148, 0.0463663890991673, 0.00927327781983346, 0.0, 0.0], [0.3801041381200329, 0.16826527815839956, 0.15620718004932857, 0.10578240613866813, 0.056179775280898875, 0.059468347492463686, 0.05398739380652234, 0.020005480953685942, 0.0, 0.0], [0.20576701268742792, 0.23967704728950404, 0.18777393310265283, 0.11418685121107267, 0.12687427912341406, 0.0698961937716263, 0.0405997693194925, 0.01522491349480969, 0.0, 0.0], [0.28269617706237427, 0.2550301810865191, 0.2238430583501006, 0.11544265593561369, 0.04904426559356137, 0.03948692152917505, 0.02590543259557344, 0.008551307847082495, 0.0, 0.0], [0.40399002493765584, 0.15557260694417802, 0.17648187224247075, 0.12679838864377518, 0.06752349894494533, 0.03644734318051026, 0.024170343372338386, 0.009015921734126223, 0.0, 0.0], [0.1830601092896175, 0.146795827123696, 0.11773472429210134, 0.20218579234972678, 0.16815697963238946, 0.11276701440635867, 0.06135121708892201, 0.007948335817188276, 0.0, 0.0], [0.27347611202635913, 0.24499882325253, 0.20851965168274889, 0.13650270651918098, 0.06142621793363144, 0.032007531183807954, 0.039538714991762765, 0.0035302424099788185, 0.0, 0.0], [0.453444051184318, 0.21344949632453036, 0.12020147018785733, 0.07745711952082766, 0.08426354478627825, 0.02858698611489246, 0.022597331881295944, 0.0, 0.0, 0.0], [0.32960780329201383, 0.22495427758585654, 0.21946758788864051, 0.08412924202397887, 0.051412314570209304, 0.051412314570209304, 0.038203617150985573, 0.000812842918106076, 0.0, 0.0], [0.3009602793539939, 0.21147970318638148, 0.15124399825403753, 0.14273243125272805, 0.07005674378000873, 0.05499781754692274, 0.06852902662592754, 0.0, 0.0, 0.0], [0.28024948024948027, 0.188981288981289, 0.19854469854469856, 0.14241164241164242, 0.06943866943866944, 0.05405405405405406, 0.06632016632016632, 0.0, 0.0, 0.0], [0.28455754771544245, 0.2533256217466744, 0.24696356275303644, 0.11239637555427029, 0.039714671293618664, 0.03450935029882398, 0.028532870638133798, 0.0, 0.0, 0.0], [0.43190661478599224, 0.14957198443579767, 0.2034241245136187, 0.12280155642023347, 0.03610894941634241, 0.04793774319066148, 0.008249027237354085, 0.0, 0.0, 0.0], [0.35243019648397106, 0.26659772492244055, 0.1640124095139607, 0.08996897621509824, 0.07466390899689762, 0.03991726990692865, 0.012409513960703205, 0.0, 0.0, 0.0], [0.22461928934010153, 0.21605329949238578, 0.2569796954314721, 0.1383248730964467, 0.05171319796954315, 0.10374365482233502, 0.008565989847715736, 0.0, 0.0, 0.0], [0.4293774684972729, 0.27985706225315027, 0.15215347000188076, 0.06940003761519654, 0.03216099304118864, 0.03272522098927967, 0.0043257476020312205, 0.0, 0.0, 0.0], [0.22331664782819313, 0.2657839496205393, 0.14677862102373648, 0.20054900694332312, 0.10447279186177943, 0.0582916195704828, 0.0008073631519457452, 0.0, 0.0, 0.0], [0.3522727272727273, 0.20833333333333334, 0.16883116883116883, 0.11228354978354978, 0.04816017316017316, 0.10795454545454546, 0.0021645021645021645, 0.0, 0.0, 0.0], [0.2903225806451613, 0.24589700056593095, 0.19666100735710243, 0.08149405772495756, 0.1366723259762309, 0.04867006225240521, 0.0002829654782116582, 0.0, 0.0, 0.0], [0.2656219771715999, 0.16347455987618495, 0.12516927839040434, 0.15534919713677695, 0.2629135229251306, 0.02747146449990327, 0.0, 0.0, 0.0, 0.0], [0.38395295106070154, 0.21319050619617727, 0.11657214870825457, 0.2127704263810124, 0.06196177273682, 0.011552194917034237, 0.0, 0.0, 0.0, 0.0], [0.21242567151937666, 0.18494976419930284, 0.07094525322944432, 0.2587656346114415, 0.2503588271478368, 0.02255484929259791, 0.0, 0.0, 0.0, 0.0], [0.47294372294372294, 0.24269480519480519, 0.1185064935064935, 0.06872294372294373, 0.08062770562770563, 0.016504329004329004, 0.0, 0.0, 0.0, 0.0], [0.3580963435867673, 0.2240278583865351, 0.144805571677307, 0.14190365641323274, 0.11491584445734185, 0.01625072547881602, 0.0, 0.0, 0.0, 0.0], [0.35602094240837695, 0.2624630093330298, 0.2255861597996813, 0.06646938310949238, 0.08331436376052812, 0.006146141588891418, 0.0, 0.0, 0.0, 0.0], [0.23646034816247583, 0.23210831721470018, 0.07172791747259832, 0.38781431334622823, 0.06366860090264345, 0.008220502901353965, 0.0, 0.0, 0.0, 0.0], [0.3016955399926281, 0.13509030593438998, 0.05952819756726871, 0.42480648728345005, 0.07500921489126429, 0.003870254330998894, 0.0, 0.0, 0.0, 0.0], [0.24732620320855614, 0.08890374331550802, 0.13279857397504458, 0.4416221033868093, 0.08890374331550802, 0.000445632798573975, 0.0, 0.0, 0.0, 0.0], [0.23860502838605027, 0.24817518248175183, 0.23828061638280618, 0.1991889699918897, 0.07575020275750202, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3765893792071803, 0.2599102468212416, 0.1637995512341062, 0.14360508601346297, 0.05609573672400898, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3546779141104294, 0.27569018404907975, 0.1851993865030675, 0.1138803680981595, 0.0705521472392638, 0.0, 0.0, 0.0, 0.0, 0.0], [0.322244623655914, 0.4022177419354839, 0.14112903225806453, 0.09206989247311828, 0.04233870967741935, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3408203125, 0.4291015625, 0.092578125, 0.1208984375, 0.0166015625, 0.0, 0.0, 0.0, 0.0, 0.0], [0.353938185443669, 0.19508142239946827, 0.3044200731139914, 0.128946493851778, 0.017613825191093387, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2911392405063291, 0.3078807676602695, 0.17803184973458555, 0.16700694160881993, 0.055941200489995915, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42088167053364267, 0.2900232018561485, 0.16102088167053363, 0.1136890951276102, 0.014385150812064965, 0.0, 0.0, 0.0, 0.0, 0.0], [0.35800064913988966, 0.3609217786432976, 0.18532943849399547, 0.09250243427458617, 0.0032456994482310936, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4977152899824253, 0.3114235500878735, 0.129701230228471, 0.05834797891036907, 0.00281195079086116, 0.0, 0.0, 0.0, 0.0, 0.0], [0.514460511679644, 0.18298109010011124, 0.17519466073414905, 0.12569521690767518, 0.0016685205784204673, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4386406544996853, 0.2448080553807426, 0.15166771554436753, 0.16488357457520453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4755989352262644, 0.32963620230700974, 0.14330079858030167, 0.05146406388642413, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42741573033707864, 0.37258426966292135, 0.12719101123595505, 0.07280898876404494, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3697893321380547, 0.3796503809950695, 0.20035858359480055, 0.0502017032720753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2834190231362468, 0.230719794344473, 0.42159383033419023, 0.06426735218508997, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.434270765206017, 0.3590582079790713, 0.16677567037279267, 0.03989535644211903, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.38286399041342123, 0.26003594967046134, 0.3385260635110845, 0.018573996405032954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5700619020821609, 0.33033202025886327, 0.09285312324141812, 0.006752954417557681, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42862791617240015, 0.4424673784104389, 0.1273230525899565, 0.0015816528272044287, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5735217163788592, 0.30507587650444795, 0.12087912087912088, 0.0005232862375719519, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24647177419354838, 0.4632056451612903, 0.2903225806451613, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.432449494949495, 0.20075757575757575, 0.3667929292929293, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6564945226917058, 0.26134585289514867, 0.08215962441314555, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.26082817705854355, 0.6216087577344122, 0.11756306520704426, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.43658679135008766, 0.23670368205727646, 0.3267095265926359, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.46619718309859154, 0.4802816901408451, 0.05352112676056338, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.561344537815126, 0.4162464985994398, 0.022408963585434174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6771004942339374, 0.2915980230642504, 0.03130148270181219, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.38388625592417064, 0.6113744075829384, 0.004739336492890996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.521613832853026, 0.4783861671469741, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8765231519090171, 0.12347684809098294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7229965156794426, 0.2770034843205575, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5443425076452599, 0.45565749235474007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.863013698630137, 0.136986301369863, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.94529262086514, 0.054707379134860054, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.955503512880562, 0.04449648711943794, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8708971553610503, 0.12910284463894967, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9128738621586476, 0.08712613784135241, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9603960396039604, 0.039603960396039604, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
5000: ([0.33045530118343747, 0.019867178524392636, 0.015680313092536032, 0.012101745733740578, 0.010239489636749142, 0.009233054053296719, 0.012924874604197606, 0.012156620991771047, 0.009500424991360066, 0.012736897656476214, 0.006789353732918616, 0.007068400257796957, 0.0071664751870429005, 0.007638168894368631, 0.007304247111459822, 0.008097019456197868, 0.006873417957986568, 0.009285594193964188, 0.009171173443177255, 0.007928891006061964, 0.006158872044908977, 0.0077058872978955925, 0.0076673578614061145, 0.0076778658895396085, 0.006254611856791922, 0.0069925089434995005, 0.010441477288648528, 0.007624158190190639, 0.007151296924183409, 0.0073427765479493, 0.007979096029366436, 0.010982056958182718, 0.008155397390272835, 0.0061577044862274775, 0.008118035512464856, 0.009721093582163439, 0.006139023547323488, 0.0068325534041340915, 0.009092947011516798, 0.0101414147075032, 0.008520843257582126, 0.0077782759361485505, 0.007053221994937466, 0.008464800440870158, 0.01117236902326711, 0.009616013300828499, 0.008500994759996638, 0.009697742408533452, 0.00634684899263037, 0.0074898889418182155, 0.007432678566424748, 0.00987871400416585, 0.008141386686094843, 0.007345111665312299, 0.005970895097187585, 0.00739648424729827, 0.00987871400416585, 0.005931198102016607, 0.004816179561184745, 0.009721093582163439, 0.00702870326262598, 0.008349212131401725, 0.005843631200904157, 0.005898506458934626, 0.0058588094637636486, 0.005708194393850235, 0.007262214998925846, 0.005653319135819766, 0.0051839605458570345, 0.004776482566013768, 0.005208479278168521, 0.0063772055183493525, 0.00893182391346989, 0.004928265194608681, 0.0051979712500350266, 0.004741455805568788, 0.015149073892453836, 0.004646883552367342, 0.004169352051634115, 0.004614191909285361, 0.003631107499462923, 0.00447525242618694, 0.0026690391459074734, 0.0036497884383669124, 0.001873931683806428, 0.004645715993685843, 0.0023841548276216365, 0.0025616237472095345, 0.0017945376934644735, 0.0007484051148410719, 0.00046702347259973284, 0.001764181167745491, 0.0013333520142722374, 0.0002907221116933337, 0.0002218361494848731, 0.00011675586814993321, 9.223713583844724e-05, 3.852943648947796e-05, 5.837793407496661e-06, 0.0] ,
[[0.21159872946779681, 0.13269217859527754, 0.13177355130710064, 0.1323459267712724, 0.11698718514932993, 0.10633464178835533, 0.0863792305436507, 0.05322031862234172, 0.022050588098123526, 0.0066176496567513805], [0.21585566525622943, 0.13381523272214385, 0.12511753643629525, 0.12858486130700517, 0.10496003761165962, 0.08192289609779031, 0.08444992947813822, 0.05577103902209685, 0.04431123648330983, 0.02521156558533145], [0.1934475055845123, 0.13179448994787787, 0.1302308265078183, 0.15152643335815338, 0.10290394638868205, 0.11027550260610573, 0.0810126582278481, 0.05681310498883098, 0.030603127326880118, 0.01139240506329114], [0.19064158224794983, 0.17250361794500724, 0.1525325615050651, 0.1225277375783888, 0.08287506029908345, 0.08075253256150507, 0.07969126869271587, 0.05779064158224795, 0.04708152436082971, 0.013603473227206947], [0.22132269099201823, 0.12120866590649942, 0.15142531356898517, 0.1152793614595211, 0.1314709236031927, 0.08437856328392246, 0.06145952109464082, 0.04230330672748005, 0.06556442417331813, 0.005587229190421893], [0.16236722306525037, 0.16451694486595853, 0.1379615579160344, 0.10609509357612544, 0.11001517450682853, 0.09319676277187658, 0.10293373798684877, 0.07182599898836621, 0.04312089023773394, 0.007966616084977238], [0.35898825654923217, 0.14706413730803974, 0.1043360433604336, 0.12899728997289972, 0.095483288166215, 0.05998193315266486, 0.05013550135501355, 0.03342366757000903, 0.02041553748870822, 0.0011743450767841012], [0.4236457933154053, 0.1250480215136381, 0.08106031502112947, 0.10766423357664233, 0.0661736457933154, 0.08932001536688436, 0.057913945447560504, 0.029389166346523242, 0.01930464848252017, 0.0004802151363810987], [0.18372864692146984, 0.138626029249109, 0.14096104215312769, 0.1510384662652083, 0.11920855352095366, 0.1013887182008111, 0.07103355044856827, 0.04706894432837655, 0.04694604891237557, 0.0], [0.26198551654597124, 0.09680080667338894, 0.130534421120176, 0.11889265743881199, 0.1420845173709781, 0.11696764139701164, 0.06875057292144102, 0.0270418920157668, 0.036941974516454305, 0.0], [0.15356835769561478, 0.09234737747205503, 0.10902837489251935, 0.188134135855546, 0.12141014617368874, 0.1399828030954428, 0.08873602751504729, 0.06139294926913156, 0.04539982803095443, 0.0], [0.1491575817641229, 0.13214403700033037, 0.10885365047902214, 0.09596960687148992, 0.1278493557978196, 0.10769739015526925, 0.08754542451271886, 0.1669970267591675, 0.023785926660059464, 0.0], [0.1679700228087325, 0.1360377973281199, 0.13831867057673508, 0.13294232649071358, 0.17318344737699576, 0.1104594330400782, 0.07103290974258716, 0.05653307266210492, 0.013522319973932877, 0.0], [0.1342097217976154, 0.14964842555793334, 0.20009171507184348, 0.13711403240599204, 0.10990522775909507, 0.12198104555181902, 0.07077346377254662, 0.06297768266585142, 0.013298685417303576, 0.0], [0.14993606138107418, 0.13970588235294118, 0.16528132992327366, 0.15233375959079284, 0.11924552429667519, 0.10342071611253197, 0.06377877237851662, 0.09398976982097186, 0.012308184143222507, 0.0], [0.2158615717375631, 0.21196827685652486, 0.11463590483056957, 0.11492429704397981, 0.09041095890410959, 0.12674837779379958, 0.0821917808219178, 0.03763518385003605, 0.005623648161499639, 0.0], [0.18158654662816376, 0.11975539323934092, 0.19636487175131645, 0.1632410395787328, 0.14727365381348734, 0.07660947851197554, 0.05639544759639885, 0.05673517920842534, 0.0020383896721589945, 0.0], [0.15000628693574752, 0.08600528102602792, 0.19564944046271848, 0.11002137558154156, 0.12033195020746888, 0.09795045894630956, 0.17653715579026782, 0.06261788004526593, 0.0008801710046523325, 0.0], [0.23819223424570338, 0.1694462126034373, 0.16397199236155316, 0.13214513049013368, 0.08274984086569065, 0.11839592616168046, 0.06199872692552515, 0.03297262889879058, 0.0001273074474856779, 0.0], [0.2837579148873509, 0.19334413193933148, 0.13444264467677808, 0.12487115299661317, 0.11780297452510675, 0.07053453099690767, 0.056545427772051245, 0.018701222205860697, 0.0, 0.0], [0.26881516587677723, 0.12587677725118485, 0.1429383886255924, 0.10786729857819906, 0.11772511848341233, 0.12360189573459715, 0.07639810426540285, 0.036777251184834124, 0.0, 0.0], [0.13893939393939395, 0.16712121212121211, 0.15045454545454545, 0.18303030303030304, 0.1896969696969697, 0.0893939393939394, 0.05196969696969697, 0.029393939393939392, 0.0, 0.0], [0.17207248363027258, 0.17329069590376123, 0.14435815440840566, 0.1498401096391046, 0.11085731688746764, 0.07933607431094868, 0.07842241510583219, 0.0918227501142074, 0.0, 0.0], [0.32116788321167883, 0.14583333333333334, 0.1583029197080292, 0.13260340632603407, 0.08743917274939172, 0.06478102189781022, 0.04759732360097323, 0.04227493917274939, 0.0, 0.0], [0.14485719619189846, 0.132536867649804, 0.11293634496919917, 0.20123203285420946, 0.2019787194325182, 0.12040321075228673, 0.07205525480679485, 0.014000373343289154, 0.0, 0.0], [0.19936550342294207, 0.18417098013023878, 0.17982968776089497, 0.1390883286024378, 0.11070295541826682, 0.07263316079479044, 0.10586074469861413, 0.008348639171814994, 0.0, 0.0], [0.3747064743374707, 0.18226545901822655, 0.11383204741138321, 0.10533378061053338, 0.13608408811360842, 0.056580565805658053, 0.029743933802974392, 0.001453650900145365, 0.0, 0.0], [0.25068912710566615, 0.18208269525267995, 0.1883614088820827, 0.1226646248085758, 0.1, 0.09571209800918837, 0.05880551301684533, 0.0016845329249617152, 0.0, 0.0], [0.22938775510204082, 0.1763265306122449, 0.1539591836734694, 0.16473469387755102, 0.13012244897959183, 0.07591836734693877, 0.06955102040816327, 0.0, 0.0, 0.0], [0.21863571314994434, 0.15980283033868659, 0.18110987438384482, 0.1776117029734457, 0.12164096040705995, 0.06837335029416441, 0.07282556845285419, 0.0, 0.0, 0.0], [0.21992976294995611, 0.22431957857769974, 0.23983026046239392, 0.12935323383084577, 0.07930933567456834, 0.06950541410594088, 0.03775241439859526, 0.0, 0.0, 0.0], [0.30352966191792474, 0.13640229640654902, 0.19264299383372316, 0.17180523070380607, 0.09419519455666596, 0.07984265362534552, 0.02158196895598554, 0.0, 0.0, 0.0], [0.2462419470293486, 0.19957050823192554, 0.15118110236220472, 0.1279885468861847, 0.09892627057981389, 0.15977093772369363, 0.01632068718682892, 0.0, 0.0, 0.0], [0.1528251801289344, 0.18297307546454303, 0.2567311338642397, 0.19036784224497535, 0.10997345468335229, 0.08797876374668183, 0.019150549867273416, 0.0, 0.0, 0.0], [0.34100388321587805, 0.2384582194736085, 0.1763267654249964, 0.13404285919746872, 0.0533582626204516, 0.050625629224795056, 0.0061843808428016685, 0.0, 0.0, 0.0], [0.17763631996156617, 0.21414845063656018, 0.15781888061494115, 0.20838337737208742, 0.16838818159980784, 0.07062214748979101, 0.0030026423252462165, 0.0, 0.0, 0.0], [0.25770254849752755, 0.19303917839482693, 0.1791555724610118, 0.16793457588436667, 0.0798782807151008, 0.1200076074553062, 0.002282236591860023, 0.0, 0.0, 0.0], [0.19822282980177716, 0.2042036910457963, 0.2332535885167464, 0.17259056732740943, 0.12987012987012986, 0.061175666438824335, 0.000683526999316473, 0.0, 0.0, 0.0], [0.18464304057524397, 0.1332819722650231, 0.16088854648176681, 0.20505906522855674, 0.2526964560862866, 0.06343091936312276, 0.0, 0.0, 0.0, 0.0], [0.22012433801519687, 0.19525673497582316, 0.18800368408933918, 0.23981119042136773, 0.09014506101772968, 0.0666589914805434, 0.0, 0.0, 0.0, 0.0], [0.17689778021375718, 0.1882707591120855, 0.1522334886270211, 0.23211838859961634, 0.20375445327486982, 0.046725130172650044, 0.0, 0.0, 0.0, 0.0], [0.29810867607325126, 0.21840288201741218, 0.18778144701290903, 0.10372260582407686, 0.1344941459021315, 0.057490243170219155, 0.0, 0.0, 0.0, 0.0], [0.21999668928985266, 0.1887104783976163, 0.1789438834630028, 0.17745406389670584, 0.17050157258732, 0.0643933123655024, 0.0, 0.0, 0.0, 0.0], [0.2635862068965517, 0.21255172413793103, 0.2315862068965517, 0.11268965517241379, 0.1456551724137931, 0.033931034482758624, 0.0, 0.0, 0.0, 0.0], [0.19834883477897378, 0.26742606332950153, 0.14724631622949105, 0.30609259065733097, 0.06458355105026649, 0.0163026439544362, 0.0, 0.0, 0.0, 0.0], [0.24781447304516754, 0.16864983001457018, 0.15517241379310345, 0.3196940262263235, 0.10356969402622632, 0.005099562894609034, 0.0, 0.0, 0.0, 0.0], [0.22181019090784232, 0.18088174701277296, 0.1980497184452685, 0.3071006729844801, 0.08968548276335668, 0.002472187886279357, 0.0, 0.0, 0.0, 0.0], [0.2224897664339032, 0.25042138213339754, 0.25162533108596197, 0.1934745966771009, 0.07861786660245605, 0.0033710570671803513, 0.0, 0.0, 0.0, 0.0], [0.24908020603384842, 0.2993009565857248, 0.22718910963944078, 0.12693156732891833, 0.09731420161883739, 0.0001839587932303164, 0.0, 0.0, 0.0, 0.0], [0.25019485580670303, 0.2936866718628215, 0.21387373343725644, 0.1289166017147311, 0.11332813717848791, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2904492617027961, 0.3498272070373861, 0.1993402450518379, 0.10226201696512724, 0.05812126924285266, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2770358113698144, 0.3921522278690462, 0.13816333766694244, 0.12764448646732066, 0.06500413662687625, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2757780008604618, 0.28409579807830204, 0.2739136670012907, 0.1048329270041589, 0.06137960705578661, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19615323477984423, 0.37609283102845337, 0.14401525989508823, 0.24304562072802416, 0.04069305356859005, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2964411419632382, 0.35001955416503716, 0.21529135705905358, 0.12240907313257723, 0.01583887368009386, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3425414364640884, 0.298973954222573, 0.2606156274664562, 0.09060773480662983, 0.007261247040252565, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36567781586100934, 0.24949769530788324, 0.3019737619666706, 0.07859591064885947, 0.004254816215577355, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5322834645669291, 0.2234251968503937, 0.15866141732283465, 0.08464566929133858, 0.000984251968503937, 0.0, 0.0, 0.0, 0.0, 0.0], [0.43903030303030305, 0.2695757575757576, 0.13503030303030303, 0.15636363636363637, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.438866202257987, 0.33965889983185205, 0.15205380735046842, 0.06942109055969253, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4325581395348837, 0.3367109634551495, 0.12574750830564785, 0.10498338870431893, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.37547196196336174, 0.46203328205845334, 0.1304712627604531, 0.032023493217731784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45894105894105897, 0.28751248751248754, 0.18681318681318682, 0.06673326673326674, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5138558986539984, 0.24564528899445764, 0.20190023752969122, 0.03859857482185273, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4944200876843364, 0.26205659625348743, 0.21960940613790356, 0.023913909924272617, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4297402331765187, 0.3953773777868685, 0.15054203313561054, 0.02434035590100225, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3885852090032154, 0.3102893890675241, 0.2954983922829582, 0.005627009646302251, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6187525815778604, 0.25051631557207765, 0.11957868649318464, 0.011152416356877323, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4369369369369369, 0.3398648648648649, 0.22297297297297297, 0.00022522522522522523, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.43290149107797604, 0.25421657296504524, 0.3128819359569787, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5113203317641785, 0.3624747814391392, 0.12620488679668235, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4553277187843281, 0.42365433906993777, 0.12101794214573416, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5644444444444444, 0.3475816993464052, 0.08797385620915033, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.591802890310353, 0.38000473821369346, 0.028192371475953566, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5873764600179695, 0.3719676549865229, 0.04065588499550764, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5774439793154396, 0.40310268406796357, 0.019453336616596897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.26458574181117533, 0.7320231213872832, 0.003391136801541426, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6231155778894473, 0.37236180904522614, 0.004522613065326633, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6454774572948754, 0.3542425091010921, 0.0002800336040324839, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6057692307692307, 0.3942307692307692, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7099678456591639, 0.290032154340836, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.663970780067832, 0.336029219932168, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8814523184601925, 0.11854768153980752, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8662827895073576, 0.13371721049264235, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9059190031152647, 0.0940809968847352, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9612968082432772, 0.03870319175672279, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9583741429970617, 0.0416258570029383, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9931631722880584, 0.0068368277119416595, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9954456733897202, 0.004554326610279766, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
10000: ([0.27547970993295623, 0.016924666550287567, 0.013424801015329014, 0.01085439559143387, 0.008958674410594902, 0.008026383456553637, 0.011066708814773226, 0.010498652990549702, 0.008720883600454822, 0.011258262522941624, 0.006256162981066378, 0.006440167774627154, 0.006482630419295025, 0.00713844237583215, 0.006747786044887734, 0.007350755599171506, 0.006724195686738916, 0.008653886983312181, 0.008529329892286424, 0.007031813956999495, 0.005837198220343382, 0.0068761175932173, 0.007024265042391874, 0.007531929549754424, 0.006135380347344434, 0.006772320017362504, 0.009506914333973418, 0.007032757571325448, 0.007949950696151469, 0.0067987412184891795, 0.007962217682388854, 0.010693037541695958, 0.007823506376473808, 0.005774919674830503, 0.007616854839090167, 0.009877754764072828, 0.007084656359252846, 0.006570386551608627, 0.008222655236351798, 0.008939802124075847, 0.007779156503154031, 0.007317729097763162, 0.00711390840335738, 0.00821982439337394, 0.010673221640850952, 0.008718052757476964, 0.008509513991441418, 0.009858882477553774, 0.007100697802794042, 0.007400767158446999, 0.007897108293898118, 0.00907002090105732, 0.008426475930757581, 0.00720826983595265, 0.007831998905407382, 0.008404772801260669, 0.009674877683992998, 0.007525324249472756, 0.00586833749309982, 0.0096928063561861, 0.008193403192247265, 0.009786224174455417, 0.008200952106854887, 0.0071875103207816905, 0.0076593174837580385, 0.006799684832815131, 0.00900679874121849, 0.009939089695259754, 0.009354992427495034, 0.006481686804969073, 0.006848752777764672, 0.00856235639369477, 0.012043349642134268, 0.009002080669588726, 0.006731744601346537, 0.006703436171567957, 0.016051823298781322, 0.007773494817198315, 0.006893102651084448, 0.007803690475628801, 0.006559063179697194, 0.010732669343385971, 0.006086312402394893, 0.006694943642634382, 0.0067053234002198625, 0.00597685314058438, 0.0042717420535878575, 0.003490429391699025, 0.0034545720473128224, 0.002115583318785946, 0.0023495996716222144, 0.0025024651924265514, 0.0016664228996324621, 0.0010021184141617639, 0.0007841435048666909, 0.0005463526947266114, 0.0005708866672013814, 0.0002453397247477011, 0.00031233634189034256, 1.7928672193101235e-05] ,
[[0.20519557033784225, 0.12864585652580487, 0.1277689670173083, 0.12831017226083352, 0.11346470690995783, 0.10342158175795794, 0.08582556064410275, 0.06036493675091885, 0.036250475267262904, 0.010752172528010797], [0.20478367528991973, 0.1269513826940232, 0.11869982158786797, 0.1219892952720785, 0.09991079393398751, 0.07799955396966994, 0.08647413024085637, 0.07822256913470116, 0.056199821587867974, 0.028768956289027654], [0.1827511070499754, 0.1244113305686371, 0.12293526393477192, 0.1430378857102692, 0.09742039783510227, 0.10437899768046671, 0.08230828705981584, 0.07457650945385534, 0.04589864342447459, 0.022281577282631616], [0.17178127445014343, 0.15543771190124314, 0.1374424063287838, 0.11153612101190993, 0.07493697296357472, 0.07554550986699123, 0.08484743110492915, 0.08910718942884464, 0.07919673128749022, 0.020168651656089717], [0.20444491257636402, 0.11196545186433537, 0.13987781756899095, 0.10659363808721298, 0.12155045291763218, 0.08068253633874026, 0.06014324836738993, 0.06562039182641669, 0.08931957025489783, 0.019801980198019802], [0.1509522689865977, 0.15295085821772866, 0.12826240300964026, 0.09863625675993416, 0.1022807430049377, 0.08723254173524571, 0.1045144603809076, 0.08711497766282623, 0.0726545967552316, 0.015400893486950389], [0.3388472032742155, 0.1388130968622101, 0.09848226466575716, 0.12175989085948158, 0.09251364256480218, 0.058066166439290585, 0.05772510231923601, 0.05422919508867667, 0.03504433833560709, 0.004519099590723056], [0.396458745281323, 0.11702318892683804, 0.07585834981125292, 0.10075498831565702, 0.06237641560309186, 0.08484630595002697, 0.06327521121696926, 0.059590149200071905, 0.03747977709868776, 0.002336868596081251], [0.16176152347976627, 0.12248431075524778, 0.12410733607444276, 0.13319627786193464, 0.105280242371781, 0.0924042415061675, 0.0815840727115343, 0.10690326769097598, 0.06957368534949145, 0.002705042198658299], [0.23954404492498532, 0.08850892632637666, 0.11943676137792306, 0.10870840667169558, 0.1299136702707233, 0.10895985248512279, 0.07778057162014919, 0.07275165535160506, 0.05439611097141899, 0.0], [0.13469079939668174, 0.08099547511312218, 0.09562594268476621, 0.16546003016591251, 0.10693815987933634, 0.12850678733031673, 0.10844645550527904, 0.11900452488687782, 0.06033182503770739, 0.0], [0.13230769230769232, 0.11721611721611722, 0.09714285714285714, 0.08527472527472528, 0.11501831501831501, 0.1136996336996337, 0.1054945054945055, 0.187985347985348, 0.04586080586080586, 0.0], [0.15007278020378456, 0.1215429403202329, 0.12358078602620087, 0.11892285298398836, 0.1572052401746725, 0.11484716157205241, 0.10131004366812227, 0.08500727802037845, 0.027510917030567687, 0.0], [0.11606080634500991, 0.12941176470588237, 0.17303370786516853, 0.11857237276933245, 0.09702577660277595, 0.12148050231328486, 0.09742233972240581, 0.09583608724388631, 0.0511566424322538, 0.0], [0.13117046566913718, 0.1222206684379807, 0.1445951615158719, 0.13354775555866313, 0.1082366102642987, 0.10306250874003636, 0.10068521885051042, 0.12250034960145434, 0.033981261362047266, 0.0], [0.19216944801026958, 0.18870346598202825, 0.10205391527599486, 0.10269576379974327, 0.08279845956354301, 0.1281129653401797, 0.11951219512195121, 0.07317073170731707, 0.010783055198973043, 0.0], [0.15029469548133595, 0.09991580129104687, 0.16222284591636263, 0.13485826550659558, 0.12250912152680325, 0.09837215829357283, 0.11801852371596969, 0.11058097109177659, 0.0032276171765366264, 0.0], [0.1300839603096718, 0.10380547377603315, 0.16966524915494494, 0.09628175771453494, 0.10740377276196707, 0.10075237160614982, 0.190818885617708, 0.09922582052120815, 0.001962708537782139, 0.0], [0.20699192388538556, 0.1472508020798761, 0.14249363867684478, 0.11549950215731829, 0.07390198030755614, 0.12357561677176679, 0.11372939484456245, 0.07534019249917026, 0.001216948777519637, 0.0], [0.2585882984433709, 0.17619431025228127, 0.12278582930756844, 0.11406333870101985, 0.1107085346215781, 0.08158883521202362, 0.09406870638754697, 0.04200214707461084, 0.0, 0.0], [0.22922728742321372, 0.10733915292596186, 0.12204978984804397, 0.09246686065308762, 0.11121888134497251, 0.14080181053992888, 0.11396702230843841, 0.08292919495635305, 0.0, 0.0], [0.12584053794428435, 0.15136544531357213, 0.13627006998764923, 0.16687251269383835, 0.17675312199807877, 0.11650885137916839, 0.07822149032523672, 0.04816797035817209, 0.0, 0.0], [0.15180010746910264, 0.15287479849543256, 0.12735088662009672, 0.1324556689951639, 0.10666308436324556, 0.1123052122514777, 0.11996238581407845, 0.09658785599140247, 0.0, 0.0], [0.2645953395139063, 0.12014532698571787, 0.1321723878727136, 0.11024805813079429, 0.08957654723127036, 0.12553244800801805, 0.1061137559508895, 0.051616136306690055, 0.0, 0.0], [0.11934789295601353, 0.10919717010150723, 0.09350968932636113, 0.17548446631805598, 0.18794217163949553, 0.15718240541371886, 0.1364195632113196, 0.020916641033528145, 0.0, 0.0], [0.16636477636895639, 0.1536853838651247, 0.15131670614462867, 0.12135989967953184, 0.1081231712414658, 0.14281733314755468, 0.14100599136129302, 0.015326738191444894, 0.0, 0.0], [0.33260545905707195, 0.1617866004962779, 0.10124069478908189, 0.09459057071960297, 0.12873449131513648, 0.1036228287841191, 0.06709677419354838, 0.01032258064516129, 0.0, 0.0], [0.21964309673956797, 0.15966724808801824, 0.17160874815510532, 0.11216959613578424, 0.10317992754595465, 0.1441030457533879, 0.0850664162082383, 0.004561921373943379, 0.0, 0.0], [0.16676557863501484, 0.1281899109792285, 0.1481305637982196, 0.14160237388724037, 0.20154302670623145, 0.1251038575667656, 0.0886646884272997, 0.0, 0.0, 0.0], [0.19083969465648856, 0.13948646773074255, 0.16072172102706453, 0.1597501734906315, 0.14226231783483692, 0.09146426092990978, 0.11547536433032617, 0.0, 0.0, 0.0], [0.17847831239630244, 0.18167812277790946, 0.1944773643043375, 0.11329698980801138, 0.09480919649205972, 0.14102867978193884, 0.09623133443944062, 0.0, 0.0, 0.0], [0.2524708789269326, 0.11339569361101307, 0.15998941051888457, 0.15531238969290503, 0.12707377338510412, 0.11966113660430638, 0.07209671726085422, 0.0, 0.0, 0.0], [0.2075744783500181, 0.17175250271378603, 0.13339766011337595, 0.11458207695091062, 0.15293691955132072, 0.16330961283319262, 0.05644674948739597, 0.0, 0.0, 0.0], [0.13169934640522876, 0.1580065359477124, 0.22238562091503267, 0.1758169934640523, 0.14738562091503268, 0.1284313725490196, 0.03627450980392157, 0.0, 0.0, 0.0], [0.29385530227948464, 0.2055252725470763, 0.15262636273538158, 0.12091179385530228, 0.09799306243805748, 0.10877106045589693, 0.020317145688800792, 0.0, 0.0, 0.0], [0.1414787925105082, 0.1704241497898357, 0.12753152464654186, 0.1800726022162782, 0.17195261750095528, 0.1361291555215896, 0.07241115781429118, 0.0, 0.0, 0.0], [0.18060735215769846, 0.13598827916888653, 0.13026105487480022, 0.15623335109216835, 0.12173681406499734, 0.1759456579648375, 0.09922749067661162, 0.0, 0.0, 0.0], [0.16673847479534684, 0.1730575901192015, 0.2024989228780698, 0.1786586241562545, 0.17248312508976016, 0.08415912681315525, 0.02240413614821198, 0.0, 0.0, 0.0], [0.16502180399357355, 0.119233417489098, 0.1502180399357356, 0.1974982786320863, 0.28517328436997935, 0.08021574477851733, 0.002639430801009869, 0.0, 0.0, 0.0], [0.20181549503905424, 0.17975511927380197, 0.1780662866793329, 0.24076419674899727, 0.11484061642389698, 0.08475828583491661, 0.0, 0.0, 0.0, 0.0], [0.15684133915574963, 0.16690926734594858, 0.14640950994662785, 0.24284327996118388, 0.22161572052401746, 0.06538088306647259, 0.0, 0.0, 0.0, 0.0], [0.2577691811734365, 0.1882656350741457, 0.17653127014829142, 0.129464861379755, 0.15525467440361057, 0.0927143778207608, 0.0, 0.0, 0.0, 0.0], [0.17654861387451917, 0.1558562143520361, 0.15452977848521024, 0.22310651280010613, 0.2232391563867887, 0.0667197241013397, 0.0, 0.0, 0.0, 0.0], [0.21937779818620134, 0.1775915509126392, 0.20066582481919412, 0.14579267592698886, 0.21122718402020435, 0.045344966134772124, 0.0, 0.0, 0.0, 0.0], [0.16983467421094509, 0.22677040049509328, 0.1536557333569092, 0.3125276279727699, 0.10732914861639112, 0.029882415347891433, 0.0, 0.0, 0.0, 0.0], [0.2210195908648122, 0.154453945232168, 0.1470938413248187, 0.34505898906808097, 0.12025110942742721, 0.012122524082692931, 0.0, 0.0, 0.0, 0.0], [0.17986249722776668, 0.14992237746728765, 0.18684852517187847, 0.32202262142381904, 0.1561321800842759, 0.005211798624972278, 0.0, 0.0, 0.0, 0.0], [0.17821592649310872, 0.20444104134762633, 0.2245405819295559, 0.28311638591117916, 0.09925344563552833, 0.010432618683001531, 0.0, 0.0, 0.0, 0.0], [0.18245847176079735, 0.2212624584717608, 0.1969435215946844, 0.15122923588039866, 0.15906976744186047, 0.08903654485049833, 0.0, 0.0, 0.0, 0.0], [0.20527859237536658, 0.2458243019252837, 0.21560627310977942, 0.16651791406349611, 0.1667729185260742, 0.0, 0.0, 0.0, 0.0, 0.0], [0.22093440076472698, 0.2750627315091409, 0.17911339467080895, 0.1452981240291552, 0.179591349026168, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24542238868081565, 0.3585101955888473, 0.14346650020807325, 0.1571993341656263, 0.09540158135663754, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2170212765957447, 0.27077267637178054, 0.2623740201567749, 0.16002239641657334, 0.08980963045912654, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16651394161539468, 0.3203298861107475, 0.16507396256054457, 0.275690535410394, 0.07239167430291923, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18879518072289156, 0.2531325301204819, 0.2272289156626506, 0.30325301204819277, 0.027590361445783133, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2494667115751656, 0.23779050185247558, 0.29808016167059614, 0.17817446951835636, 0.03648815538340631, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3045937774309958, 0.22042329074417244, 0.3030332585584707, 0.15956305471569296, 0.012386618550668096, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3438244514106583, 0.19059561128526645, 0.25579937304075234, 0.204012539184953, 0.005768025078369906, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2992442514873774, 0.23653320469528863, 0.22366940022511658, 0.23878437047756873, 0.0017687731146486573, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3596183800623053, 0.3305101246105919, 0.2171923676012461, 0.09258177570093458, 9.735202492211838e-05, 0.0, 0.0, 0.0, 0.0, 0.0], [0.31440746285845905, 0.32304503051940575, 0.21190832661522516, 0.15063918000691007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.30257448654903096, 0.3896442001735609, 0.24857776492141548, 0.059203548355992675, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2740766309975837, 0.24416062593487517, 0.330456794384996, 0.15130594868254515, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3679926480241565, 0.27084153866351585, 0.29788630694499146, 0.06327950636733622, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3418750769988912, 0.3395343107059258, 0.2837255143525933, 0.03486509794258963, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.31681931723563694, 0.39369969469886207, 0.2497918401332223, 0.03968914793227866, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.29837611314824514, 0.35159769512833944, 0.32404400209533785, 0.025982189628077527, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.41991835184657744, 0.25064084306465395, 0.32042153232697235, 0.00901927276179626, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2672987694169861, 0.25983457736534193, 0.47205971353641313, 0.0008069396812588259, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4194205852380259, 0.3009171640704615, 0.27966225069151257, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44020391292367045, 0.37489666574813996, 0.1848994213281896, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.41271765483799866, 0.46616707075159797, 0.12111527441040335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45937475515161014, 0.44934576510224866, 0.09127947974614119, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5969601677148847, 0.3372117400419287, 0.06582809224318659, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5420521446593777, 0.39921502663302494, 0.05873282870759742, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5537725225225225, 0.3980855855855856, 0.04814189189189189, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.37946034918582094, 0.6078419845982012, 0.012697666215977897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6125273124544792, 0.3738771546491867, 0.013595532896334061, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5467488021902807, 0.4444900752908966, 0.008761122518822724, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5440145102781136, 0.45574365175332526, 0.0002418379685610641, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5581930657459359, 0.44180693425406414, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42394935818533497, 0.576050641814665, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6241860465116279, 0.3758139534883721, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7422128259337561, 0.25778717406624385, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5989304812834224, 0.40106951871657753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.929428481212504, 0.07057151878749605, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.920035343494588, 0.07996465650541197, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9591781562584483, 0.04082184374155177, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9937175635072385, 0.0062824364927615405, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] ),
math.inf: ([0.25945452693035187, 0.0160018427545637, 0.012672481384192177, 0.010301706762820655, 0.008544002409755967, 0.0077165347354805555, 0.010483324400108084, 0.010012890422548937, 0.00825164231070791, 0.010748220368639507, 0.006066915025094242, 0.006279540551674648, 0.006211323195230101, 0.006857173232218083, 0.00657633034919313, 0.007149533331266141, 0.0064248346615045915, 0.008355297254915859, 0.008140013909253197, 0.0067003619063650335, 0.005692162534496277, 0.006702133785753204, 0.006777438659750431, 0.007235469481592388, 0.006059827507541562, 0.006692388449118268, 0.009526509530496259, 0.006973231332143221, 0.007860942905616414, 0.007015756437459302, 0.007892836734603476, 0.010276900451386274, 0.007534917098193126, 0.0057338017001182726, 0.00730457277773102, 0.009398934214548016, 0.006833252860477788, 0.0064655878874325026, 0.00791321334756743, 0.008605132248647835, 0.01215952230131695, 0.008180767135181108, 0.008070024673420481, 0.009023295784255966, 0.011031721070746714, 0.008977226920163544, 0.008267589225201441, 0.00941931082751197, 0.006897040518451909, 0.007173453703006437, 0.0077014737606811105, 0.008911667382801252, 0.0080629371558678, 0.007062711241245808, 0.009059619311713452, 0.008132040452006432, 0.009331602797797553, 0.00883370468972177, 0.005940225648840084, 0.009432599922923246, 0.008148873306194047, 0.009642567630421397, 0.00808951534669035, 0.007179655280865032, 0.007458726284501814, 0.006816420006290172, 0.008736251323372418, 0.009772800765451896, 0.009258955742882583, 0.0066507492834962725, 0.007804242765194973, 0.009058733372019366, 0.011915002945749483, 0.009310340245139513, 0.0066206273338973816, 0.00693956562376799, 0.01538877248625686, 0.007911441468179261, 0.006971459452755051, 0.007823733438464844, 0.0073568432196820365, 0.011250548175185716, 0.006598478841545256, 0.006901470216922334, 0.007594275057696822, 0.006132474562456533, 0.005005559271580384, 0.0038830736791746587, 0.004779644649588703, 0.003483514877142313, 0.003036115331629376, 0.003307212878019393, 0.0030671232209223518, 0.0015238162738262406, 0.008127610753536007, 0.0016655666248798445, 0.000973647723799441, 0.0003410867822227341, 0.0003410867822227341, 2.0376612963955543e-05] ,
[[0.2045530598447029, 0.12824303928866548, 0.12736889550567168, 0.12790840612173818, 0.11310942504558523, 0.1030977470309843, 0.08556023738467107, 0.060247628543526215, 0.03745842695094551, 0.012453134283509415], [0.20335511017606023, 0.12606577344701583, 0.11787177499723175, 0.1211383014062673, 0.09921381906765585, 0.0774554312922157, 0.08587088915956151, 0.07773225556416787, 0.05769017827483114, 0.0336064666149928], [0.18176733780760626, 0.12374161073825503, 0.1222734899328859, 0.1422678970917226, 0.09689597315436242, 0.10381711409395973, 0.0818652125279642, 0.07487416107382551, 0.048308165548098435, 0.02418903803131991], [0.16993464052287582, 0.15376676986584106, 0.13596491228070176, 0.11033711730306157, 0.07413140694874441, 0.07473340213278294, 0.08393532851737186, 0.08857929136566907, 0.0823873409012728, 0.026229790161678708], [0.20126503525508088, 0.11022397345499793, 0.13770219825798424, 0.10493571132310245, 0.11965989216092908, 0.07942762339278307, 0.059207797594359186, 0.06584404811281626, 0.09612194110327665, 0.025611779344670262], [0.14741676234213547, 0.14936854190585533, 0.12525832376578644, 0.09632606199770379, 0.09988518943742825, 0.08518943742824339, 0.10206659012629161, 0.08691159586681975, 0.0807118254879449, 0.026865671641791045], [0.3358404462097524, 0.1375813403194456, 0.09760838333474182, 0.12067945575931717, 0.09169272373869687, 0.057550916927237385, 0.05721287923603482, 0.05383250232400913, 0.03828276852869095, 0.00971858362207386], [0.3902849053264909, 0.11520084940718457, 0.07467704831003362, 0.09918598478145461, 0.061405061051141394, 0.08352503981596178, 0.06228986020173421, 0.059281543089718634, 0.04016988143691382, 0.013979826579366484], [0.1605110586214301, 0.12153747047455443, 0.12314794932359888, 0.1321666308782478, 0.10446639467468327, 0.09168992913893065, 0.08095340347863432, 0.10682843031994846, 0.07440412282585356, 0.004294610264118532], [0.2355753379492252, 0.08704253214638971, 0.11745796241345202, 0.10690735245631389, 0.12776129244971976, 0.10715463237718431, 0.07649192218925156, 0.07228816353445433, 0.06758984503791625, 0.0017309594460929772], [0.13040303738317757, 0.07841705607476636, 0.09258177570093458, 0.1601927570093458, 0.1035338785046729, 0.1244158878504673, 0.10499415887850468, 0.12353971962616822, 0.08192172897196262, 0.0], [0.12739841986455983, 0.11286681715575621, 0.09353837471783295, 0.08211060948081264, 0.11075056433408578, 0.10948081264108352, 0.10158013544018059, 0.1872178329571106, 0.07505643340857787, 0.0], [0.14705462844102124, 0.1190985594066467, 0.1210954214805306, 0.11653116531165311, 0.1540436456996149, 0.11253744116388532, 0.09998573669947226, 0.09328198545143346, 0.036371416345742404, 0.0], [0.11343669250645995, 0.1264857881136951, 0.16912144702842377, 0.11589147286821705, 0.09483204134366925, 0.11873385012919897, 0.09560723514211886, 0.09754521963824289, 0.06834625322997416, 0.0], [0.1263640037720598, 0.11774215276842247, 0.13929678027751582, 0.128654182944901, 0.10427051057523912, 0.09928600296376128, 0.09820827158830661, 0.13700660110467466, 0.04917149400511922, 0.0], [0.18550185873605948, 0.1821561338289963, 0.09851301115241635, 0.09913258983890955, 0.07992565055762081, 0.12366790582403965, 0.11821561338289963, 0.08661710037174722, 0.026270136307311027, 0.0], [0.1476833976833977, 0.09817981246552675, 0.1594043022614451, 0.13251516822945394, 0.12038058466629895, 0.09666298952013237, 0.1167953667953668, 0.1167953667953668, 0.011583011583011582, 0.0], [0.12649772028416922, 0.10094369632064468, 0.16498780617113773, 0.09362739900328704, 0.10444279503764183, 0.09797476407591985, 0.19467712861838618, 0.10995652634927368, 0.006892164139539816, 0.0], [0.20363517631693515, 0.14486286460600784, 0.14018284719198956, 0.11362646930779277, 0.07270352633870265, 0.12168045276447541, 0.11471484545058773, 0.08663474096647801, 0.00195907705703091, 0.0], [0.25479307153246067, 0.17360835647229936, 0.12098373661245537, 0.11238926351976729, 0.10908369694565649, 0.08356472299352109, 0.09837366124553748, 0.0465423773634801, 0.0006611133148221605, 0.0], [0.22070038910505838, 0.10334630350194553, 0.11750972762645914, 0.0890272373540856, 0.1070817120622568, 0.1368093385214008, 0.12124513618677042, 0.10428015564202335, 0.0, 0.0], [0.12121612690019828, 0.1458030403172505, 0.13126239259748843, 0.16074025115664242, 0.1702577660277594, 0.11222736285525446, 0.09081295439524124, 0.06768010575016524, 0.0, 0.0], [0.1477124183006536, 0.14875816993464053, 0.12392156862745098, 0.1288888888888889, 0.10379084967320261, 0.10980392156862745, 0.12718954248366013, 0.10993464052287581, 0.0, 0.0], [0.2586016897269499, 0.11742377862128077, 0.12917840088159668, 0.1077507040528958, 0.08754744704297784, 0.12293375780580384, 0.11338312721929718, 0.063181094649198, 0.0, 0.0], [0.11345029239766082, 0.10380116959064327, 0.08888888888888889, 0.16681286549707602, 0.17865497076023393, 0.15, 0.1631578947368421, 0.03523391812865497, 0.0, 0.0], [0.1580619539316918, 0.14601535610272703, 0.1437648927720413, 0.11530315064866296, 0.10272703203600742, 0.1437648927720413, 0.16190097961344982, 0.028461742123378344, 0.0, 0.0], [0.31163396261508414, 0.1515856040174835, 0.09485724913977495, 0.08862642983353483, 0.12061750209243932, 0.09913512508137264, 0.08555751883195388, 0.047986608388356736, 0.0, 0.0], [0.20797865582518105, 0.1511879049676026, 0.16249523567526364, 0.10621267945623174, 0.0977004192605768, 0.14674120188032017, 0.09960614915512642, 0.028077753779697623, 0.0, 0.0], [0.15834554265750028, 0.12171757015665502, 0.1406514144032458, 0.1344528344415643, 0.1913670686351854, 0.12645103121830273, 0.1252113152259664, 0.0018032232615800743, 0.0, 0.0], [0.17363303447404976, 0.12690996337921454, 0.14623058466978153, 0.14534663467609546, 0.12943553478974618, 0.09571915645914889, 0.17931557014774593, 0.0034095214042177044, 0.0, 0.0], [0.16904254125042092, 0.1720731844202492, 0.18419575709956224, 0.10730721742058592, 0.08979683466157817, 0.14468514984846784, 0.1328993152991357, 0.0, 0.0, 0.0], [0.24663793103448275, 0.11077586206896552, 0.15629310344827585, 0.15172413793103448, 0.12413793103448276, 0.12508620689655173, 0.0853448275862069, 0.0, 0.0, 0.0], [0.20235155790711346, 0.16743092298647855, 0.13004115226337448, 0.11169900058788948, 0.14944150499706055, 0.1676660787771899, 0.0713697824808936, 0.0, 0.0, 0.0], [0.12453646477132262, 0.14941285537700866, 0.21029048207663784, 0.1665636588380717, 0.14323238566131025, 0.14894932014833126, 0.05701483312731768, 0.0, 0.0, 0.0], [0.287689508793208, 0.20121285627653124, 0.14942389326864766, 0.11837477258944815, 0.0960582171012735, 0.11740448756822316, 0.029836264402668285, 0.0, 0.0, 0.0], [0.13959845414270902, 0.16815911018946178, 0.12583655386935622, 0.1776793288717127, 0.1699500424168159, 0.14101234800640966, 0.07776416250353474, 0.0, 0.0, 0.0], [0.17580707895760406, 0.13237391417088032, 0.12679891092959938, 0.15208090237261765, 0.11914948787760923, 0.17840010372099052, 0.11538960197069882, 0.0, 0.0, 0.0], [0.1590846807344478, 0.16511372978898328, 0.19320361742943273, 0.1704576596327761, 0.16812825431625103, 0.11194847903535216, 0.03206357906275692, 0.0, 0.0, 0.0], [0.16099417823555753, 0.11632333184057322, 0.14655172413793102, 0.19267801164352888, 0.283922973578146, 0.0960591133004926, 0.003470667263770712, 0.0, 0.0, 0.0], [0.1968495830330485, 0.175332029239164, 0.17368475239369915, 0.23494286008442294, 0.11942757129620096, 0.09183568413466488, 0.007927519818799546, 0.0, 0.0, 0.0], [0.09420765027322404, 0.10025500910746812, 0.08794171220400729, 0.14586520947176684, 0.13377049180327868, 0.437959927140255, 0.0, 0.0, 0.0, 0.0], [0.21648256443578082, 0.15811132770197098, 0.14825644357808102, 0.10872861165258826, 0.13666883257526533, 0.23175222005631363, 0.0, 0.0, 0.0, 0.0], [0.146119222746734, 0.1289933033263805, 0.12789548797892195, 0.18498188604676694, 0.20452299923152925, 0.20748710066966736, 0.0, 0.0, 0.0, 0.0], [0.18762886597938144, 0.15189003436426116, 0.17172312223858616, 0.12528227785959745, 0.18487972508591066, 0.17859597447226314, 0.0, 0.0, 0.0, 0.0], [0.15427240603919049, 0.20599100546097013, 0.13957597173144876, 0.2838901381304208, 0.10512367491166077, 0.11114680372630903, 0.0, 0.0, 0.0, 0.0], [0.20151978683509325, 0.14082700088818711, 0.13411625382413894, 0.3146156123556696, 0.15158393368202902, 0.05733741241488207, 0.0, 0.0, 0.0, 0.0], [0.1738105443634805, 0.14487783969138449, 0.18066866695242179, 0.3117231033004715, 0.17702528932704673, 0.011894556365195028, 0.0, 0.0, 0.0, 0.0], [0.17513167795334839, 0.20090293453724606, 0.22065462753950338, 0.2786869826937547, 0.11013920240782543, 0.014484574868322046, 0.0, 0.0, 0.0, 0.0], [0.17636480411046884, 0.2138728323699422, 0.19087989723827875, 0.15080282594733463, 0.17867694283879254, 0.08940269749518305, 0.0, 0.0, 0.0, 0.0], [0.19883907620106211, 0.23811288131406694, 0.2095837964678276, 0.16351735210571816, 0.18982339137952328, 0.00012350253180190194, 0.0, 0.0, 0.0, 0.0], [0.21269987346140573, 0.26481076728402164, 0.17335787415161624, 0.1495456114114805, 0.1995858736914759, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23451635351426584, 0.34267819862809423, 0.1376876429068496, 0.1726811810319117, 0.11243662391887861, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21294363256784968, 0.26568508955059883, 0.257554114932425, 0.1632787605757609, 0.10053840237336556, 0.0, 0.0, 0.0, 0.0, 0.0], [0.15955845459106874, 0.3070747616658304, 0.15855494229804315, 0.29327646763672854, 0.08153537380832915, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1532368472521025, 0.20545667905339332, 0.19127713671034619, 0.2880891844318404, 0.1619401525523176, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2420743000326833, 0.23074408977012748, 0.2901187493190979, 0.1897810218978102, 0.04728183898028108, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2964967245798918, 0.21465869173075097, 0.3023829868033798, 0.16823317193582074, 0.01822842495015665, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27499749272891383, 0.15244208203790993, 0.22003811052050948, 0.34349613880252733, 0.009026175910139404, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27755406413124534, 0.21938851603281134, 0.2116331096196868, 0.2841163310961969, 0.007307979120059657, 0.0, 0.0, 0.0, 0.0, 0.0], [0.34695219310603925, 0.3214990138067061, 0.2211890673429135, 0.10763595378979994, 0.0027237719545411855, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2968036529680365, 0.30495759947814743, 0.2121113285496847, 0.18612741900413132, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.28831312017640576, 0.37183020948180817, 0.2534913634693128, 0.08636530687247336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2608695652173913, 0.233490307742854, 0.3497973934946884, 0.15584273354506625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.34587857847976305, 0.2653010858835143, 0.3202122408687068, 0.0686080947680158, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.32961159282575125, 0.3284238033020549, 0.29611592825751276, 0.045848675614681075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.29737457759292957, 0.38055627761892385, 0.26865089680270343, 0.0534182479854432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2888145218537674, 0.3424601967346111, 0.34124328161444073, 0.027481999797180814, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.40096092829299246, 0.24839089837730033, 0.3392258181488532, 0.011422355180853957, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2537556214716295, 0.2517462443785284, 0.48732178738876664, 0.007176346761075495, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3837751431996803, 0.28799786865592114, 0.32463034501132276, 0.003596643133075796, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3626972414576002, 0.37359518674083325, 0.26370757180156656, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.37574572127139366, 0.4541809290953545, 0.17007334963325182, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4373559372444048, 0.44516320916053237, 0.11748085359506283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5423922352269483, 0.3803406603863355, 0.07726710438671615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.521744948481199, 0.40840358624381107, 0.06985146527498996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5082343929528916, 0.42920975360653646, 0.06255585344057193, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3742084052964882, 0.61237766263673, 0.013413932066781807, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5792833146696529, 0.4006718924972004, 0.020044792833146696, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5320879400177914, 0.4582539077392299, 0.009658152242978777, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5335749065790963, 0.46348091948816666, 0.0029441739327369493, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5215558766859345, 0.4784441233140655, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4303488463658556, 0.5696511536341444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6345327604726101, 0.3654672395273899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7224646983311939, 0.2775353016688062, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6444237050863276, 0.35557629491367243, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9274776076278533, 0.07252239237214678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.904070796460177, 0.09592920353982301, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9621263974446727, 0.0378736025553274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9341983317886933, 0.06580166821130677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9888097660223805, 0.011190233977619531, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] ),
},
}
TRUNCATION_MODES = list(sorted(TRUNCATION_DICT.keys()))


def calculate_aligned_length(ref_len, truncation_mode='ont_r9'):
    truncation_dict = TRUNCATION_DICT[truncation_mode]
    pro_list = None
    tlens = list(sorted(truncation_dict.keys()))
    for tlen in tlens:
        if ref_len <= tlen:
            pro_list = truncation_dict[tlen]
    if not pro_list:
        pro_list = truncation_dict[tlens[-1]]

    pro3_list = pro_list[0]
    tr3_index = np.random.choice(TR3, p=pro3_list)
    del3 = int(ref_len * TR_PERC3[tr3_index])
    pro5_lists = pro_list[1]
    tr5_index = np.random.choice(TR5, p=pro5_lists[tr3_index])
    del5 = int(ref_len * TR_PERC5[tr5_index])

    new_len = max(min(50, ref_len - 1), ref_len - del3 - del5 - 1)
    start_pos = max(0, min(del5, ref_len - new_len - 1))

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
    parser_t.add_argument('--truncation_mode', help='truncation mode',
                          default='none', choices=TRUNCATION_MODES)

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

