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
    'curio_ctrl': {500: ([0.5872816810938122, 0.044015538494478944, 0.011256232220964732, 0.0066069189123053864, 0.004374025020646622, 0.0026917077050133058, 0.0024470070045575507, 0.0011929159147218058, 0.0019576056036460406, 0.002171718716544826, 0.0011623283271648365, 0.0014376166151775609, 0.0011623283271648365, 0.000795277276481204, 0.0008258648640381733, 0.0009788028018230203, 0.0025081821796714894, 0.0018046676658611935, 0.0015599669654054384, 0.0018964304285321017, 0.0032422842810387546, 0.07441960052610651, 0.007096320313216897, 0.003425809806380571, 0.000642339338696357, 0.0008564524515951427, 0.0010705655644939283, 0.010705655644939284, 0.0014682042027345304, 0.0011317407396078671, 0.004221087082861775, 0.0011011531520508978, 0.0013458538525066529, 0.0040987367326338975, 0.0007341021013672652, 0.03168874070902028, 0.0023552442418866425, 0.002814058055241183, 0.0009482152142660508, 0.0011011531520508978, 0.0003670510506836326, 0.0004894014009115101, 0.000397638638240602, 0.0005505765760254489, 0.0004588138133545407, 0.0003364634631266632, 0.0004588138133545407, 0.0003670510506836326, 0.0004894014009115101, 0.0007035145138102958, 0.0005199889884684795, 0.0005505765760254489, 0.0011011531520508978, 0.13476891077600708, 0.003915211207292081, 0.0017434924907472548, 0.0021105435414308874, 0.0005505765760254489, 0.0009176276267090814, 0.0008258648640381733, 0.0008564524515951427, 0.0009788028018230203, 0.000795277276481204, 0.0001835255253418163, 0.00042822622579757137, 0.0012235035022787753, 0.0001835255253418163, 0.0003670510506836326, 0.00024470070045575507, 0.000397638638240602, 0.0008870400391521121, 0.0007646896889242345, 0.000397638638240602, 0.000642339338696357, 0.003701098094393295, 0.0016211421405193773, 0.0011929159147218058, 0.00021411311289878569, 0.00012235035022787753, 0.0, 6.117517511393877e-05, 6.117517511393877e-05, 3.0587587556969384e-05, 9.176276267090815e-05, 0.00015293793778484692, 0.0, 9.176276267090815e-05, 3.0587587556969384e-05, 0.0, 6.117517511393877e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
    [[0.28328125, 0.13223958333333333, 0.1446875, 0.07729166666666666, 0.09895833333333333, 0.09473958333333334, 0.0875, 0.05, 0.030729166666666665, 0.0005729166666666667], [0.20152883947185546, 0.12369701181375956, 0.11674774148714386, 0.11883252258512857, 0.12786657400972898, 0.12786657400972898, 0.0965948575399583, 0.05350938151494093, 0.03335649756775538, 0.0], [0.41032608695652173, 0.059782608695652176, 0.10597826086956522, 0.07065217391304347, 0.057065217391304345, 0.09782608695652174, 0.15217391304347827, 0.035326086956521736, 0.010869565217391304, 0.0], [0.2037037037037037, 0.06018518518518518, 0.13425925925925927, 0.09722222222222222, 0.11574074074074074, 0.1111111111111111, 0.21296296296296297, 0.041666666666666664, 0.023148148148148147, 0.0], [0.4965034965034965, 0.06293706293706294, 0.04895104895104895, 0.06293706293706294, 0.06993006993006994, 0.04895104895104895, 0.16083916083916083, 0.04895104895104895, 0.0, 0.0], [0.29545454545454547, 0.13636363636363635, 0.13636363636363635, 0.045454545454545456, 0.125, 0.056818181818181816, 0.13636363636363635, 0.06818181818181818, 0.0, 0.0], [0.225, 0.05, 0.275, 0.1125, 0.1375, 0.0625, 0.125, 0.0125, 0.0, 0.0], [0.46153846153846156, 0.20512820512820512, 0.10256410256410256, 0.02564102564102564, 0.05128205128205128, 0.02564102564102564, 0.10256410256410256, 0.02564102564102564, 0.0, 0.0], [0.359375, 0.09375, 0.0625, 0.046875, 0.328125, 0.03125, 0.046875, 0.03125, 0.0, 0.0], [0.5633802816901409, 0.08450704225352113, 0.08450704225352113, 0.056338028169014086, 0.056338028169014086, 0.04225352112676056, 0.11267605633802817, 0.0, 0.0, 0.0], [0.5263157894736842, 0.07894736842105263, 0.15789473684210525, 0.07894736842105263, 0.07894736842105263, 0.05263157894736842, 0.0, 0.02631578947368421, 0.0, 0.0], [0.6595744680851063, 0.1702127659574468, 0.06382978723404255, 0.02127659574468085, 0.0, 0.0851063829787234, 0.0, 0.0, 0.0, 0.0], [0.3684210526315789, 0.07894736842105263, 0.13157894736842105, 0.10526315789473684, 0.21052631578947367, 0.05263157894736842, 0.02631578947368421, 0.02631578947368421, 0.0, 0.0], [0.6153846153846154, 0.0, 0.11538461538461539, 0.07692307692307693, 0.11538461538461539, 0.07692307692307693, 0.0, 0.0, 0.0, 0.0], [0.6296296296296297, 0.037037037037037035, 0.1111111111111111, 0.07407407407407407, 0.07407407407407407, 0.0, 0.07407407407407407, 0.0, 0.0, 0.0], [0.6875, 0.09375, 0.0625, 0.03125, 0.03125, 0.03125, 0.0625, 0.0, 0.0, 0.0], [0.8658536585365854, 0.06097560975609756, 0.024390243902439025, 0.0, 0.036585365853658534, 0.012195121951219513, 0.0, 0.0, 0.0, 0.0], [0.5932203389830508, 0.11864406779661017, 0.11864406779661017, 0.05084745762711865, 0.06779661016949153, 0.05084745762711865, 0.0, 0.0, 0.0, 0.0], [0.47058823529411764, 0.17647058823529413, 0.1568627450980392, 0.058823529411764705, 0.09803921568627451, 0.0392156862745098, 0.0, 0.0, 0.0, 0.0], [0.5645161290322581, 0.14516129032258066, 0.1774193548387097, 0.016129032258064516, 0.06451612903225806, 0.016129032258064516, 0.016129032258064516, 0.0, 0.0, 0.0], [0.4056603773584906, 0.2830188679245283, 0.22641509433962265, 0.03773584905660377, 0.03773584905660377, 0.009433962264150943, 0.0, 0.0, 0.0, 0.0], [0.458281956432388, 0.24373201808466913, 0.24496506370735718, 0.006987258528565557, 0.04192355117139334, 0.004110152075626798, 0.0, 0.0, 0.0, 0.0], [0.49137931034482757, 0.2413793103448276, 0.21551724137931033, 0.008620689655172414, 0.03017241379310345, 0.01293103448275862, 0.0, 0.0, 0.0, 0.0], [0.4732142857142857, 0.17857142857142858, 0.25, 0.05357142857142857, 0.0, 0.044642857142857144, 0.0, 0.0, 0.0, 0.0], [0.42857142857142855, 0.23809523809523808, 0.2857142857142857, 0.0, 0.0, 0.0, 0.047619047619047616, 0.0, 0.0, 0.0], [0.42857142857142855, 0.21428571428571427, 0.14285714285714285, 0.07142857142857142, 0.10714285714285714, 0.03571428571428571, 0.0, 0.0, 0.0, 0.0], [0.5428571428571428, 0.11428571428571428, 0.08571428571428572, 0.14285714285714285, 0.05714285714285714, 0.05714285714285714, 0.0, 0.0, 0.0, 0.0], [0.4828571428571429, 0.22857142857142856, 0.24857142857142858, 0.017142857142857144, 0.02, 0.002857142857142857, 0.0, 0.0, 0.0, 0.0], [0.5625, 0.14583333333333334, 0.1875, 0.0625, 0.041666666666666664, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5945945945945946, 0.10810810810810811, 0.16216216216216217, 0.02702702702702703, 0.08108108108108109, 0.02702702702702703, 0.0, 0.0, 0.0, 0.0], [0.47101449275362317, 0.1956521739130435, 0.30434782608695654, 0.014492753623188406, 0.007246376811594203, 0.007246376811594203, 0.0, 0.0, 0.0, 0.0], [0.4444444444444444, 0.16666666666666666, 0.3888888888888889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.18181818181818182, 0.18181818181818182, 0.06818181818181818, 0.06818181818181818, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8432835820895522, 0.05970149253731343, 0.04477611940298507, 0.04477611940298507, 0.007462686567164179, 0.0, 0.0, 0.0, 0.0, 0.0], [0.625, 0.125, 0.08333333333333333, 0.125, 0.041666666666666664, 0.0, 0.0, 0.0, 0.0, 0.0], [0.013513513513513514, 0.0019305019305019305, 0.006756756756756757, 0.0, 0.0, 0.9777992277992278, 0.0, 0.0, 0.0, 0.0], [0.2077922077922078, 0.03896103896103896, 0.1038961038961039, 0.012987012987012988, 0.09090909090909091, 0.5454545454545454, 0.0, 0.0, 0.0, 0.0], [0.16304347826086957, 0.021739130434782608, 0.010869565217391304, 0.021739130434782608, 0.16304347826086957, 0.6195652173913043, 0.0, 0.0, 0.0, 0.0], [0.3225806451612903, 0.0, 0.03225806451612903, 0.0, 0.1935483870967742, 0.45161290322580644, 0.0, 0.0, 0.0, 0.0], [0.3611111111111111, 0.05555555555555555, 0.1111111111111111, 0.027777777777777776, 0.0, 0.4444444444444444, 0.0, 0.0, 0.0, 0.0], [0.5833333333333334, 0.08333333333333333, 0.16666666666666666, 0.08333333333333333, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0], [0.625, 0.125, 0.0, 0.1875, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0], [0.5384615384615384, 0.15384615384615385, 0.15384615384615385, 0.15384615384615385, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6666666666666666, 0.05555555555555555, 0.16666666666666666, 0.05555555555555555, 0.05555555555555555, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4666666666666667, 0.2, 0.2, 0.13333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5454545454545454, 0.18181818181818182, 0.0, 0.09090909090909091, 0.18181818181818182, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6, 0.06666666666666667, 0.13333333333333333, 0.13333333333333333, 0.06666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 0.16666666666666666, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4375, 0.25, 0.0625, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4782608695652174, 0.2608695652173913, 0.13043478260869565, 0.13043478260869565, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47058823529411764, 0.23529411764705882, 0.23529411764705882, 0.058823529411764705, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6111111111111112, 0.1111111111111111, 0.2777777777777778, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19444444444444445, 0.6944444444444444, 0.1111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.003631411711302769, 0.9464366772582842, 0.04993191103041307, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.046875, 0.90625, 0.046875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.14035087719298245, 0.8245614035087719, 0.03508771929824561, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2753623188405797, 0.6521739130434783, 0.057971014492753624, 0.014492753623188406, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5555555555555556, 0.4444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36666666666666664, 0.5666666666666667, 0.06666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.07407407407407407, 0.7407407407407407, 0.18518518518518517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4642857142857143, 0.5, 0.03571428571428571, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21875, 0.75, 0.03125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.46153846153846156, 0.038461538461538464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.3333333333333333, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8571428571428571, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.175, 0.8, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.6666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.5833333333333334, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6153846153846154, 0.3076923076923077, 0.07692307692307693, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3793103448275862, 0.6206896551724138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.32, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.38095238095238093, 0.6190476190476191, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0743801652892562, 0.9256198347107438, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16981132075471697, 0.8301886792452831, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1282051282051282, 0.8717948717948718, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7142857142857143, 0.2857142857142857, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
    1000: ([0.7001150687797479, 0.027437179171056467, 0.011559181965583974, 0.010273998191776318, 0.009235390374571294, 0.009272750367995936, 0.0064744868604903125, 0.0043076072418611255, 0.009048590407448088, 0.004053559286573565, 0.003844343323395575, 0.006347462882846533, 0.005488183034079786, 0.003956423303669499, 0.0034744793884916277, 0.0035006313838888766, 0.005667511002518064, 0.006403502872983495, 0.005405991048545575, 0.0034707433891491635, 0.004845591147175958, 0.011943989897857778, 0.003448327393094379, 0.004486935210299403, 0.003007479470683613, 0.002734751518683733, 0.002491911561423565, 0.006104622925586365, 0.004109599276710527, 0.004337495236600838, 0.003934007307614714, 0.0063213108874492835, 0.0018567916732046656, 0.0024358715712866036, 0.0024470795693139957, 0.006728534815777873, 0.0023760955818071775, 0.002734751518683733, 0.00428519124580634, 0.002719807521313876, 0.002417191574574283, 0.0021780876166565795, 0.0017708636883279909, 0.004176847264874882, 0.002009967646245694, 0.005009975118244379, 0.0017559196909581343, 0.0020697436357251203, 0.0012141997863008376, 0.0018642636718895938, 0.0018605276725471296, 0.0015056077350130387, 0.001382319756711723, 0.01796268483856747, 0.0021071036291497613, 0.0018904156672868426, 0.0014047357527665076, 0.0008107118573147131, 0.0011768397928761964, 0.0009601518310132777, 0.001486927738300718, 0.0013636397599994022, 0.0017073516995061008, 0.001867999671232058, 0.0011133278040543065, 0.001382319756711723, 0.0013038637705199763, 0.0012440877810405504, 0.0004333759237258374, 0.0007584078665202155, 0.0007173118737531102, 0.0007808238625750002, 0.0005865518967668662, 0.0006762158809860049, 0.0009825678270680624, 0.0005902878961093303, 0.000646327886246292, 0.000452055920438158, 0.00034371193950669864, 0.0003511839381916269, 0.0002166879618629187, 0.000291407948712201, 0.0002166879618629187, 0.00019427196580813403, 0.00022042396120538284, 0.0002166879618629187, 0.00013449597632870817, 0.00014943997369856464, 0.00024657595660263165, 0.00019800796515059813, 0.0004744719164929427, 0.00020174396449306226, 0.0001419679750136364, 7.845598619174644e-05, 1.4943997369856463e-05, 0.0, 0.0, 0.0, 0.0, 0.0] ,
    [[0.28049008255201524, 0.14257432082690757, 0.09403565692086853, 0.11639460610361958, 0.1096922576135157, 0.07404067300970667, 0.0654012604257272, 0.06822948072807995, 0.038837334642496946, 0.0103043271770626], [0.15631808278867101, 0.11587690631808278, 0.11301742919389979, 0.12023420479302832, 0.10620915032679738, 0.1090686274509804, 0.12295751633986927, 0.08946078431372549, 0.05664488017429194, 0.010212418300653595], [0.19553975436328377, 0.11667744020685197, 0.09340659340659341, 0.13994828700711054, 0.10795087265675501, 0.09631544925662573, 0.09663865546218488, 0.083710407239819, 0.06593406593406594, 0.003878474466709761], [0.18727272727272729, 0.1789090909090909, 0.11709090909090909, 0.10472727272727272, 0.10909090909090909, 0.08945454545454545, 0.08472727272727272, 0.0669090909090909, 0.06036363636363636, 0.0014545454545454545], [0.16747572815533981, 0.13794498381877024, 0.16464401294498382, 0.10558252427184465, 0.10072815533980582, 0.12378640776699029, 0.09344660194174757, 0.059870550161812294, 0.043284789644012944, 0.003236245954692557], [0.30016116035455276, 0.11281224818694602, 0.09307010475423046, 0.11281224818694602, 0.10072522159548751, 0.03585817888799355, 0.0427074939564867, 0.14020950846091862, 0.06164383561643835, 0.0], [0.31332948643969993, 0.16387766878245816, 0.09924985574148874, 0.1257934218118869, 0.0738603577611079, 0.06347374495095211, 0.06462781304096941, 0.06520484708597807, 0.030582804385458743, 0.0], [0.2532523850823938, 0.15871639202081528, 0.11795316565481354, 0.1222896790980052, 0.08933217692974849, 0.0797918473547268, 0.07198612315698179, 0.06851691240242845, 0.03816131830008673, 0.0], [0.4240297274979356, 0.1494632535094963, 0.07679603633360858, 0.061932287365813375, 0.08133773740710157, 0.06358381502890173, 0.061932287365813375, 0.06358381502890173, 0.017341040462427744, 0.0], [0.2672811059907834, 0.11705069124423963, 0.07373271889400922, 0.2184331797235023, 0.07096774193548387, 0.09861751152073733, 0.05806451612903226, 0.08294930875576037, 0.012903225806451613, 0.0], [0.27405247813411077, 0.2031098153547133, 0.12147716229348883, 0.11661807580174927, 0.07580174927113703, 0.07094266277939747, 0.08357628765792031, 0.04664723032069971, 0.007774538386783284, 0.0], [0.23366686286050617, 0.20247204237786934, 0.16303708063566805, 0.125956444967628, 0.09535020600353149, 0.051206592113007654, 0.051206592113007654, 0.07474985285462037, 0.002354326074161271, 0.0], [0.21238938053097345, 0.22600408441116407, 0.11572498298162015, 0.12321307011572498, 0.08032675289312458, 0.06943498978897208, 0.05037440435670524, 0.1191286589516678, 0.0034036759700476512, 0.0], [0.251180358829084, 0.11614730878186968, 0.1643059490084986, 0.11803588290840415, 0.10009442870632672, 0.10953729933899906, 0.0679886685552408, 0.07176581680830972, 0.0009442870632672333, 0.0], [0.2752688172043011, 0.14623655913978495, 0.13870967741935483, 0.1064516129032258, 0.10967741935483871, 0.08494623655913978, 0.1010752688172043, 0.03763440860215054, 0.0, 0.0], [0.23372465314834578, 0.25827107790821774, 0.16542155816435433, 0.10779082177161152, 0.08858057630736393, 0.09284951974386339, 0.0288153681963714, 0.02454642475987193, 0.0, 0.0], [0.4495715227422544, 0.15095583388266315, 0.13381674357284112, 0.1041529334212261, 0.06394199077125906, 0.04087013843111404, 0.02109426499670402, 0.035596572181938034, 0.0, 0.0], [0.470828471411902, 0.14935822637106183, 0.1161026837806301, 0.11901983663943991, 0.05542590431738623, 0.044924154025670945, 0.028004667444574097, 0.01633605600933489, 0.0, 0.0], [0.38147892190739463, 0.18659295093296477, 0.13476157567380787, 0.13890808569454044, 0.061506565307532825, 0.038700760193503804, 0.04008293020041465, 0.01796821008984105, 0.0, 0.0], [0.333692142088267, 0.15715823466092574, 0.13240043057050593, 0.14531754574811626, 0.09041980624327234, 0.060279870828848225, 0.05597416576964478, 0.024757804090419805, 0.0, 0.0], [0.3053199691595991, 0.17733230531996916, 0.15497301464919044, 0.13338473400154202, 0.08172706245181187, 0.059367771781033155, 0.06630686198920586, 0.02158828064764842, 0.0, 0.0], [0.4460431654676259, 0.23052862058179543, 0.2108226462308414, 0.030028151391929936, 0.05223647169221145, 0.018454801376290273, 0.009696590553644042, 0.002189552705661558, 0.0, 0.0], [0.4452871072589382, 0.22318526543878656, 0.11592632719393282, 0.07258938244853738, 0.0628385698808234, 0.044420368364030335, 0.030335861321776816, 0.005417118093174431, 0.0, 0.0], [0.3996669442131557, 0.1340549542048293, 0.1340549542048293, 0.16652789342214822, 0.06078268109908409, 0.058284762697751874, 0.03996669442131557, 0.006661115736885929, 0.0, 0.0], [0.5142857142857142, 0.16024844720496895, 0.09192546583850932, 0.11055900621118013, 0.05217391304347826, 0.05093167701863354, 0.017391304347826087, 0.002484472049689441, 0.0, 0.0], [0.4849726775956284, 0.16666666666666666, 0.11065573770491803, 0.09699453551912568, 0.07923497267759563, 0.04371584699453552, 0.017759562841530054, 0.0, 0.0, 0.0], [0.31634182908545727, 0.25487256371814093, 0.15142428785607195, 0.08695652173913043, 0.10944527736131934, 0.053973013493253376, 0.026986506746626688, 0.0, 0.0, 0.0], [0.3543451652386781, 0.19033047735618114, 0.2331701346389229, 0.06731946144430845, 0.0966952264381885, 0.04528763769889841, 0.012851897184822521, 0.0, 0.0, 0.0], [0.41818181818181815, 0.24363636363636362, 0.11636363636363636, 0.07818181818181819, 0.08727272727272728, 0.03636363636363636, 0.02, 0.0, 0.0, 0.0], [0.4427217915590009, 0.17571059431524547, 0.11800172265288544, 0.09043927648578812, 0.12661498708010335, 0.03359173126614987, 0.012919896640826873, 0.0, 0.0, 0.0], [0.4254510921177588, 0.15954415954415954, 0.15194681861348527, 0.15384615384615385, 0.05887939221272555, 0.0389363722697056, 0.011396011396011397, 0.0, 0.0, 0.0], [0.3747044917257683, 0.15780141843971632, 0.208628841607565, 0.17848699763593381, 0.031323877068557916, 0.04196217494089834, 0.0070921985815602835, 0.0, 0.0, 0.0], [0.4164989939637827, 0.22334004024144868, 0.13279678068410464, 0.12474849094567404, 0.05432595573440644, 0.0482897384305835, 0.0, 0.0, 0.0, 0.0], [0.5153374233128835, 0.1579754601226994, 0.09815950920245399, 0.09355828220858896, 0.07975460122699386, 0.05521472392638037, 0.0, 0.0, 0.0, 0.0], [0.35877862595419846, 0.24580152671755726, 0.16030534351145037, 0.13129770992366413, 0.07633587786259542, 0.02748091603053435, 0.0, 0.0, 0.0, 0.0], [0.1799000555247085, 0.09550249861188229, 0.03498056635202665, 0.06718489727928928, 0.03720155469183787, 0.5852304275402554, 0.0, 0.0, 0.0, 0.0], [0.449685534591195, 0.21540880503144655, 0.08490566037735849, 0.08962264150943396, 0.07075471698113207, 0.08962264150943396, 0.0, 0.0, 0.0, 0.0], [0.43169398907103823, 0.20765027322404372, 0.09153005464480875, 0.08196721311475409, 0.09153005464480875, 0.09562841530054644, 0.0, 0.0, 0.0, 0.0], [0.5710549258936356, 0.1918047079337402, 0.05579773321708806, 0.07585004359197908, 0.08020924149956409, 0.025283347863993024, 0.0, 0.0, 0.0, 0.0], [0.635989010989011, 0.18269230769230768, 0.07280219780219781, 0.04395604395604396, 0.027472527472527472, 0.03708791208791209, 0.0, 0.0, 0.0, 0.0], [0.41731066460587324, 0.15765069551777433, 0.12210200927357033, 0.2009273570324575, 0.080370942812983, 0.021638330757341576, 0.0, 0.0, 0.0, 0.0], [0.5248713550600344, 0.19897084048027444, 0.09605488850771869, 0.0686106346483705, 0.09777015437392796, 0.0137221269296741, 0.0, 0.0, 0.0, 0.0], [0.5443037974683544, 0.22784810126582278, 0.08438818565400844, 0.08438818565400844, 0.052742616033755275, 0.006329113924050633, 0.0, 0.0, 0.0, 0.0], [0.4347048300536673, 0.3148479427549195, 0.06708407871198568, 0.10554561717352415, 0.07334525939177101, 0.004472271914132379, 0.0, 0.0, 0.0, 0.0], [0.4739776951672863, 0.19516728624535315, 0.1449814126394052, 0.10408921933085502, 0.07992565055762081, 0.0018587360594795538, 0.0, 0.0, 0.0, 0.0], [0.6331096196868009, 0.16405667412378822, 0.0947054436987323, 0.07755406413124534, 0.030574198359433258, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4978723404255319, 0.17446808510638298, 0.1148936170212766, 0.15319148936170213, 0.059574468085106386, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5595667870036101, 0.17148014440433212, 0.13357400722021662, 0.09025270758122744, 0.04512635379061372, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6461538461538462, 0.20615384615384616, 0.04923076923076923, 0.07076923076923076, 0.027692307692307693, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7154308617234469, 0.1683366733466934, 0.05410821643286573, 0.04609218436873747, 0.01603206412825651, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6265060240963856, 0.20883534136546184, 0.09839357429718876, 0.05823293172690763, 0.008032128514056224, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5880893300248139, 0.17866004962779156, 0.15384615384615385, 0.07692307692307693, 0.0024813895781637717, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6702702702702703, 0.1864864864864865, 0.07297297297297298, 0.05675675675675676, 0.013513513513513514, 0.0, 0.0, 0.0, 0.0, 0.0], [0.06405990016638935, 0.8810316139767055, 0.050124792013311145, 0.00478369384359401, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.549645390070922, 0.29609929078014185, 0.08865248226950355, 0.05673758865248227, 0.008865248226950355, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5454545454545454, 0.21739130434782608, 0.16996047430830039, 0.06719367588932806, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.651595744680851, 0.25, 0.07180851063829788, 0.026595744680851064, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7419354838709677, 0.17050691244239632, 0.055299539170506916, 0.03225806451612903, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7206349206349206, 0.1873015873015873, 0.07301587301587302, 0.01904761904761905, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6536964980544747, 0.245136186770428, 0.08949416342412451, 0.011673151750972763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8140703517587939, 0.12562814070351758, 0.05025125628140704, 0.010050251256281407, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6356164383561644, 0.29315068493150687, 0.06575342465753424, 0.005479452054794521, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7177242888402626, 0.19693654266958424, 0.0700218818380744, 0.015317286652078774, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.704, 0.232, 0.062, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7684563758389261, 0.17114093959731544, 0.05704697986577181, 0.003355704697986577, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.754054054054054, 0.2, 0.04594594594594595, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.504297994269341, 0.45272206303724927, 0.04297994269340974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6456456456456456, 0.2972972972972973, 0.057057057057057055, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6896551724137931, 0.2413793103448276, 0.06896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7931034482758621, 0.16748768472906403, 0.03940886699507389, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.234375, 0.015625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6985645933014354, 0.2966507177033493, 0.004784688995215311, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8089171974522293, 0.18471337579617833, 0.006369426751592357, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7458563535911602, 0.2541436464088398, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.41064638783269963, 0.5817490494296578, 0.0076045627376425855, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5759493670886076, 0.4240506329113924, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6416184971098265, 0.3583815028901734, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.768595041322314, 0.23140495867768596, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7282608695652174, 0.2717391304347826, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8936170212765957, 0.10638297872340426, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8793103448275862, 0.1206896551724138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9102564102564102, 0.08974358974358974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9310344827586207, 0.06896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9615384615384616, 0.038461538461538464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9830508474576272, 0.01694915254237288, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
    2000: ([0.5907305662643846, 0.025437664966721722, 0.013383674377626577, 0.014074748128944955, 0.011697599151896925, 0.010655561244149844, 0.010324497117209364, 0.008801240314565193, 0.008140921154274073, 0.011657799092920473, 0.005572008256703144, 0.007533065708088277, 0.008039611913243108, 0.005638944719527176, 0.0051287803271926665, 0.005479744483621371, 0.007612665826041178, 0.006339063938794746, 0.005032898366931216, 0.005761963083636206, 0.005704072088761368, 0.013132210368639001, 0.006917973887543124, 0.00855701267993697, 0.004774197983584285, 0.0040831242322659075, 0.006402382214439099, 0.008122830218375687, 0.004470270260491386, 0.004432279295104774, 0.004499215757928805, 0.0073015017285889255, 0.00506908023872799, 0.0033034048950454356, 0.00335225042197108, 0.005505071793879113, 0.003169531969397373, 0.002767913192453185, 0.005669699310554433, 0.004790479825892833, 0.003109831880932696, 0.004379815580999452, 0.002968722580925279, 0.004871889037435574, 0.002695549448859638, 0.004005333207902844, 0.0030067135463118914, 0.0030664136347765677, 0.0030700318219562452, 0.005161344011809763, 0.006512736923419259, 0.0033088321758149514, 0.0031261137232412445, 0.011947254067294664, 0.0027480131629649598, 0.003328732205303177, 0.003410141416845918, 0.0027480131629649598, 0.0022939306719154504, 0.002328303450122385, 0.0022686033616577087, 0.002152821371908033, 0.002454940001411093, 0.0065615824503449036, 0.001861557303944005, 0.0041482516015001, 0.0018181390577878766, 0.0025218764642351245, 0.0014309930295623983, 0.0009208286372278898, 0.002020757539849809, 0.001342347443660303, 0.0010637470308251457, 0.0015938114526478798, 0.0012537018577582074, 0.000919019543638051, 0.0011741017398053054, 0.0006784100961895062, 0.0006277554756740231, 0.0004504643038698321, 0.0004088551513035424, 0.000548155357721121, 0.0005499644513109596, 0.0004378006487409613, 0.0004088551513035424, 0.00044684611669015476, 0.0003401095948896724, 0.00036724599873725266, 0.00031478228463193087, 0.0001917639205229004, 0.00035639143719822056, 0.0003618187179677366, 0.0002442276346282222, 0.00028221860001483457, 0.0002532731025774156, 5.2463714105321814e-05, 1.0854561539032099e-05, 5.4272807695160494e-06, 0.0, 0.0] ,
    [[0.17727097331365188, 0.1052815327040982, 0.07265705868301617, 0.10331542810243344, 0.09993140071171762, 0.09646468667887571, 0.13485578837119566, 0.10131869881849975, 0.07412704343192439, 0.03477738918458721], [0.096436953275016, 0.07709266766232843, 0.07488798805205889, 0.09352108669369177, 0.09778820852001992, 0.168764668231278, 0.17416968921129364, 0.10596685868714885, 0.08171538297418392, 0.029656496692980584], [0.10935387942687212, 0.08556366585563666, 0.07610164909434983, 0.12287104622871046, 0.11016490943498243, 0.12124898621248986, 0.14233576642335766, 0.10043254933765883, 0.11192214111922141, 0.020005406866720737], [0.08637532133676093, 0.093573264781491, 0.07197943444730077, 0.08341902313624679, 0.08933161953727506, 0.24344473007712084, 0.16915167095115682, 0.07159383033419023, 0.07904884318766067, 0.012082262210796915], [0.09078255490256727, 0.0867615218063718, 0.08335910918651407, 0.08660686668728734, 0.10980513454995361, 0.25394370553665324, 0.14676770801113517, 0.07036807918342097, 0.06263532322919889, 0.008969996906897619], [0.15042444821731749, 0.14159592529711376, 0.07538200339558573, 0.14634974533106962, 0.10764006791171477, 0.10424448217317488, 0.10594227504244483, 0.10203735144312394, 0.06485568760611206, 0.0015280135823429542], [0.1801296653232872, 0.1766251971263361, 0.07552128964429648, 0.13930261082880674, 0.07394427895566848, 0.11109164184335027, 0.1121429823024356, 0.08393201331697915, 0.0466094270194498, 0.0007008936393902226], [0.15847893114080164, 0.12702980472764647, 0.06413155190133607, 0.1358684480986639, 0.08859198355601233, 0.14779033915724563, 0.10298047276464542, 0.11963001027749229, 0.0552929085303186, 0.00020554984583761563], [0.25955555555555554, 0.12, 0.08755555555555555, 0.07777777777777778, 0.11022222222222222, 0.10733333333333334, 0.11044444444444444, 0.0908888888888889, 0.036222222222222225, 0.0], [0.07526381129733085, 0.05819366852886406, 0.04903786468032278, 0.12011173184357542, 0.09279950341402855, 0.38687150837988826, 0.13656114214773432, 0.0521415270018622, 0.029019242706393545, 0.0], [0.12175324675324675, 0.09480519480519481, 0.08149350649350649, 0.09707792207792208, 0.1331168831168831, 0.23214285714285715, 0.13116883116883116, 0.07857142857142857, 0.02987012987012987, 0.0], [0.12079731027857829, 0.13328530259365995, 0.10470701248799232, 0.1186359269932757, 0.13280499519692604, 0.18419788664745437, 0.09486071085494717, 0.09462055715658022, 0.016090297790585975, 0.0], [0.09473447344734473, 0.10531053105310531, 0.06233123312331233, 0.08303330333033303, 0.10306030603060307, 0.3892889288928893, 0.07875787578757876, 0.06975697569756975, 0.013726372637263727, 0.0], [0.12447866538338145, 0.0792428617260186, 0.08533846647417388, 0.10362528071863972, 0.15527751042669233, 0.25665704202759065, 0.1017003529034328, 0.08020532563362208, 0.013474494706448507, 0.0], [0.14038800705467372, 0.1001763668430335, 0.09559082892416226, 0.13686067019400353, 0.22257495590828924, 0.12310405643738977, 0.09664902998236331, 0.07513227513227513, 0.009523809523809525, 0.0], [0.11125784087157478, 0.13898976559920767, 0.10762627930009905, 0.18586992406734895, 0.12545394519643446, 0.1914823374050842, 0.09210960713106636, 0.04522944866292506, 0.0019808517662594917, 0.0], [0.18512357414448669, 0.08555133079847908, 0.08531368821292776, 0.09149239543726236, 0.11692015209125475, 0.3341254752851711, 0.05869771863117871, 0.04230038022813688, 0.0004752851711026616, 0.0], [0.2585616438356164, 0.13641552511415525, 0.1021689497716895, 0.12557077625570776, 0.09674657534246575, 0.16666666666666666, 0.07248858447488585, 0.04138127853881279, 0.0, 0.0], [0.23112868439971243, 0.13407620416966212, 0.12437095614665708, 0.15240833932422718, 0.13227893601725377, 0.10460100647016535, 0.08087706685837527, 0.040258806613946804, 0.0, 0.0], [0.15384615384615385, 0.13783359497645212, 0.1315541601255887, 0.1497645211930926, 0.1792778649921507, 0.11397174254317112, 0.09105180533751962, 0.04270015698587127, 0.0, 0.0], [0.18553758325404376, 0.13606089438629876, 0.11988582302568981, 0.13574373612432603, 0.11195686647637171, 0.15667618141452586, 0.10244211861718998, 0.051696796701554075, 0.0, 0.0], [0.21201267392202783, 0.13142306102768977, 0.11378977820636452, 0.0469761675161868, 0.1210910593745695, 0.3340680534508885, 0.031271525003444, 0.00936768149882904, 0.0, 0.0], [0.12578451882845187, 0.07479079497907949, 0.07322175732217573, 0.09309623430962342, 0.16422594142259414, 0.40193514644351463, 0.05543933054393305, 0.011506276150627616, 0.0, 0.0], [0.1843551797040169, 0.080338266384778, 0.14482029598308668, 0.1828752642706131, 0.18520084566596196, 0.16405919661733614, 0.048625792811839326, 0.009725158562367865, 0.0, 0.0], [0.17999242137173171, 0.10155361879499811, 0.0799545282303903, 0.12277377794619174, 0.13338385752178855, 0.3008715422508526, 0.07616521409624857, 0.005305039787798408, 0.0, 0.0], [0.20115197164377493, 0.1227292866637129, 0.09924678777137794, 0.1546300398759415, 0.14754098360655737, 0.19893664155959237, 0.07221976074435091, 0.003544528134692069, 0.0, 0.0], [0.18310257134783836, 0.14665159649618537, 0.10285391353489687, 0.160497315625883, 0.15851935575021192, 0.17632099463125175, 0.07035885843458604, 0.0016953941791466515, 0.0, 0.0], [0.1866369710467706, 0.1510022271714922, 0.17527839643652562, 0.13140311804008908, 0.1271714922048998, 0.1861915367483296, 0.042316258351893093, 0.0, 0.0, 0.0], [0.22784297855119384, 0.1906110886280858, 0.11493322541481182, 0.12788344799676243, 0.15783083771752326, 0.12343180898421692, 0.05746661270740591, 0.0, 0.0, 0.0], [0.26816326530612244, 0.14775510204081632, 0.13183673469387755, 0.13387755102040816, 0.15224489795918367, 0.1126530612244898, 0.053469387755102044, 0.0, 0.0, 0.0], [0.2513067953357459, 0.13791716928025735, 0.15520707679935666, 0.1467631684760756, 0.12625653397667874, 0.08805790108564536, 0.09449135504624046, 0.0, 0.0, 0.0], [0.26734390485629334, 0.13974231912784935, 0.19672943508424182, 0.14246778989098116, 0.10431119920713577, 0.11546085232903865, 0.03394449950445986, 0.0, 0.0, 0.0], [0.2259100642398287, 0.21020699500356888, 0.15060670949321914, 0.12919343326195576, 0.08493932905067808, 0.14953604568165596, 0.0496074232690935, 0.0, 0.0, 0.0], [0.27546549835706463, 0.17031763417305587, 0.13088718510405256, 0.1566265060240964, 0.10350492880613363, 0.13964950711938665, 0.023548740416210297, 0.0, 0.0, 0.0], [0.20561252023745277, 0.20021586616297896, 0.1651376146788991, 0.15434430652995143, 0.11980572045331894, 0.13491635186184567, 0.019967620075553156, 0.0, 0.0, 0.0], [0.15018074268813672, 0.10417351298061124, 0.0782122905027933, 0.14393690437068682, 0.10187315149523496, 0.4170226749917844, 0.004600722970752547, 0.0, 0.0, 0.0], [0.29337899543379, 0.1969178082191781, 0.1506849315068493, 0.136986301369863, 0.08789954337899543, 0.1329908675799087, 0.001141552511415525, 0.0, 0.0, 0.0], [0.26405228758169935, 0.2169934640522876, 0.14444444444444443, 0.1392156862745098, 0.12026143790849673, 0.11437908496732026, 0.00065359477124183, 0.0, 0.0, 0.0], [0.3283343969368219, 0.2131461391193363, 0.10146777281429484, 0.104977664326739, 0.13146139119336311, 0.1206126356094448, 0.0, 0.0, 0.0, 0.0], [0.270392749244713, 0.26132930513595165, 0.1506797583081571, 0.1952416918429003, 0.08496978851963746, 0.037386706948640484, 0.0, 0.0, 0.0, 0.0], [0.2716695753344968, 0.19313554392088422, 0.16870273414776032, 0.1820826061663758, 0.14485165794066318, 0.03955788248981966, 0.0, 0.0, 0.0, 0.0], [0.24576621230896323, 0.18380834365964477, 0.1697645600991326, 0.1416769929781082, 0.22057001239157373, 0.03841387856257745, 0.0, 0.0, 0.0, 0.0], [0.2687385740402194, 0.238878732480195, 0.17184643510054845, 0.12797074954296161, 0.15843997562461914, 0.03412553321145643, 0.0, 0.0, 0.0, 0.0], [0.331229112513925, 0.25547716301522466, 0.1318232454511697, 0.16450055699962868, 0.10434459710360193, 0.012625324916450055, 0.0, 0.0, 0.0, 0.0], [0.38456375838926177, 0.21946308724832214, 0.17248322147651007, 0.09731543624161074, 0.11006711409395974, 0.016107382550335572, 0.0, 0.0, 0.0, 0.0], [0.46973803071364045, 0.1996386630532972, 0.13504968383017163, 0.11517615176151762, 0.07768744354110207, 0.0027100271002710027, 0.0, 0.0, 0.0, 0.0], [0.34416365824308065, 0.2352587244283995, 0.1690734055354994, 0.1371841155234657, 0.11311672683513839, 0.0012033694344163659, 0.0, 0.0, 0.0, 0.0], [0.36932153392330386, 0.25309734513274335, 0.20530973451327433, 0.10619469026548672, 0.06607669616519174, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4295816146140248, 0.26222746022392457, 0.16087212728344136, 0.07719505008839128, 0.07012374779021803, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44970206799859797, 0.2576235541535226, 0.18226428321065544, 0.07045215562565721, 0.03995793901156677, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44055555555555553, 0.26805555555555555, 0.17666666666666667, 0.06944444444444445, 0.04527777777777778, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36686714051394204, 0.3225806451612903, 0.17495899398578457, 0.0989611809732094, 0.036632039365773646, 0.0, 0.0, 0.0, 0.0, 0.0], [0.41550925925925924, 0.2650462962962963, 0.1909722222222222, 0.0931712962962963, 0.03530092592592592, 0.0, 0.0, 0.0, 0.0, 0.0], [0.15006056935190792, 0.7204724409448819, 0.08918837068443368, 0.03255602665051484, 0.007722592368261659, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4575378538512179, 0.29295589203423306, 0.15273206056616195, 0.08426596445029624, 0.012508229098090849, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45163043478260867, 0.2668478260869565, 0.18858695652173912, 0.08641304347826087, 0.006521739130434782, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4742705570291777, 0.30875331564986735, 0.1336870026525199, 0.08063660477453581, 0.002652519893899204, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4970375246872943, 0.2705727452271231, 0.14944042132982224, 0.08294930875576037, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44873817034700314, 0.2941640378548896, 0.19085173501577288, 0.06624605678233439, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5042735042735043, 0.2867132867132867, 0.15384615384615385, 0.05516705516705517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5829346092503987, 0.2511961722488038, 0.12280701754385964, 0.0430622009569378, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5016806722689076, 0.2991596638655462, 0.16134453781512606, 0.037815126050420166, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5630066322770818, 0.26897568165070007, 0.1156963890935888, 0.05232129697862933, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7267714364488558, 0.17866004962779156, 0.08243727598566308, 0.01213123793768955, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.577259475218659, 0.2108843537414966, 0.2001943634596696, 0.011661807580174927, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7104230266027038, 0.22328826864369822, 0.0623637156563454, 0.0039249890972525075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5890547263681593, 0.3154228855721393, 0.08955223880597014, 0.005970149253731343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6836441893830703, 0.24390243902439024, 0.07245337159253945, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7370417193426043, 0.168141592920354, 0.09481668773704172, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7210216110019646, 0.18860510805500982, 0.09037328094302555, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5058191584601611, 0.4386750223813787, 0.05550581915846016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5390835579514824, 0.4110512129380054, 0.04986522911051213, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7091836734693877, 0.21768707482993196, 0.07312925170068027, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6401816118047673, 0.2542565266742338, 0.10556186152099886, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5685425685425686, 0.38095238095238093, 0.050505050505050504, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6850393700787402, 0.2992125984251969, 0.015748031496062992, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6040061633281972, 0.39599383667180277, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6826666666666666, 0.31733333333333336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7550432276657061, 0.24495677233429394, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7991967871485943, 0.20080321285140562, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7920353982300885, 0.2079646017699115, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8415841584158416, 0.15841584158415842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8585526315789473, 0.14144736842105263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9090909090909091, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9557522123893806, 0.04424778761061947, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9878542510121457, 0.012145748987854251, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9787234042553191, 0.02127659574468085, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
    math.inf: ([0.4809035509952379, 0.020085691527630272, 0.012774504094008017, 0.012275600397409983, 0.010806725136181439, 0.009873154270444346, 0.009714704169378875, 0.00885714652780157, 0.008742591387166399, 0.010587250334029942, 0.0064065230052417, 0.007512461886327041, 0.008103437938949604, 0.006079987323991915, 0.005651743807598753, 0.005929031484463325, 0.007176290725958409, 0.00634014526020076, 0.006369051697557299, 0.006415087875569564, 0.006027527493233753, 0.010887020795505156, 0.006490030490938367, 0.00808737880708486, 0.004848787214361575, 0.005007237315427044, 0.006177412723971359, 0.007052100106204392, 0.0051453458494638395, 0.005013660968172942, 0.005648531981225805, 0.007048888279831443, 0.0051603343725376, 0.004523322141902771, 0.004953706875877899, 0.00563568467573401, 0.00416895063208743, 0.004004076878276063, 0.005438692658193155, 0.006563902497516188, 0.004282435163931618, 0.004808104080304224, 0.0044751447463085405, 0.005751310425160163, 0.004868058172599267, 0.005177464113193326, 0.004038336359587516, 0.005343408475795676, 0.004456944396861832, 0.005487940662578368, 0.0071345369831100755, 0.0053327023878858475, 0.004663571893521532, 0.009633337901264175, 0.0039955120079482, 0.004236398985919353, 0.004445167700161019, 0.004938718352804138, 0.0037267892014114907, 0.004333824385898797, 0.003954828873890849, 0.004349883517763541, 0.0037846020761245673, 0.007047817671040461, 0.003934487306862174, 0.005066120798931104, 0.0047684915550378565, 0.004536169447394566, 0.0040244184453047385, 0.0030458820103463633, 0.003787813902497516, 0.0034826903970673886, 0.0036604114563705504, 0.0036507759772517042, 0.007105630545753537, 0.0030137637466168764, 0.004767420946246874, 0.004019065401349824, 0.0035726215355099526, 0.00342059508719038, 0.00427922333755867, 0.008244758299359348, 0.004000865051903114, 0.0026518979752646545, 0.002993422179588201, 0.0029687981773955942, 0.0020084620918839288, 0.002186183151187091, 0.0031101185378053374, 0.0020138151358388436, 0.0014731576963924766, 0.0014763695227654253, 0.0011733872349172633, 0.001200152454691836, 0.0038509798211655076, 0.0007601322415978622, 0.0004774915207783754, 0.00022268662852444415, 0.0002837113296104697, 1.391791428277776e-05] ,
    [[0.1294894999187421, 0.07726403480065096, 0.05391963703312874, 0.07717943751711426, 0.07625999861972853, 0.07821018862967985, 0.11584262233769009, 0.11426421512643954, 0.13598790704094285, 0.1415824589758831], [0.07302382602206706, 0.05825915462928415, 0.056713394808379086, 0.07254410745695858, 0.07718138691967379, 0.13400138585363253, 0.14663397473482223, 0.12312776504450722, 0.14897926549757476, 0.10953573903310058], [0.06830372108615487, 0.054056319141803554, 0.048273550117331546, 0.08112638283607107, 0.07609788803218237, 0.08959101575595038, 0.12068387529332886, 0.1376131411330875, 0.20549782098558497, 0.11875628561850486], [0.060788417931275075, 0.06480027908599337, 0.0514564800279086, 0.061660561660561664, 0.06584685156113727, 0.18001046572475143, 0.14207221350078492, 0.11372754229897088, 0.17259724402581544, 0.08703994418280132], [0.0592431147216168, 0.057261739647315235, 0.05646918961759461, 0.05934218347533188, 0.07975034674063801, 0.17832375668714087, 0.13384188626907073, 0.12026946701010502, 0.18040420051515751, 0.07509411531602933], [0.09694209499024073, 0.09260464107568857, 0.0495554109737584, 0.09780958577315116, 0.07720667967902842, 0.08338755150726523, 0.11515940143135979, 0.15256994144437216, 0.186402081977879, 0.04836261114725656], [0.11428256557196385, 0.11351113070310778, 0.050694291381970465, 0.09323341415031959, 0.0588494599955918, 0.09367423407538021, 0.1297112629490853, 0.1543971787524796, 0.16398501212254793, 0.027661450297553448], [0.09488698174785447, 0.07530520971836094, 0.03988879487489423, 0.0872718481808292, 0.06333857125589266, 0.11434787864136348, 0.12317176356823402, 0.17998307748096218, 0.19448809379910553, 0.027317780732503325], [0.14511388684790597, 0.06931178055351457, 0.05241244183198628, 0.05106539309331374, 0.07323046779328925, 0.08474161156012736, 0.12356110702914523, 0.16176830761694833, 0.23059025226549107, 0.008204751408278227], [0.04985337243401759, 0.03913439174840732, 0.033977146324198605, 0.08403276367681262, 0.07098796642734351, 0.27960359995955103, 0.13611083021539083, 0.13166144200626959, 0.17029022145818587, 0.004348265749823036], [0.06366978609625669, 0.05080213903743316, 0.0456216577540107, 0.05815508021390375, 0.08572860962566844, 0.15641711229946523, 0.1467245989304813, 0.1869986631016043, 0.20588235294117646, 0.0], [0.07325067692746187, 0.08108878438078951, 0.06484252529571041, 0.07738349722103463, 0.09248966794926607, 0.14336611087359272, 0.13253527148353997, 0.19695026364543253, 0.13809320222317228, 0.0], [0.05786761791518034, 0.06420927467300833, 0.04610912934337429, 0.05535737878187343, 0.07477870260272163, 0.2732197119830889, 0.16356189721231337, 0.1717532038578412, 0.09314308363059849, 0.0], [0.07078711040676175, 0.047015319598520865, 0.053530551153372075, 0.06356752949462934, 0.10142630744849446, 0.18665257967952104, 0.17010036978341259, 0.20073956682514527, 0.10618066561014262, 0.0], [0.07804508429626823, 0.056260655427164234, 0.05493464671339269, 0.08107596135631748, 0.1399886342110248, 0.10930100397802614, 0.1602576245501042, 0.21784428869103997, 0.10229210077666225, 0.0], [0.0628385698808234, 0.07818707114481763, 0.06229685807150596, 0.11213434452871072, 0.0917298663777537, 0.15167930660888407, 0.158540989526905, 0.21271217045864932, 0.06988082340195016, 0.0], [0.11741011487393704, 0.054751603759510666, 0.058332090108906456, 0.06698493211994629, 0.09145158884081754, 0.2579442040877219, 0.14724750111890197, 0.17246009249589736, 0.033417872594360735, 0.0], [0.15653495440729484, 0.08308004052684904, 0.06484295845997974, 0.08054711246200608, 0.08156028368794327, 0.1534954407294833, 0.16312056737588654, 0.19030732860520094, 0.0265113137453563, 0.0], [0.11363254328458565, 0.07060010085728693, 0.06791057320558078, 0.08959488989746175, 0.0926206085056312, 0.12909732728189613, 0.15985879979828543, 0.23802319717599596, 0.03866195999327618, 0.0], [0.08327770360480641, 0.07843791722296395, 0.07726969292389853, 0.09279038718291055, 0.14152202937249667, 0.1620493991989319, 0.18791722296395194, 0.17473297730307077, 0.0020026702269692926, 0.0], [0.10621669626998224, 0.07815275310834814, 0.07300177619893428, 0.08632326820603908, 0.0886323268206039, 0.15683836589698047, 0.19236234458259324, 0.21847246891651864, 0.0, 0.0], [0.1534074146917101, 0.09666633887304553, 0.08683253023896155, 0.04798898613432983, 0.10787688071590126, 0.29117907365522666, 0.11544891336414594, 0.10059986232667913, 0.0, 0.0], [0.08215110524579347, 0.04932365555922138, 0.05328274496865721, 0.07406796436819532, 0.1369185087429891, 0.3160673045199604, 0.17387000989772353, 0.11431870669745958, 0.0, 0.0], [0.12205454064072015, 0.05599682287529786, 0.09729944400317712, 0.12337834259994705, 0.14283823140058247, 0.16150383902568174, 0.16123907863383638, 0.1356897008207572, 0.0, 0.0], [0.1086332523735924, 0.06314859792448664, 0.05409582689335394, 0.08854051667034665, 0.11746522411128284, 0.2616471627290793, 0.21130492382424376, 0.09516449547361448, 0.0, 0.0], [0.10369895231986316, 0.06200555911909344, 0.06392986957451358, 0.10006414368184734, 0.1276459268762027, 0.20846696600384862, 0.2407526192003421, 0.09343596322428907, 0.0, 0.0], [0.11594454072790294, 0.09462738301559792, 0.0731369150779896, 0.11663778162911612, 0.13483535528596188, 0.19185441941074524, 0.20311958405545927, 0.06984402079722704, 0.0, 0.0], [0.1301047517838166, 0.10611811143160771, 0.12661302565659632, 0.10353727038105359, 0.12615758311826325, 0.21299529376043722, 0.1717018369515713, 0.022772126916654017, 0.0, 0.0], [0.12359550561797752, 0.1075738660008323, 0.0688722430295464, 0.0898876404494382, 0.14357053682896379, 0.17249271743653766, 0.2505201831044528, 0.04348730753225135, 0.0, 0.0], [0.14712790945974802, 0.08392056374119154, 0.07858210548793508, 0.09886824685030964, 0.15246636771300448, 0.2015801836429639, 0.23361093316250267, 0.003843689942344651, 0.0, 0.0], [0.1211144806671721, 0.07391963608794541, 0.09609552691432904, 0.11808188021228203, 0.1256633813495072, 0.18821076573161485, 0.27691432903714935, 0.0, 0.0, 0.0], [0.16874240583232078, 0.09295261239368165, 0.13517618469015796, 0.11619076549210207, 0.14064398541919806, 0.20215674362089914, 0.14413730255164034, 0.0, 0.0, 0.0], [0.13672199170124483, 0.13132780082987552, 0.10601659751037344, 0.1087136929460581, 0.12676348547717842, 0.22302904564315354, 0.16742738589211617, 0.0, 0.0, 0.0], [0.12828402366863906, 0.08828402366863905, 0.07976331360946745, 0.11905325443786982, 0.15266272189349112, 0.2539644970414201, 0.17798816568047338, 0.0, 0.0, 0.0], [0.1136805705640804, 0.10481953749729847, 0.11216771125999568, 0.11432893883725956, 0.18456883509833585, 0.2535119948130538, 0.11692241192997622, 0.0, 0.0, 0.0], [0.09232522796352584, 0.07104863221884498, 0.0668693009118541, 0.1295592705167173, 0.14703647416413373, 0.4101443768996961, 0.08301671732522796, 0.0, 0.0, 0.0], [0.1386748844375963, 0.09989727786337955, 0.09758602978941962, 0.12994350282485875, 0.17847971237801746, 0.2717000513610683, 0.08371854134566, 0.0, 0.0, 0.0], [0.12272727272727273, 0.10267379679144385, 0.10294117647058823, 0.12406417112299466, 0.2144385026737968, 0.2786096256684492, 0.05454545454545454, 0.0, 0.0, 0.0], [0.20610236220472442, 0.14153543307086613, 0.08877952755905512, 0.11535433070866141, 0.2017716535433071, 0.23996062992125985, 0.006496062992125984, 0.0, 0.0, 0.0], [0.12689610177785027, 0.1265698907192954, 0.10161474473984668, 0.15560267493068014, 0.2229652585222639, 0.25803294731691406, 0.008318381993149567, 0.0, 0.0, 0.0], [0.1255, 0.0975, 0.11325, 0.16475, 0.23475, 0.26425, 0.0, 0.0, 0.0, 0.0], [0.14562458249833, 0.11422845691382766, 0.12736584279670451, 0.15030060120240482, 0.27054108216432865, 0.19193943442440436, 0.0, 0.0, 0.0, 0.0], [0.13133971291866028, 0.12009569377990431, 0.14401913875598085, 0.17464114832535885, 0.2521531100478469, 0.1777511961722488, 0.0, 0.0, 0.0, 0.0], [0.17833209233060313, 0.15897244973938943, 0.1142963514519732, 0.17926284437825762, 0.24478778853313476, 0.12434847356664185, 0.0, 0.0, 0.0, 0.0], [0.14119199472179458, 0.10622388387948098, 0.1141411919947218, 0.18935561908950957, 0.29030129755883, 0.15878601275566306, 0.0, 0.0, 0.0, 0.0], [0.23159636062861869, 0.12551695616211744, 0.11311000827129859, 0.19396195202646815, 0.21836228287841192, 0.11745244003308519, 0.0, 0.0, 0.0, 0.0], [0.16728525980911982, 0.1312301166489926, 0.13361611876988336, 0.2059915164369035, 0.28950159066808057, 0.07237539766702014, 0.0, 0.0, 0.0, 0.0], [0.1446603886996594, 0.13724704468042476, 0.1580845521939491, 0.25806451612903225, 0.2660789420957724, 0.03586455620116209, 0.0, 0.0, 0.0, 0.0], [0.19433101128993513, 0.13860196973336536, 0.14316598606773961, 0.19841460485227, 0.29521979341820803, 0.030266634638481865, 0.0, 0.0, 0.0, 0.0], [0.26199765899336713, 0.16269996098322279, 0.17245415528677333, 0.17928209129925868, 0.22259071400702302, 0.0009754194303550526, 0.0, 0.0, 0.0, 0.0], [0.25945378151260506, 0.1752701080432173, 0.15741296518607442, 0.24474789915966386, 0.1631152460984394, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16583015458743225, 0.16703473198152982, 0.16281871110218832, 0.3057618952017667, 0.19855450712708292, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18434343434343434, 0.15059687786960516, 0.19077134986225897, 0.2676767676767677, 0.2066115702479339, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11824849966659258, 0.5569015336741499, 0.11502556123583019, 0.13125138919759946, 0.07857301622582796, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21221864951768488, 0.17282958199356913, 0.18542336548767416, 0.2877813504823151, 0.1417470525187567, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24361890320950214, 0.18397776092999749, 0.20975486479656305, 0.26105635582512005, 0.10159211523881728, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24614643545279383, 0.200626204238921, 0.19051059730250483, 0.2892581888246628, 0.07345857418111754, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2386733145458487, 0.18816388467374812, 0.24907869065683938, 0.2863646217212226, 0.03771948840234121, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2459063487503591, 0.19132433208848032, 0.276644642344154, 0.27635736857224935, 0.009767308244757253, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27124505928853754, 0.23641304347826086, 0.24703557312252963, 0.2349308300395257, 0.010375494071146246, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25473741201949107, 0.20194910665944776, 0.25446670276123445, 0.28884677855982677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.22471080482402167, 0.2579374846172779, 0.28624169333005167, 0.2311100172286488, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2857142857142857, 0.23988684582743988, 0.2727015558698727, 0.2016973125884017, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.43870575725353184, 0.1921616284368829, 0.23803736898070788, 0.13109524532887742, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24761904761904763, 0.27564625850340135, 0.3268027210884354, 0.14993197278911566, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4340659340659341, 0.22950126796280643, 0.2445054945054945, 0.091927303465765, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.29434216434665467, 0.3293668612483161, 0.31544678940278403, 0.06084418500224517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3580363464715601, 0.2773188576823224, 0.322397923058768, 0.04224687278734954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24740622505985635, 0.21335461558925245, 0.5163607342378292, 0.022878425113061984, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27451669595782074, 0.36203866432337434, 0.35711775043936733, 0.00632688927943761, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2826455624646693, 0.4053137365743358, 0.3120407009609949, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.29787888103289273, 0.41254226867506916, 0.2895788502920381, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.31617431997660134, 0.4036267914594911, 0.28019888856390757, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36686217008797656, 0.4252199413489736, 0.20791788856304985, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5562754256441164, 0.33825523579930694, 0.10546933855657677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4088809946714032, 0.4628774422735346, 0.12824156305506218, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.31596676397933976, 0.5984729395912868, 0.08556029642937346, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.37906233351092167, 0.5508790623335109, 0.07005860415556739, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3596044351213665, 0.6119268804315253, 0.02846868444710818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42065727699530514, 0.5683881064162755, 0.010954616588419406, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.33675256442331747, 0.6632474355766825, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3673548889754577, 0.6326451110245422, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5022745517795023, 0.49772544822049775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.64715381509891, 0.35284618490109004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7478540772532188, 0.2521459227467811, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7345834835917778, 0.26541651640822217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7921108742004265, 0.20788912579957355, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9324191968658179, 0.06758080313418217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9514629948364888, 0.048537005163511185, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.98989898989899, 0.010101010101010102, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] )},
    'curio_se': {500: ([0.7243125, 0.10115625, 0.03040625, 0.01840625, 0.0096875, 0.00796875, 0.006125, 0.00328125, 0.0028125, 0.0021875, 0.0020625, 0.0016875, 0.000875, 0.0015625, 0.00134375, 0.0006875, 0.0009375, 0.0004375, 0.001125, 0.00140625, 0.001375, 0.03259375, 0.0036875, 0.00221875, 0.0029375, 0.00125, 0.0010625, 0.00421875, 0.00215625, 0.000875, 0.002, 0.00184375, 0.0014375, 0.0010625, 0.002125, 0.001, 0.0013125, 0.0015, 0.00046875, 0.00121875, 0.00040625, 0.00096875, 0.000375, 0.00034375, 0.0005625, 0.00028125, 0.0005, 0.0003125, 0.0005625, 0.00034375, 0.0003125, 0.00040625, 0.0003125, 0.00296875, 0.00021875, 0.001, 0.00034375, 0.00084375, 0.00015625, 0.00040625, 0.000375, 0.00025, 6.25e-05, 0.0005, 0.00021875, 0.00021875, 0.00015625, 0.000125, 0.00034375, 0.0001875, 0.00021875, 6.25e-05, 6.25e-05, 0.000125, 0.000125, 3.125e-05, 0.000125, 3.125e-05, 0.00015625, 0.0, 3.125e-05, 0.0, 0.0, 0.0, 3.125e-05, 0.000125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
    [[0.2710760203641384, 0.12140823194408491, 0.10069893864871861, 0.08201743032185693, 0.15376650271809475, 0.14824402450599708, 0.08900681680904306, 0.02601604970230391, 0.007765984985762361, 0.0], [0.21501390176088972, 0.17207290701266606, 0.10874266295953043, 0.09731232622798888, 0.1566265060240964, 0.14118010503552672, 0.07476058078467716, 0.025332097621254247, 0.008958912573370404, 0.0], [0.2682425488180884, 0.13463514902363824, 0.11408016443987667, 0.08016443987667009, 0.13052415210688592, 0.1500513874614594, 0.08633093525179857, 0.029804727646454265, 0.006166495375128468, 0.0], [0.266553480475382, 0.12903225806451613, 0.11205432937181664, 0.06791171477079797, 0.1460101867572156, 0.15789473684210525, 0.1035653650254669, 0.013582342954159592, 0.003395585738539898, 0.0], [0.25806451612903225, 0.13870967741935483, 0.06129032258064516, 0.08387096774193549, 0.22258064516129034, 0.14516129032258066, 0.08387096774193549, 0.0064516129032258064, 0.0, 0.0], [0.3333333333333333, 0.1411764705882353, 0.058823529411764705, 0.047058823529411764, 0.20784313725490197, 0.12941176470588237, 0.07450980392156863, 0.00784313725490196, 0.0, 0.0], [0.2602040816326531, 0.22448979591836735, 0.08163265306122448, 0.09693877551020408, 0.14285714285714285, 0.09693877551020408, 0.08673469387755102, 0.01020408163265306, 0.0, 0.0], [0.3238095238095238, 0.14285714285714285, 0.0380952380952381, 0.08571428571428572, 0.23809523809523808, 0.08571428571428572, 0.05714285714285714, 0.02857142857142857, 0.0, 0.0], [0.23333333333333334, 0.14444444444444443, 0.06666666666666667, 0.12222222222222222, 0.17777777777777778, 0.13333333333333333, 0.1111111111111111, 0.011111111111111112, 0.0, 0.0], [0.3, 0.12857142857142856, 0.15714285714285714, 0.08571428571428572, 0.12857142857142856, 0.11428571428571428, 0.04285714285714286, 0.04285714285714286, 0.0, 0.0], [0.3333333333333333, 0.18181818181818182, 0.15151515151515152, 0.06060606060606061, 0.10606060606060606, 0.07575757575757576, 0.06060606060606061, 0.030303030303030304, 0.0, 0.0], [0.5555555555555556, 0.1111111111111111, 0.07407407407407407, 0.05555555555555555, 0.05555555555555555, 0.037037037037037035, 0.1111111111111111, 0.0, 0.0, 0.0], [0.39285714285714285, 0.21428571428571427, 0.14285714285714285, 0.0, 0.07142857142857142, 0.14285714285714285, 0.03571428571428571, 0.0, 0.0, 0.0], [0.28, 0.14, 0.3, 0.12, 0.1, 0.04, 0.02, 0.0, 0.0, 0.0], [0.2558139534883721, 0.3023255813953488, 0.11627906976744186, 0.18604651162790697, 0.06976744186046512, 0.06976744186046512, 0.0, 0.0, 0.0, 0.0], [0.18181818181818182, 0.13636363636363635, 0.13636363636363635, 0.18181818181818182, 0.045454545454545456, 0.18181818181818182, 0.13636363636363635, 0.0, 0.0, 0.0], [0.4, 0.16666666666666666, 0.16666666666666666, 0.1, 0.1, 0.06666666666666667, 0.0, 0.0, 0.0, 0.0], [0.2857142857142857, 0.0, 0.42857142857142855, 0.14285714285714285, 0.0, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0], [0.5277777777777778, 0.2222222222222222, 0.08333333333333333, 0.05555555555555555, 0.08333333333333333, 0.027777777777777776, 0.0, 0.0, 0.0, 0.0], [0.4444444444444444, 0.08888888888888889, 0.13333333333333333, 0.2, 0.08888888888888889, 0.022222222222222223, 0.022222222222222223, 0.0, 0.0, 0.0], [0.4772727272727273, 0.045454545454545456, 0.1590909090909091, 0.045454545454545456, 0.25, 0.022727272727272728, 0.0, 0.0, 0.0, 0.0], [0.6625119846596357, 0.087248322147651, 0.23106423777564716, 0.014381591562799617, 0.004793863854266539, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5338983050847458, 0.1016949152542373, 0.2542372881355932, 0.01694915254237288, 0.05084745762711865, 0.03389830508474576, 0.00847457627118644, 0.0, 0.0, 0.0], [0.5070422535211268, 0.08450704225352113, 0.2676056338028169, 0.056338028169014086, 0.04225352112676056, 0.04225352112676056, 0.0, 0.0, 0.0, 0.0], [0.5425531914893617, 0.10638297872340426, 0.2872340425531915, 0.010638297872340425, 0.031914893617021274, 0.02127659574468085, 0.0, 0.0, 0.0, 0.0], [0.55, 0.125, 0.225, 0.025, 0.075, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5882352941176471, 0.11764705882352941, 0.17647058823529413, 0.08823529411764706, 0.029411764705882353, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6518518518518519, 0.06666666666666667, 0.22962962962962963, 0.014814814814814815, 0.02962962962962963, 0.007407407407407408, 0.0, 0.0, 0.0, 0.0], [0.5942028985507246, 0.08695652173913043, 0.2318840579710145, 0.028985507246376812, 0.043478260869565216, 0.014492753623188406, 0.0, 0.0, 0.0, 0.0], [0.42857142857142855, 0.10714285714285714, 0.10714285714285714, 0.14285714285714285, 0.17857142857142858, 0.03571428571428571, 0.0, 0.0, 0.0, 0.0], [0.59375, 0.109375, 0.171875, 0.078125, 0.03125, 0.015625, 0.0, 0.0, 0.0, 0.0], [0.6610169491525424, 0.05084745762711865, 0.22033898305084745, 0.0, 0.05084745762711865, 0.01694915254237288, 0.0, 0.0, 0.0, 0.0], [0.6521739130434783, 0.10869565217391304, 0.17391304347826086, 0.0, 0.06521739130434782, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5882352941176471, 0.058823529411764705, 0.17647058823529413, 0.029411764705882353, 0.14705882352941177, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5735294117647058, 0.16176470588235295, 0.20588235294117646, 0.04411764705882353, 0.0, 0.014705882352941176, 0.0, 0.0, 0.0, 0.0], [0.375, 0.03125, 0.25, 0.0, 0.125, 0.21875, 0.0, 0.0, 0.0, 0.0], [0.47619047619047616, 0.09523809523809523, 0.30952380952380953, 0.023809523809523808, 0.047619047619047616, 0.047619047619047616, 0.0, 0.0, 0.0, 0.0], [0.5208333333333334, 0.10416666666666667, 0.16666666666666666, 0.020833333333333332, 0.1875, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5333333333333333, 0.06666666666666667, 0.3333333333333333, 0.06666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5641025641025641, 0.07692307692307693, 0.28205128205128205, 0.05128205128205128, 0.0, 0.02564102564102564, 0.0, 0.0, 0.0, 0.0], [0.6153846153846154, 0.07692307692307693, 0.15384615384615385, 0.07692307692307693, 0.07692307692307693, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3548387096774194, 0.45161290322580644, 0.06451612903225806, 0.0967741935483871, 0.03225806451612903, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.08333333333333333, 0.08333333333333333, 0.25, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36363636363636365, 0.18181818181818182, 0.0, 0.45454545454545453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6111111111111112, 0.05555555555555555, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.4444444444444444, 0.2222222222222222, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4375, 0.1875, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2777777777777778, 0.2222222222222222, 0.16666666666666666, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36363636363636365, 0.45454545454545453, 0.09090909090909091, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6923076923076923, 0.07692307692307693, 0.15384615384615385, 0.07692307692307693, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.10526315789473684, 0.8315789473684211, 0.06315789473684211, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42857142857142855, 0.5714285714285714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.40625, 0.09375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8181818181818182, 0.18181818181818182, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.48148148148148145, 0.4444444444444444, 0.07407407407407407, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6923076923076923, 0.15384615384615385, 0.15384615384615385, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8333333333333334, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.375, 0.5, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8125, 0.1875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8571428571428571, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8571428571428571, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
    1000: ([0.6234847327411223, 0.07329157299692254, 0.03143374729903747, 0.02487286378418491, 0.017098456904642383, 0.011964991160486282, 0.009468101359757295, 0.005190213239627213, 0.007722024576030731, 0.005085448632603619, 0.004574721173363599, 0.008791496606063251, 0.006259685269659733, 0.006041425671693913, 0.003802082196564594, 0.0039024816116288714, 0.0034354060719820156, 0.0018770325425060566, 0.004063993714123579, 0.0028635659253115654, 0.0025841936399153154, 0.006635091778160945, 0.0036056485583953555, 0.005342994958203287, 0.003518344719209027, 0.0022655346268852174, 0.0027544361263286553, 0.009970098435078683, 0.007791867647379794, 0.007839884758932275, 0.003915577187506821, 0.008023222821223563, 0.0026060195997118975, 0.0026671322871423273, 0.005556889364209791, 0.003924307571425453, 0.0024357771132985572, 0.0031691293624637146, 0.0037016827815003167, 0.00193378003797717, 0.0031298426348298665, 0.002117118100268459, 0.002252439051007268, 0.004365191959316411, 0.002125848484187092, 0.0030512691795621712, 0.0011829670209747474, 0.0023441080821529126, 0.0011218543335443176, 0.0010651068380732043, 0.0028853918851081475, 0.0026365759434271124, 0.0017897287033197284, 0.00159329506515049, 0.003003252068009691, 0.0032258768579348277, 0.0009385162712530283, 0.0008250212803108017, 0.0007289870572058406, 0.0007770041687583211, 0.0006329528341008796, 0.0006504136019381452, 0.0007377174411244735, 0.0009297858873343956, 0.001444878538533732, 0.0005019970753213873, 0.0010651068380732043, 0.0007246218652465242, 0.0008992295436191807, 0.0007289870572058406, 0.0006766047536940437, 0.0005543793788331841, 0.0004758059235654888, 0.00038413689241984417, 0.00031429382107078156, 0.0003361197808673636, 0.00043215400397232466, 0.00029683305323351594, 0.0003972324682977934, 0.00017460767837265643, 0.00022262478992513695, 0.0001178601829015431, 0.0001222253748608595, 0.00010912979898291027, 9.166903114564464e-05, 0.00010912979898291027, 1.7460767837265645e-05, 6.984307134906258e-05, 4.801711155248052e-05, 4.801711155248052e-05, 3.492153567453129e-05, 1.7460767837265645e-05, 4.365191959316411e-06, 4.365191959316411e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
    [[0.13729512500787644, 0.08029069319685503, 0.09399920185393927, 0.11058523709838901, 0.13847134025526672, 0.15224986172469562, 0.13983659009598756, 0.09546246963194265, 0.049793112139521534, 0.002016368995526181], [0.13567599761762955, 0.08880285884455033, 0.09440142942227517, 0.11048243001786778, 0.13936867182846932, 0.14228707564026205, 0.15008933889219775, 0.09696247766527695, 0.04038117927337701, 0.0015485407980941036], [0.14747951673378698, 0.09609776419941675, 0.09734759061241494, 0.11817803082905153, 0.14164699347312873, 0.14359116789334814, 0.13567560061102624, 0.08123871684488265, 0.0383280099986113, 0.00041660880433273156], [0.19305019305019305, 0.13566163566163567, 0.09740259740259741, 0.08862758862758863, 0.1310986310986311, 0.14566514566514566, 0.1261846261846262, 0.05738855738855739, 0.024745524745524744, 0.0001755001755001755], [0.15011488383967322, 0.11360735256573909, 0.11769211130967577, 0.09599183048251213, 0.1166709216236916, 0.18049527699770232, 0.13224406433495023, 0.07786571355629308, 0.015317845289762573, 0.0], [0.12075884713608172, 0.08391098139365195, 0.11163808828894564, 0.09485589201021526, 0.13863553447646845, 0.11856986501276906, 0.08901860634804816, 0.2032105071141919, 0.039401678219627874, 0.0], [0.08713692946058091, 0.05532503457814661, 0.07238358690640849, 0.09727985246657446, 0.11341632088520055, 0.16551406177962194, 0.25311203319502074, 0.12632549562010142, 0.02950668510834486, 0.0], [0.1118587047939445, 0.13036164844407064, 0.10933557611438183, 0.12867956265769553, 0.13877207737594618, 0.1757779646761985, 0.13204373423044574, 0.061396131202691336, 0.011774600504625737, 0.0], [0.20576596947427925, 0.10740531373657433, 0.0746184284906727, 0.08648954211418881, 0.15036743923120408, 0.1317128321085359, 0.18032786885245902, 0.058790276992651214, 0.0045223289994347085, 0.0], [0.15107296137339055, 0.07982832618025751, 0.09871244635193133, 0.14678111587982834, 0.15965665236051502, 0.13218884120171673, 0.13390557939914163, 0.08583690987124463, 0.01201716738197425, 0.0], [0.15362595419847327, 0.21278625954198474, 0.12595419847328243, 0.13263358778625955, 0.11736641221374046, 0.13740458015267176, 0.07538167938931298, 0.04198473282442748, 0.0028625954198473282, 0.0], [0.1519364448857994, 0.2224428997020854, 0.14895729890764647, 0.12462760675273088, 0.12859980139026814, 0.105759682224429, 0.06752730883813307, 0.04865938430983118, 0.0014895729890764648, 0.0], [0.18828451882845187, 0.14086471408647142, 0.08786610878661087, 0.14435146443514643, 0.15550906555090654, 0.16108786610878661, 0.09693165969316597, 0.024407252440725245, 0.000697350069735007, 0.0], [0.22760115606936415, 0.08598265895953758, 0.15173410404624277, 0.13800578034682082, 0.14234104046242774, 0.15534682080924855, 0.07803468208092486, 0.020953757225433526, 0.0, 0.0], [0.2365097588978186, 0.10677382319173363, 0.11710677382319173, 0.13088404133180254, 0.18828932261768083, 0.10907003444316878, 0.09299655568312284, 0.018369690011481057, 0.0, 0.0], [0.19798657718120805, 0.1174496644295302, 0.19686800894854586, 0.17785234899328858, 0.10961968680089486, 0.09731543624161074, 0.08612975391498881, 0.016778523489932886, 0.0, 0.0], [0.2337992376111817, 0.14104193138500634, 0.21855146124523506, 0.1473951715374841, 0.10038119440914867, 0.08386277001270648, 0.054637865311308764, 0.020330368487928845, 0.0, 0.0], [0.21395348837209302, 0.16046511627906976, 0.16279069767441862, 0.1325581395348837, 0.09767441860465116, 0.14418604651162792, 0.06511627906976744, 0.023255813953488372, 0.0, 0.0], [0.28356605800214824, 0.09989258861439312, 0.10741138560687433, 0.17722878625134264, 0.0966702470461869, 0.12352309344790548, 0.1041890440386681, 0.007518796992481203, 0.0, 0.0], [0.20121951219512196, 0.07469512195121951, 0.125, 0.21341463414634146, 0.12347560975609756, 0.15853658536585366, 0.09298780487804878, 0.010670731707317074, 0.0, 0.0], [0.1858108108108108, 0.11993243243243243, 0.16554054054054054, 0.15371621621621623, 0.21621621621621623, 0.07432432432432433, 0.07939189189189189, 0.005067567567567568, 0.0, 0.0], [0.49407894736842106, 0.08552631578947369, 0.18881578947368421, 0.07236842105263158, 0.05131578947368421, 0.07368421052631578, 0.03355263157894737, 0.0006578947368421052, 0.0, 0.0], [0.22639225181598063, 0.09927360774818401, 0.1513317191283293, 0.14527845036319612, 0.2106537530266344, 0.11138014527845036, 0.05569007263922518, 0.0, 0.0, 0.0], [0.2908496732026144, 0.06290849673202614, 0.08251633986928104, 0.37745098039215685, 0.09477124183006536, 0.061274509803921566, 0.029411764705882353, 0.0008169934640522876, 0.0, 0.0], [0.3535980148883375, 0.1141439205955335, 0.11538461538461539, 0.2518610421836228, 0.07568238213399504, 0.06823821339950373, 0.02109181141439206, 0.0, 0.0, 0.0], [0.24084778420038536, 0.14836223506743737, 0.18497109826589594, 0.18882466281310212, 0.13872832369942195, 0.07707129094412331, 0.02119460500963391, 0.0, 0.0, 0.0], [0.18066561014263074, 0.2329635499207607, 0.23771790808240886, 0.1109350237717908, 0.11251980982567353, 0.10776545166402536, 0.017432646592709985, 0.0, 0.0, 0.0], [0.2535026269702277, 0.11777583187390543, 0.28809106830122594, 0.09588441330998249, 0.16637478108581435, 0.07705779334500876, 0.0013134851138353765, 0.0, 0.0, 0.0], [0.30868347338935576, 0.1619047619047619, 0.16806722689075632, 0.15182072829131651, 0.12549019607843137, 0.08179271708683473, 0.002240896358543417, 0.0, 0.0, 0.0], [0.3730512249443207, 0.22884187082405344, 0.1364142538975501, 0.1319599109131403, 0.08630289532293986, 0.04287305122494432, 0.0005567928730512249, 0.0, 0.0, 0.0], [0.3010033444816054, 0.2129319955406912, 0.1750278706800446, 0.14492753623188406, 0.11259754738015608, 0.05016722408026756, 0.0033444816053511705, 0.0, 0.0, 0.0], [0.2823721436343852, 0.13601741022850924, 0.3400435255712731, 0.176278563656148, 0.04461371055495103, 0.02013057671381937, 0.000544069640914037, 0.0, 0.0, 0.0], [0.25963149078726966, 0.19262981574539365, 0.21608040201005024, 0.17252931323283083, 0.11892797319932999, 0.038525963149078725, 0.0016750418760469012, 0.0, 0.0, 0.0], [0.2078559738134206, 0.1718494271685761, 0.23076923076923078, 0.14238952536824878, 0.2176759410801964, 0.029459901800327332, 0.0, 0.0, 0.0, 0.0], [0.19010212097407697, 0.211311861743912, 0.23723487824037706, 0.17831893165750196, 0.15710919088766692, 0.025923016496465043, 0.0, 0.0, 0.0, 0.0], [0.23136818687430477, 0.20133481646273638, 0.12458286985539488, 0.22246941045606228, 0.19243604004449388, 0.027808676307007785, 0.0, 0.0, 0.0, 0.0], [0.35842293906810035, 0.14516129032258066, 0.2222222222222222, 0.13978494623655913, 0.10931899641577061, 0.025089605734767026, 0.0, 0.0, 0.0, 0.0], [0.3608815426997245, 0.23553719008264462, 0.21212121212121213, 0.09917355371900827, 0.07851239669421488, 0.013774104683195593, 0.0, 0.0, 0.0, 0.0], [0.5908018867924528, 0.12735849056603774, 0.1544811320754717, 0.08136792452830188, 0.04481132075471698, 0.0011792452830188679, 0.0, 0.0, 0.0, 0.0], [0.43115124153498874, 0.17381489841986456, 0.17832957110609482, 0.11286681715575621, 0.09255079006772009, 0.011286681715575621, 0.0, 0.0, 0.0, 0.0], [0.2412831241283124, 0.15481171548117154, 0.15481171548117154, 0.35564853556485354, 0.08647140864714087, 0.00697350069735007, 0.0, 0.0, 0.0, 0.0], [0.4865979381443299, 0.177319587628866, 0.13195876288659794, 0.12577319587628866, 0.07628865979381444, 0.002061855670103093, 0.0, 0.0, 0.0, 0.0], [0.3701550387596899, 0.20155038759689922, 0.2248062015503876, 0.1298449612403101, 0.07364341085271318, 0.0, 0.0, 0.0, 0.0, 0.0], [0.275, 0.405, 0.126, 0.151, 0.043, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2874743326488706, 0.31211498973305957, 0.19096509240246407, 0.17248459958932238, 0.03696098562628337, 0.0, 0.0, 0.0, 0.0, 0.0], [0.13161659513590845, 0.1688125894134478, 0.35765379113018597, 0.2989985693848355, 0.04291845493562232, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2177121771217712, 0.14391143911439114, 0.28044280442804426, 0.2730627306273063, 0.08487084870848709, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2905027932960894, 0.1601489757914339, 0.4376163873370577, 0.09869646182495345, 0.01303538175046555, 0.0, 0.0, 0.0, 0.0, 0.0], [0.22178988326848248, 0.38910505836575876, 0.22568093385214008, 0.14007782101167315, 0.023346303501945526, 0.0, 0.0, 0.0, 0.0, 0.0], [0.35655737704918034, 0.3155737704918033, 0.19672131147540983, 0.11475409836065574, 0.01639344262295082, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5264750378214826, 0.1573373676248109, 0.24508320726172467, 0.0680786686838124, 0.0030257186081694403, 0.0, 0.0, 0.0, 0.0, 0.0], [0.640728476821192, 0.20033112582781457, 0.11589403973509933, 0.041390728476821195, 0.0016556291390728477, 0.0, 0.0, 0.0, 0.0, 0.0], [0.48048780487804876, 0.21707317073170732, 0.2121951219512195, 0.09024390243902439, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4191780821917808, 0.40273972602739727, 0.13424657534246576, 0.043835616438356165, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6875, 0.16569767441860464, 0.11046511627906977, 0.03488372093023256, 0.0014534883720930232, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4871447902571042, 0.2963464140730717, 0.16373477672530445, 0.05277401894451962, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4744186046511628, 0.2744186046511628, 0.20465116279069767, 0.046511627906976744, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42857142857142855, 0.36507936507936506, 0.17989417989417988, 0.026455026455026454, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.437125748502994, 0.3532934131736527, 0.19161676646706588, 0.017964071856287425, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4101123595505618, 0.4157303370786517, 0.16292134831460675, 0.011235955056179775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.496551724137931, 0.3724137931034483, 0.12413793103448276, 0.006896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4161073825503356, 0.44966442953020136, 0.1342281879194631, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5621301775147929, 0.3076923076923077, 0.1242603550295858, 0.005917159763313609, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5211267605633803, 0.41784037558685444, 0.06103286384976526, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7824773413897281, 0.18429003021148035, 0.03323262839879154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5739130434782609, 0.3217391304347826, 0.10434782608695652, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.319672131147541, 0.6229508196721312, 0.05737704918032787, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.42771084337349397, 0.07228915662650602, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4854368932038835, 0.5048543689320388, 0.009708737864077669, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6706586826347305, 0.2215568862275449, 0.10778443113772455, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6193548387096774, 0.38064516129032255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6299212598425197, 0.3543307086614173, 0.015748031496062992, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7247706422018348, 0.27522935779816515, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7045454545454546, 0.29545454545454547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7916666666666666, 0.20833333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8701298701298701, 0.12987012987012986, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9595959595959596, 0.04040404040404041, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7352941176470589, 0.2647058823529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9560439560439561, 0.04395604395604396, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.975, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9803921568627451, 0.0196078431372549, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9629629629629629, 0.037037037037037035, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
    2000: ([0.4772083011499751, 0.04734776454720466, 0.022236903496975092, 0.016147877466682874, 0.012834788609247635, 0.008792428121655054, 0.008614031029331618, 0.006161561111786363, 0.006757525024603116, 0.008531693909797725, 0.005220565459970437, 0.007467192578680959, 0.00556559719896961, 0.0060243325792298736, 0.004953950025289258, 0.005263694427345334, 0.005202921791498888, 0.004850048422067916, 0.00556363679136166, 0.005479339264219817, 0.006908476410415253, 0.006479147144274238, 0.006159600704178413, 0.007159408584232834, 0.00492454391117001, 0.005279377688208933, 0.008768903230359656, 0.009749107034334578, 0.009386431626863858, 0.00746327176346506, 0.0077690953503052355, 0.00973342377347098, 0.0074064199428345145, 0.005169594862163741, 0.006298789644342852, 0.005953757905343679, 0.005767519182588444, 0.004069806194103878, 0.008359178040298138, 0.006459543068194739, 0.00669283157354077, 0.009762829887590228, 0.00555579516092986, 0.008670882849962164, 0.0046285223623695835, 0.005199000976282989, 0.0037816262757352508, 0.006026292986837823, 0.004830444345988418, 0.003975706628922286, 0.006398770432348294, 0.005240169536049936, 0.004030598041944881, 0.004218797172308066, 0.004785354971005571, 0.006267423122615654, 0.0062791855682633535, 0.00429133225380221, 0.004626561954761634, 0.005800846111923591, 0.004330540405961207, 0.010268615050441288, 0.0048441671992440665, 0.0038129927974624484, 0.005273496465385083, 0.005338189916447428, 0.0035483377703892194, 0.005967480758599328, 0.004477570976557446, 0.0024544303251532056, 0.0025112821457837515, 0.002083913287250685, 0.0018133770373536067, 0.003550298177997169, 0.0020799924720347855, 0.0032307517379013447, 0.0017388815482515124, 0.0014114934777238884, 0.0017937729612741082, 0.0012174131245368538, 0.000909629130088728, 0.0007625985594924896, 0.000897866684441029, 0.000935114428992076, 0.0010076495104862204, 0.0009919662496226216, 0.0005920430976008532, 0.00046853741830001294, 0.0003215068477037746, 0.0002587738042493795, 0.00011958486408494053, 0.00015291179342008792, 0.000113703641261091, 8.037671192594364e-05, 5.68518206305455e-05, 2.9406114119247675e-05, 1.960407607949845e-06, 0.0, 0.0, 0.0] ,
    [[0.08857421032523631, 0.05515912629455721, 0.06845285778254315, 0.08775259527653509, 0.12129502963976288, 0.14853156850420873, 0.16621272435225923, 0.14582845499398167, 0.10200761637150146, 0.01618581645941427], [0.10156508777740973, 0.06889698575687314, 0.07668102020536602, 0.09999171911228884, 0.13174892348459755, 0.1444600861212322, 0.16238820801589932, 0.13332229214971844, 0.07403113613779397, 0.006914541238820801], [0.10808428105439478, 0.07881512827294367, 0.08198889182755885, 0.10773164065943754, 0.13021246583796173, 0.15093008904169972, 0.16671074671603633, 0.11099356431279203, 0.05880278585912016, 0.005730406418055188], [0.14277042612601676, 0.10270729634575695, 0.09008134029379629, 0.08911011290518393, 0.13002306665047955, 0.15163287604710451, 0.15223989316498726, 0.08923151632876047, 0.04795435231273522, 0.00424911982517907], [0.10065678936917673, 0.07667634030853826, 0.08675729341683214, 0.08400794256911563, 0.12967771498396213, 0.16098976630517794, 0.1565602566060791, 0.14541011150145106, 0.05758362608828471, 0.0016801588513823125], [0.08784838350055742, 0.06376811594202898, 0.09119286510590859, 0.09698996655518395, 0.13600891861761427, 0.13110367892976588, 0.15072463768115943, 0.19219620958751393, 0.04972129319955407, 0.0004459308807134894], [0.06053709604005462, 0.05097860719162494, 0.058716431497496585, 0.09968138370505235, 0.11697769685935366, 0.18730086481565772, 0.23236231224396905, 0.15179790623577605, 0.04164770141101502, 0.0], [0.08526885141584474, 0.07349665924276169, 0.07031498568246898, 0.13108495068405981, 0.12090359529112313, 0.16926503340757237, 0.19630925867006044, 0.11963092586700605, 0.03372573973910277, 0.0], [0.11778357992457208, 0.0673049028140412, 0.0675950101537569, 0.10008703220191471, 0.16594139831737742, 0.15752828546562228, 0.2059762111981433, 0.09428488540760081, 0.02349869451697128, 0.0], [0.07421875, 0.04204963235294118, 0.08065257352941177, 0.09650735294117647, 0.13212316176470587, 0.19738051470588236, 0.22633272058823528, 0.11787683823529412, 0.03285845588235294, 0.0], [0.08636875704093128, 0.10063837776943298, 0.09425460007510326, 0.10214044310927525, 0.14194517461509576, 0.172361997746902, 0.17386406308674426, 0.10777318813368382, 0.020653398422831395, 0.0], [0.09713835652402206, 0.1407193489104752, 0.11157784195326857, 0.12260435809923864, 0.152533473352586, 0.1514833289577317, 0.12654239957994223, 0.08611184037805199, 0.011289052244683644, 0.0], [0.11553363860514265, 0.08770693906305037, 0.08030996829869673, 0.1306798168369144, 0.1785840084536809, 0.16731243395561818, 0.14758717858400847, 0.08171891511095457, 0.010567101091933779, 0.0], [0.11812561015294501, 0.06150341685649203, 0.09534656687276277, 0.11454604620891637, 0.21737715587373901, 0.16856492027334852, 0.13472177025707777, 0.08232997071265864, 0.007484542792059876, 0.0], [0.11159477641472101, 0.06925207756232687, 0.10051444400474871, 0.13850415512465375, 0.184804115552038, 0.1669964384645825, 0.14364859517214087, 0.08191531460229522, 0.002770083102493075, 0.0], [0.09608938547486033, 0.06443202979515829, 0.1303538175046555, 0.16238361266294227, 0.15456238361266295, 0.15977653631284916, 0.15083798882681565, 0.08044692737430167, 0.0011173184357541898, 0.0], [0.09683496608892238, 0.09382064807837227, 0.13602110022607386, 0.11567445365486059, 0.15749811605124342, 0.15599095704596835, 0.1680482290881688, 0.07611152976639035, 0.0, 0.0], [0.07154405820533549, 0.08973322554567502, 0.11600646725949879, 0.13217461600646727, 0.17259498787388844, 0.20371867421180276, 0.15804365400161682, 0.05618431689571544, 0.0, 0.0], [0.12403100775193798, 0.06307258632840028, 0.10570824524312897, 0.1522198731501057, 0.16596194503171247, 0.19027484143763213, 0.16807610993657504, 0.0306553911205074, 0.0, 0.0], [0.118783542039356, 0.0851520572450805, 0.11520572450805008, 0.16493738819320214, 0.13595706618962433, 0.17638640429338104, 0.156350626118068, 0.047227191413237925, 0.0, 0.0], [0.08399545970488081, 0.07406356413166856, 0.11889897843359819, 0.11691259931895573, 0.12400681044267878, 0.2437570942111237, 0.19721906923950056, 0.04114642451759364, 0.0, 0.0], [0.2574886535552194, 0.07473524962178517, 0.12012102874432677, 0.11860816944024205, 0.12344931921331316, 0.18517397881996975, 0.10741301059001512, 0.013010590015128594, 0.0, 0.0], [0.08847867600254615, 0.049331635900700194, 0.09707192870782941, 0.14704010184595798, 0.21037555697008276, 0.24220241884150223, 0.1524506683640993, 0.013049013367281986, 0.0, 0.0], [0.17278203723986857, 0.05366922234392114, 0.0988499452354874, 0.24288061336254108, 0.19304490690032858, 0.14074479737130338, 0.09309967141292443, 0.004928806133625411, 0.0, 0.0], [0.1480891719745223, 0.06687898089171974, 0.10270700636942676, 0.15963375796178345, 0.17794585987261147, 0.2089968152866242, 0.1305732484076433, 0.0051751592356687895, 0.0, 0.0], [0.08057927961381359, 0.06832528778314148, 0.1344225770516153, 0.1422205718529521, 0.2350538432974378, 0.20794652803564798, 0.13070924619383587, 0.0007426661715558856, 0.0, 0.0], [0.12117147328414934, 0.07936507936507936, 0.12854907221104403, 0.19517102615694165, 0.29085624860272746, 0.12206572769953052, 0.0628213726805276, 0.0, 0.0, 0.0], [0.15785240297607078, 0.10898853810577117, 0.22159662175749045, 0.15724914538507942, 0.2079227830283531, 0.11421677056102957, 0.03217373818620551, 0.0, 0.0, 0.0], [0.13909774436090225, 0.11131996658312447, 0.13032581453634084, 0.13993316624895571, 0.2627401837928154, 0.1683375104427736, 0.04824561403508772, 0.0, 0.0, 0.0], [0.21040189125295508, 0.1415812976096664, 0.12188074599422118, 0.1667980036774363, 0.1728395061728395, 0.136327817178881, 0.050170738114000524, 0.0, 0.0, 0.0], [0.12137269745142569, 0.10219530658591976, 0.11178400201867272, 0.13676507696189755, 0.1806712086802927, 0.186222558667676, 0.16098914963411556, 0.0, 0.0, 0.0], [0.20926485397784492, 0.13554884189325278, 0.21309164149043303, 0.1540785498489426, 0.15891238670694863, 0.08821752265861027, 0.040886203423967774, 0.0, 0.0, 0.0], [0.1850185283218634, 0.19640021175224986, 0.15034409740603494, 0.16437268395976706, 0.14690312334568556, 0.1069348861831657, 0.05002646903123346, 0.0, 0.0, 0.0], [0.13120970800151688, 0.13765642775881684, 0.1626848691695108, 0.2006067500948047, 0.2328403488813045, 0.11149032992036405, 0.023511566173682216, 0.0, 0.0, 0.0], [0.14565826330532214, 0.15717398070339247, 0.20292561469032058, 0.20510426392779335, 0.1870525988173047, 0.0915032679738562, 0.010582010582010581, 0.0, 0.0, 0.0], [0.13730655251893314, 0.11787948633519921, 0.14027000329272307, 0.26440566348370104, 0.22522225880803423, 0.11261112940401712, 0.0023049061573921633, 0.0, 0.0, 0.0], [0.18728755948334466, 0.12440516655336506, 0.22501699524133242, 0.23113528212100612, 0.1583956492182189, 0.07375934738273283, 0.0, 0.0, 0.0, 0.0], [0.1960500963391137, 0.16907514450867053, 0.20134874759152216, 0.18882466281310212, 0.15558766859344894, 0.08911368015414259, 0.0, 0.0, 0.0, 0.0], [0.21904315196998123, 0.14118198874296436, 0.13602251407129456, 0.14681050656660413, 0.2849437148217636, 0.07199812382739212, 0.0, 0.0, 0.0, 0.0], [0.1526555386949924, 0.21122913505311078, 0.165402124430956, 0.2094081942336874, 0.2006069802731411, 0.06069802731411229, 0.0, 0.0, 0.0, 0.0], [0.10984182776801406, 0.15377855887521968, 0.1751611013473931, 0.2521968365553603, 0.2680140597539543, 0.041007615700058585, 0.0, 0.0, 0.0, 0.0], [0.13112449799196788, 0.14216867469879518, 0.19236947791164657, 0.22791164658634538, 0.2863453815261044, 0.020080321285140562, 0.0, 0.0, 0.0, 0.0], [0.15949188426252647, 0.16125617501764292, 0.20889202540578689, 0.21383203952011293, 0.2268877911079746, 0.029640084685956247, 0.0, 0.0, 0.0, 0.0], [0.13429798779109203, 0.23626497852136558, 0.1962468912502826, 0.24869997739091115, 0.1770291657246213, 0.007460999321727334, 0.0, 0.0, 0.0, 0.0], [0.22744599745870395, 0.18297331639135958, 0.19229140194832697, 0.22193985599322322, 0.16857263871241, 0.006776789495976281, 0.0, 0.0, 0.0, 0.0], [0.14969834087481146, 0.19004524886877827, 0.24509803921568626, 0.23868778280542988, 0.17496229260935142, 0.0015082956259426848, 0.0, 0.0, 0.0, 0.0], [0.1824779678589943, 0.18921721099015035, 0.2089165370658372, 0.2939346811819596, 0.12441679626749612, 0.0010368066355624676, 0.0, 0.0, 0.0, 0.0], [0.1977878985035784, 0.22218607677293428, 0.3109954456733897, 0.18542615484710476, 0.08360442420299284, 0.0, 0.0, 0.0, 0.0, 0.0], [0.14488636363636365, 0.2780032467532468, 0.2780032467532468, 0.21266233766233766, 0.0864448051948052, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2465483234714004, 0.2465483234714004, 0.28303747534516766, 0.1814595660749507, 0.04240631163708087, 0.0, 0.0, 0.0, 0.0, 0.0], [0.22977941176470587, 0.27328431372549017, 0.25796568627450983, 0.21323529411764705, 0.025735294117647058, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3363262252151141, 0.24354657687991021, 0.23718668163112608, 0.16049382716049382, 0.02244668911335578, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3132295719844358, 0.22081712062256809, 0.25, 0.20525291828793774, 0.010700389105058366, 0.0, 0.0, 0.0, 0.0, 0.0], [0.266728624535316, 0.25743494423791824, 0.2342007434944238, 0.22769516728624536, 0.013940520446096654, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4219582138467841, 0.20647275706677592, 0.20606308889799263, 0.15731257681278166, 0.00819336337566571, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36315295589615265, 0.2696277760400375, 0.24585548952142633, 0.11510791366906475, 0.006255864873318737, 0.0, 0.0, 0.0, 0.0, 0.0], [0.46425226350296595, 0.2450827349359975, 0.2169840774274118, 0.07336871682797377, 0.00031220730565095225, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3663773412517131, 0.2987665600730927, 0.2206486980356327, 0.11420740063956145, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25508474576271184, 0.3008474576271186, 0.36610169491525424, 0.07796610169491526, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.30348090571138897, 0.31463332206826633, 0.32342007434944237, 0.05846569787090233, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3091896785875962, 0.34268899954730647, 0.2820280669986419, 0.06609325486645541, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18136693394425354, 0.2367315769377625, 0.5353188239786177, 0.046582665139366174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45366248482395793, 0.2962363415621206, 0.23269931201942534, 0.017401861594496155, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4318766066838046, 0.289974293059126, 0.2652956298200514, 0.012853470437017995, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36654275092936806, 0.2721189591078067, 0.3513011152416357, 0.010037174721189592, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47337495409474845, 0.3257436650752846, 0.19867792875504958, 0.0022034520749173708, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4878453038674033, 0.35027624309392263, 0.16077348066298341, 0.0011049723756906078, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6034822601839684, 0.30223390275952694, 0.09395532194480946, 0.000328515111695138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6335376532399299, 0.27189141856392296, 0.09457092819614711, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6214057507987221, 0.26757188498402557, 0.1110223642172524, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5964090554254489, 0.33489461358313816, 0.06869633099141297, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6020696142991533, 0.34807149576669805, 0.04985888993414864, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6432432432432432, 0.3318918918918919, 0.024864864864864864, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6449475427940364, 0.33959138597459965, 0.015461071231363888, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6880301602262017, 0.29783223374175305, 0.01413760603204524, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7773058252427184, 0.220873786407767, 0.0018203883495145632, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7576099210822999, 0.24239007891770012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6541666666666667, 0.3458333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8316939890710382, 0.16830601092896175, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.821256038647343, 0.178743961352657, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8663793103448276, 0.1336206896551724, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8817480719794345, 0.11825192802056556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9235807860262009, 0.07641921397379912, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.960167714884696, 0.039832285115303984, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9902723735408561, 0.009727626459143969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9901185770750988, 0.009881422924901186, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9966887417218543, 0.0033112582781456954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
    math.inf: ([0.2506254077827773, 0.022628774843056452, 0.011940217777131711, 0.008802191969764069, 0.0073029035824033915, 0.005798544321128303, 0.005293992366644467, 0.004376164188136184, 0.005029461777442725, 0.005828969564614765, 0.004502936035996444, 0.005312585570997306, 0.004605198659937053, 0.004799582159989452, 0.004447156422937929, 0.004392221955531816, 0.004633933612118712, 0.004562941377316967, 0.005177362266613028, 0.004767466625198186, 0.005808686068957124, 0.005478234118868046, 0.005381042368841846, 0.006090964716859304, 0.004632243320813909, 0.005048900127447965, 0.007080630275821735, 0.0071566933845378915, 0.00818946137177281, 0.006939490951870645, 0.007090772023650556, 0.008796275950197257, 0.006948787554047065, 0.005850943351577211, 0.006692708421369339, 0.0068127191040103854, 0.00644000987130122, 0.005800234612433106, 0.006998651147538767, 0.00827651137397019, 0.007279239504136142, 0.008460753126193769, 0.007519260869418235, 0.008385535163130014, 0.0072741686302217326, 0.007920705054309059, 0.007002031730148374, 0.008639924004502936, 0.0085249841957763, 0.007447423488964088, 0.008295104578323028, 0.00806691525217456, 0.007973104084757968, 0.0076756128151125565, 0.008341587589205123, 0.008494558952289838, 0.008775147308887213, 0.009415767713407729, 0.0092070167372645, 0.00971410412870554, 0.009282234700328254, 0.01353331733190898, 0.009045593917655768, 0.008305246326151849, 0.009347310915563189, 0.010095264817938723, 0.00887571964152302, 0.011169444942141329, 0.010332750746263612, 0.008727819152352717, 0.009092922074190267, 0.010007369670088943, 0.00942083858732214, 0.010587139587636534, 0.010189921131007718, 0.010500089585439154, 0.01058544929633173, 0.009615222087374538, 0.00985439830700423, 0.011415382326990233, 0.014287187253851329, 0.015996071763007637, 0.012411809051171879, 0.008520758467514291, 0.008715987113219092, 0.01016710219839287, 0.007766888545571944, 0.0063909914234619196, 0.007490525917236576, 0.006925123475779816, 0.004661823418647969, 0.004492794288167623, 0.004451382151199937, 0.005052280710057571, 0.007281774941093348, 0.002608119483311754, 0.0012584218764261834, 0.000503706808831434, 0.0002476276761537083, 3.127038913886419e-05] ,
    [[0.07275069381919223, 0.045368862271410605, 0.05641601499930871, 0.07272708879199587, 0.10139370824860815, 0.12749075188755915, 0.15111601196437663, 0.15999824648369398, 0.14831375802149407, 0.0644248635123606], [0.09161531279178338, 0.06214752567693744, 0.0692436974789916, 0.09064425770308124, 0.12022408963585435, 0.13393090569561159, 0.15566760037348273, 0.14450046685340803, 0.10491129785247433, 0.02711484593837535], [0.08677802944507361, 0.06342015855039637, 0.06610985277463194, 0.08741506228765572, 0.10652604756511891, 0.12896375990939976, 0.1499858437146093, 0.1341308040770102, 0.1282559456398641, 0.04841449603624009], [0.11291406625060009, 0.0814210273643783, 0.07172347575612098, 0.07220355256841095, 0.10638502160345656, 0.12664426308209314, 0.14661545847335575, 0.1114738358137302, 0.13490158425348056, 0.0357177148343735], [0.07684295799097327, 0.05878949195694943, 0.0672375882421016, 0.06596458743201018, 0.10646915866219188, 0.1332021756741118, 0.14500636500405045, 0.17509547506075684, 0.13169772017127648, 0.03969447980557806], [0.057717533887188456, 0.042122139629791576, 0.060924063547587816, 0.06500510129718699, 0.10056843025797989, 0.10741874362337851, 0.1475003643783705, 0.219938784433756, 0.16163824515376768, 0.03716659379099257], [0.04246487867177522, 0.035919540229885055, 0.041507024265644954, 0.07215836526181353, 0.08828224776500639, 0.14687100893997446, 0.20849297573435505, 0.19173052362707535, 0.15788633461047255, 0.014687100893997445], [0.05233680957898803, 0.04499806875241406, 0.04499806875241406, 0.08478176902278872, 0.08420239474700657, 0.13016608729239088, 0.18153727307840864, 0.1966010042487447, 0.17303978370027037, 0.007338740826573967], [0.06855990589816838, 0.03948916148546463, 0.04083347336582087, 0.06301461939169888, 0.10754495042849942, 0.1256931608133087, 0.18585111745925054, 0.19710972945723407, 0.16938329692488657, 0.002520584775667955], [0.04726692764970277, 0.026968247063940843, 0.052486588371755835, 0.06437581557198782, 0.09830361026533276, 0.15180513266637669, 0.20806147600405975, 0.20110192837465565, 0.14919530230535016, 0.00043497172683775554], [0.04335585585585586, 0.05067567567567568, 0.04842342342342342, 0.05518018018018018, 0.08577327327327328, 0.12481231231231231, 0.20964714714714713, 0.23085585585585586, 0.15127627627627627, 0.0], [0.05933821189945911, 0.08574610244988864, 0.06840598154629335, 0.07922367165128857, 0.10626789691377665, 0.12424435252943047, 0.1496977410117722, 0.20442252624880689, 0.12265351574928413, 0.0], [0.06056157093044595, 0.04624701780143146, 0.043861258946595705, 0.07249036520462471, 0.10735914846760873, 0.1255276197467425, 0.17104055790053221, 0.24206276380987338, 0.13084969719214534, 0.0], [0.06409579151259025, 0.03486529318541997, 0.05405881317133298, 0.06849797499559782, 0.13805247402711746, 0.1583025180489523, 0.19211128719845044, 0.20707871104067618, 0.08293713681986266, 0.0], [0.053591790193842644, 0.03458760927404029, 0.05131128848346636, 0.07601672367920942, 0.11041429114405168, 0.14215127328012162, 0.20505511212466743, 0.22690992018244013, 0.0999619916381604, 0.0], [0.05002886280546469, 0.03444294785453146, 0.06946315181835674, 0.0931306522994035, 0.10044256301712526, 0.13257648643448144, 0.20761978064267847, 0.2568789686357514, 0.05541658649220704, 0.0], [0.04723691409812147, 0.047054532190406714, 0.06985227065475105, 0.06693416013131498, 0.10122195878168885, 0.12967353638519058, 0.2057267919022433, 0.29655298194419116, 0.03574685391209192, 0.0], [0.033524726801259494, 0.04223004260048157, 0.056491942952398594, 0.07075384330431561, 0.10742730135210224, 0.16632709761066863, 0.24689757362474532, 0.2589368401555844, 0.017410631598444155, 0.0], [0.05794972249428665, 0.030362389813907934, 0.05223636957231472, 0.0821090434214822, 0.10643160300359125, 0.1665034280117532, 0.232615083251714, 0.26101860920666015, 0.010773751224289911, 0.0], [0.05956390710866868, 0.04431838326537848, 0.06045027477397624, 0.09094132246055664, 0.0992731785144478, 0.17886899485906754, 0.24428292855876618, 0.22230101045913844, 0.0, 0.0], [0.044085552160628545, 0.03841117415975556, 0.06416412047140986, 0.07565837334497308, 0.09239051360395752, 0.221155245162229, 0.26436781609195403, 0.1997672050050924, 0.0, 0.0], [0.13252082690527614, 0.040573896945387225, 0.06710891700092564, 0.07374267201481025, 0.10166615242209194, 0.1982412835544585, 0.23218142548596113, 0.15396482567108918, 0.0, 0.0], [0.04444793466310664, 0.026543112926024817, 0.05387152505104445, 0.0832417150934506, 0.14056855662007225, 0.21752787812156432, 0.27406942044919114, 0.1597298570755458, 0.0, 0.0], [0.08852504509504648, 0.02886082974885528, 0.05661162758429305, 0.13514638545858193, 0.1318162897183294, 0.1813514638545858, 0.2579436658803941, 0.11974469265991397, 0.0, 0.0], [0.06951286261631089, 0.0341178617040686, 0.05564677978471082, 0.08739281153074256, 0.13902572523262177, 0.21729611384783798, 0.2895457033388068, 0.10746214194490057, 0.0, 0.0], [0.03766320723133579, 0.032808838299296955, 0.07080682959491129, 0.08319383997321728, 0.1576832942751925, 0.2132574489454302, 0.33093404753933714, 0.07365249414127888, 0.0, 0.0], [0.06564812604440201, 0.045953688231081403, 0.07555502506564812, 0.12174743375507281, 0.2222487467175937, 0.1739078539030795, 0.2368106946765338, 0.05812843160658868, 0.0, 0.0], [0.09376476145488899, 0.06601322626358054, 0.13804912612187056, 0.11088804912612187, 0.17312234293811998, 0.18209730751062825, 0.21551724137931033, 0.02054794520547945, 0.0, 0.0], [0.06986584107327141, 0.056862745098039215, 0.07254901960784314, 0.08813209494324045, 0.18617131062951497, 0.22456140350877193, 0.27275541795665637, 0.02910216718266254, 0.0, 0.0], [0.09950066983315065, 0.06783582998416758, 0.0679576178297406, 0.10071854828888077, 0.15722810863475825, 0.22201924247960053, 0.2843746194129826, 0.00036536353671903543, 0.0, 0.0], [0.058402860548271755, 0.05256257449344458, 0.06495828367103695, 0.09284862932061978, 0.15768772348033372, 0.28164481525625745, 0.29189511323003575, 0.0, 0.0, 0.0], [0.10165257494235204, 0.06888931591083781, 0.11462336664104535, 0.10876249039200615, 0.1723674096848578, 0.2495196003074558, 0.18418524212144505, 0.0, 0.0, 0.0], [0.08708343468742398, 0.09584042811967891, 0.08379956215032838, 0.11116516662612504, 0.18803210897591827, 0.2554123084407687, 0.17866699099975675, 0.0, 0.0, 0.0], [0.0522894698829987, 0.057345081612017915, 0.07727863642929365, 0.12841253791708795, 0.22230247002744474, 0.3114256825075834, 0.1509461216235736, 0.0, 0.0, 0.0], [0.0646546281096098, 0.07778759944437429, 0.10392726354337668, 0.14597802752872838, 0.2124005556257103, 0.2956181336027276, 0.09963379214547291, 0.0, 0.0, 0.0], [0.05321920357275772, 0.05173055452177149, 0.07517677707480462, 0.1594095025431088, 0.21945168093288675, 0.31137575983128646, 0.1296365215233842, 0.0, 0.0, 0.0], [0.07611548556430446, 0.0594488188976378, 0.11023622047244094, 0.15328083989501312, 0.20065616797900263, 0.32860892388451446, 0.07165354330708662, 0.0, 0.0, 0.0], [0.06323765117295643, 0.06090630919422993, 0.08655107096022148, 0.14643741803875857, 0.24755937636602068, 0.3578609937345184, 0.037447180533294475, 0.0, 0.0, 0.0], [0.1156865112909069, 0.0789759690858592, 0.09539910638811738, 0.13549088274363, 0.3016543895664775, 0.26929114841202756, 0.003501992512981524, 0.0, 0.0, 0.0], [0.05544776881445931, 0.08179311753293168, 0.08853262534463392, 0.16705810272643726, 0.3309506790564689, 0.27560502399673237, 0.0006126825283365669, 0.0, 0.0, 0.0], [0.04806687565308255, 0.07477069546035063, 0.10774410774410774, 0.21003134796238246, 0.33588761174968074, 0.22349936143039592, 0.0, 0.0, 0.0, 0.0], [0.07042253521126761, 0.08400759164918589, 0.12666067325941463, 0.21366496853461192, 0.32823893716911395, 0.17700529417640595, 0.0, 0.0, 0.0, 0.0], [0.058896257165336634, 0.0638417444082275, 0.12307519388557941, 0.20085422052377205, 0.35393953017871194, 0.1993930538383725, 0.0, 0.0, 0.0, 0.0], [0.06490626889739971, 0.11731505744809514, 0.13404555533158638, 0.22838137472283815, 0.3155613787542834, 0.1397903648457972, 0.0, 0.0, 0.0, 0.0], [0.07017543859649122, 0.06947833159056582, 0.10840013942140118, 0.22133147438131753, 0.3963053328685953, 0.1343092831416289, 0.0, 0.0, 0.0, 0.0], [0.051856594110115235, 0.07554417413572344, 0.13486982501067007, 0.25970977379428084, 0.38583013230900554, 0.09218950064020487, 0.0, 0.0, 0.0, 0.0], [0.050452625226312615, 0.06143633071816536, 0.11213035606517803, 0.2691611345805673, 0.44212432106216054, 0.06469523234761618, 0.0, 0.0, 0.0, 0.0], [0.06974469333855032, 0.08862369167563337, 0.17431282402425902, 0.2478724444879194, 0.3977306074537807, 0.021715739019857183, 0.0, 0.0, 0.0, 0.0], [0.047586001784475064, 0.09923664122137404, 0.15177951819173194, 0.25022305938336475, 0.4349162288093586, 0.016258550609695648, 0.0, 0.0, 0.0, 0.0], [0.07126645483431684, 0.0876078075351793, 0.15785292782569224, 0.2939173853835679, 0.38856105310939626, 0.0007943713118474807, 0.0, 0.0, 0.0, 0.0], [0.09332654100866021, 0.12256749872643913, 0.17524197656647988, 0.3365257259296994, 0.27233825776872134, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11126244106862232, 0.11765322158198009, 0.17391304347826086, 0.3451021477213201, 0.2520691461498167, 0.0, 0.0, 0.0, 0.0, 0.0], [0.08108967564129743, 0.09709561161755353, 0.21337714649141404, 0.36092855628577486, 0.24750900996396014, 0.0, 0.0, 0.0, 0.0, 0.0], [0.078837260515305, 0.12365117815459149, 0.19390002202158116, 0.38416648315349045, 0.21944505615503193, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11925025329280649, 0.10344478216818642, 0.19189463019250252, 0.3894630192502533, 0.19594731509625127, 0.0, 0.0, 0.0, 0.0, 0.0], [0.13700129340364142, 0.13839418963287234, 0.21788876728683712, 0.373694159785096, 0.13302158989155308, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16652219974959068, 0.129346046422036, 0.2013868824039295, 0.431089280554753, 0.07165559086969084, 0.0, 0.0, 0.0, 0.0, 0.0], [0.12081500762947671, 0.12736738174311102, 0.2269993716901535, 0.48056727403285165, 0.04425096490440714, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0946392509638333, 0.14273912245272627, 0.24646594455663667, 0.5047732696897375, 0.011382412337066276, 0.0, 0.0, 0.0, 0.0, 0.0], [0.12667478684531058, 0.1668696711327649, 0.2904993909866017, 0.4151731338089438, 0.0007830172263789803, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11235545843576436, 0.1736319766912501, 0.3013748520440681, 0.4126377128289174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.11653031911571848, 0.1874102291887841, 0.4264659963779429, 0.26959345531755446, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.17322246099224517, 0.2026534616462674, 0.37428758292067643, 0.249836494440811, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1455174519181846, 0.18632339472880838, 0.42281469420983003, 0.24534445914317696, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1495479204339964, 0.23625678119349006, 0.4365280289330922, 0.17766726943942135, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.17161992465466722, 0.2240267894516534, 0.4632063624947677, 0.1411469233989117, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18482193867834698, 0.2970862692820415, 0.44029708626928205, 0.07779470577032946, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23274818401937045, 0.27338075060532685, 0.4584594430992736, 0.03541162227602906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19344020938982495, 0.28324881400294455, 0.514886307868477, 0.008424668738753477, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.20596494625738357, 0.33543139343468575, 0.45676382298828316, 0.0018398373196475259, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2371967654986523, 0.35384329398642994, 0.40895994051491774, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1969428257748501, 0.4065535005489401, 0.39650367367620976, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23190095989952453, 0.4068359199784695, 0.3612631201220059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2561666799712621, 0.43585854554163006, 0.30797477448710786, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3228000331757485, 0.4686903873268641, 0.2085095794973874, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3153573728267869, 0.5256761107533805, 0.1589665164198326, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.26922155688622756, 0.6334530938123752, 0.0973253493013972, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3375230728663092, 0.6158038147138964, 0.04667311241979432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4044596912521441, 0.5861921097770154, 0.00934819897084048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.40230991337824834, 0.5961353372325461, 0.001554749389205597, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2888494528246081, 0.711150547175392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.32836688328842395, 0.6716331167115761, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42925234917608607, 0.5707476508239139, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6377702836738742, 0.3622297163261258, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7352855619121497, 0.2647144380878503, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6937655860349127, 0.30623441396508727, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8205658324265506, 0.1794341675734494, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9395662523142025, 0.06043374768579741, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9647974726390612, 0.035202527360938735, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9985355137905785, 0.001464486209421528, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] )},
    'curio_le': {500: ([0.7016661427224143, 0.054856963219113486, 0.021848475322225714, 0.02059100911662999, 0.008645080163470606, 0.008802263439170073, 0.002514932411191449, 0.005501414649481296, 0.0014146494812951901, 0.0017290160326941214, 0.002672115686890915, 0.00015718327569946557, 0.002672115686890915, 0.0011002829298962591, 0.0007859163784973279, 0.004872681546683433, 0.0014146494812951901, 0.0014146494812951901, 0.0012574662055957245, 0.0011002829298962591, 0.0015718327569946558, 0.009116629990569003, 0.0014146494812951901, 0.0011002829298962591, 0.0014146494812951901, 0.002514932411191449, 0.0006287331027978623, 0.0007859163784973279, 0.0007859163784973279, 0.0007859163784973279, 0.0006287331027978623, 0.0045583149952845015, 0.003300848789688777, 0.00047154982709839675, 0.0020433825840930524, 0.0009430996541967935, 0.007387613957874882, 0.014303678088651368, 0.0017290160326941214, 0.0012574662055957245, 0.011317195850361521, 0.004715498270983967, 0.0022005658597925182, 0.0015718327569946558, 0.0023577491354919836, 0.0014146494812951901, 0.0015718327569946558, 0.00031436655139893113, 0.0007859163784973279, 0.0006287331027978623, 0.0014146494812951901, 0.0007859163784973279, 0.00047154982709839675, 0.009430996541967935, 0.0022005658597925182, 0.0009430996541967935, 0.00675888085507702, 0.0009430996541967935, 0.0009430996541967935, 0.004715498270983967, 0.01603269412134549, 0.0009430996541967935, 0.002514932411191449, 0.0022005658597925182, 0.0015718327569946558, 0.00031436655139893113, 0.0011002829298962591, 0.0007859163784973279, 0.0006287331027978623, 0.0007859163784973279, 0.0009430996541967935, 0.00047154982709839675, 0.0007859163784973279, 0.0028292989625903803, 0.0006287331027978623, 0.00031436655139893113, 0.00015718327569946557, 0.0028292989625903803, 0.00015718327569946557, 0.00031436655139893113, 0.00015718327569946557, 0.00047154982709839675, 0.0006287331027978623, 0.00015718327569946557, 0.0006287331027978623, 0.002672115686890915, 0.0015718327569946558, 0.00015718327569946557, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
    [[0.40412186379928317, 0.1496415770609319, 0.09901433691756273, 0.06496415770609319, 0.09520609318996416, 0.09274193548387097, 0.0642921146953405, 0.026657706093189962, 0.003360215053763441, 0.0], [0.3467048710601719, 0.1318051575931232, 0.10601719197707736, 0.07163323782234957, 0.12320916905444126, 0.11174785100286533, 0.0659025787965616, 0.022922636103151862, 0.02005730659025788, 0.0], [0.2517985611510791, 0.07913669064748201, 0.04316546762589928, 0.08633093525179857, 0.16546762589928057, 0.2302158273381295, 0.11510791366906475, 0.007194244604316547, 0.02158273381294964, 0.0], [0.26717557251908397, 0.12213740458015267, 0.05343511450381679, 0.11450381679389313, 0.12213740458015267, 0.21374045801526717, 0.10687022900763359, 0.0, 0.0, 0.0], [0.4727272727272727, 0.10909090909090909, 0.12727272727272726, 0.03636363636363636, 0.12727272727272726, 0.09090909090909091, 0.01818181818181818, 0.01818181818181818, 0.0, 0.0], [0.23214285714285715, 0.19642857142857142, 0.03571428571428571, 0.03571428571428571, 0.125, 0.10714285714285714, 0.25, 0.017857142857142856, 0.0, 0.0], [0.3125, 0.0625, 0.25, 0.0625, 0.0, 0.1875, 0.125, 0.0, 0.0, 0.0], [0.4857142857142857, 0.08571428571428572, 0.05714285714285714, 0.02857142857142857, 0.11428571428571428, 0.08571428571428572, 0.05714285714285714, 0.08571428571428572, 0.0, 0.0], [0.2222222222222222, 0.1111111111111111, 0.1111111111111111, 0.0, 0.2222222222222222, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0], [0.36363636363636365, 0.09090909090909091, 0.09090909090909091, 0.2727272727272727, 0.09090909090909091, 0.0, 0.09090909090909091, 0.0, 0.0, 0.0], [0.47058823529411764, 0.17647058823529413, 0.058823529411764705, 0.0, 0.11764705882352941, 0.17647058823529413, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47058823529411764, 0.058823529411764705, 0.0, 0.17647058823529413, 0.11764705882352941, 0.11764705882352941, 0.058823529411764705, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.06451612903225806, 0.0, 0.03225806451612903, 0.0, 0.0, 0.8709677419354839, 0.03225806451612903, 0.0, 0.0, 0.0], [0.1111111111111111, 0.0, 0.1111111111111111, 0.0, 0.0, 0.0, 0.7777777777777778, 0.0, 0.0, 0.0], [0.6666666666666666, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.125, 0.25, 0.125, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0], [0.5714285714285714, 0.0, 0.0, 0.0, 0.0, 0.14285714285714285, 0.2857142857142857, 0.0, 0.0, 0.0], [0.4, 0.3, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], [0.7758620689655172, 0.1206896551724138, 0.10344827586206896, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5555555555555556, 0.3333333333333333, 0.0, 0.0, 0.1111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4444444444444444, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.2222222222222222, 0.0, 0.0, 0.0], [0.0, 0.125, 0.6875, 0.0, 0.0625, 0.0, 0.125, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.2, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8275862068965517, 0.034482758620689655, 0.13793103448275862, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.09523809523809523, 0.8095238095238095, 0.09523809523809523, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6666666666666666, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8461538461538461, 0.15384615384615385, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.16666666666666666, 0.0, 0.0, 0.6666666666666666, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9230769230769231, 0.03296703296703297, 0.0, 0.0, 0.04395604395604396, 0.0, 0.0, 0.0, 0.0, 0.0], [0.36363636363636365, 0.5454545454545454, 0.0, 0.0, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0], [0.875, 0.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.19444444444444445, 0.0, 0.013888888888888888, 0.7916666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.5, 0.06666666666666667, 0.0, 0.03333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7142857142857143, 0.14285714285714285, 0.0, 0.07142857142857142, 0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7333333333333333, 0.0, 0.26666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7777777777777778, 0.0, 0.0, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2222222222222222, 0.0, 0.7777777777777778, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.45, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6666666666666666, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9767441860465116, 0.0, 0.0, 0.023255813953488372, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8333333333333334, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9666666666666667, 0.03333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8725490196078431, 0.12745098039215685, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6666666666666666, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1875, 0.8125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2857142857142857, 0.0, 0.7142857142857143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
    1000: ([0.620247171567107, 0.049483290614337254, 0.02017860349697158, 0.01758281227041941, 0.012978956132760844, 0.013583008179191223, 0.007444533328979805, 0.004914045026366056, 0.01039949063719328, 0.004571204675689354, 0.00749351052193362, 0.009534226895009225, 0.005958891809380765, 0.004407947365843306, 0.005697680113627088, 0.0175664865394348, 0.005779308768550112, 0.0019101105251987658, 0.004701810523566193, 0.0026937456124597977, 0.0020896935660294188, 0.004620181868643169, 0.0020896935660294188, 0.004652833330612378, 0.003591660816613064, 0.0021386707589832335, 0.004016129822212789, 0.0076241163698104585, 0.005403816955904201, 0.009142409351378708, 0.004701810523566193, 0.014317666073498441, 0.005387491224919595, 0.002187647951937048, 0.0030039345011672897, 0.0028406771913212416, 0.0051752567221197325, 0.004685484792581588, 0.0034610549687362252, 0.0017142017533835078, 0.007150670171256918, 0.00648131520088812, 0.0018121561392911367, 0.0030039345011672897, 0.002203973682921653, 0.0031182146180595235, 0.0013550356717222014, 0.0017795046773219272, 0.0013060584787683868, 0.0013550356717222014, 0.001191778361876153, 0.001893784794214161, 0.001208104092860758, 0.004309992979935676, 0.0018284818702757415, 0.0007673093562764273, 0.0023345795307984914, 0.000799960818245637, 0.000897915204153266, 0.0011264754379377336, 0.0023019280688292816, 0.001093823975968524, 0.0007673093562764273, 0.0015182929815682497, 0.0013550356717222014, 0.0007673093562764273, 0.011134148531500498, 0.001191778361876153, 0.000995869590060895, 0.001975413449137185, 0.00047344619855354023, 0.00031018888870749185, 0.0008489380111994515, 0.0006856807013534031, 0.00040814327461512087, 0.0004407947365843306, 0.00024488596476907254, 0.0008326122802148466, 0.0015672701745220643, 0.0003754918126459112, 0.0006203777774149837, 0.0005387491224919595, 0.0005060976605227499, 0.0008489380111994515, 0.0005060976605227499, 0.00035916608166130636, 0.00031018888870749185, 0.00019590877181525804, 0.00016325730984604835, 0.0004244690055997257, 0.00013060584787683867, 1.6325730984604834e-05, 0.00011428011689223384, 1.6325730984604834e-05, 6.530292393841934e-05, 0.0, 0.0, 0.0, 0.0, 0.0] ,
    [[0.4042429985260055, 0.15674352495262162, 0.12965887555274794, 0.10196883554432512, 0.07096230785428512, 0.05445883343861865, 0.04403558643925037, 0.027163613392293114, 0.010160033691303432, 0.0006053906085491682], [0.30649950511382384, 0.18277796106895414, 0.1329594193335533, 0.10755526228967338, 0.08116133289343451, 0.06994391290003299, 0.06103596172880237, 0.037611349389640385, 0.020125371164632134, 0.0003299241174529858], [0.24271844660194175, 0.16181229773462782, 0.13754045307443366, 0.17637540453074432, 0.09466019417475728, 0.08090614886731391, 0.05177993527508091, 0.03802588996763754, 0.015372168284789644, 0.0008090614886731392], [0.26369545032497677, 0.21448467966573817, 0.11420612813370473, 0.1392757660167131, 0.08449396471680594, 0.08449396471680594, 0.0584958217270195, 0.035283194057567316, 0.005571030640668524, 0.0], [0.24150943396226415, 0.17358490566037735, 0.21635220125786164, 0.11320754716981132, 0.09056603773584905, 0.0830188679245283, 0.045283018867924525, 0.033962264150943396, 0.0025157232704402514, 0.0], [0.2932692307692308, 0.20432692307692307, 0.15985576923076922, 0.15625, 0.04326923076923077, 0.030048076923076924, 0.037259615384615384, 0.039663461538461536, 0.036057692307692304, 0.0], [0.20175438596491227, 0.05043859649122807, 0.11842105263157894, 0.17543859649122806, 0.125, 0.15789473684210525, 0.06578947368421052, 0.08991228070175439, 0.015350877192982455, 0.0], [0.2591362126245847, 0.132890365448505, 0.16943521594684385, 0.13953488372093023, 0.06976744186046512, 0.059800664451827246, 0.05647840531561462, 0.10299003322259136, 0.009966777408637873, 0.0], [0.41130298273155413, 0.27472527472527475, 0.0847723704866562, 0.07221350078492936, 0.04552590266875981, 0.0141287284144427, 0.03767660910518053, 0.04866562009419152, 0.01098901098901099, 0.0], [0.25, 0.125, 0.09285714285714286, 0.15714285714285714, 0.225, 0.07857142857142857, 0.039285714285714285, 0.03214285714285714, 0.0, 0.0], [0.09586056644880174, 0.19825708061002179, 0.18736383442265794, 0.3202614379084967, 0.06318082788671024, 0.08932461873638345, 0.0196078431372549, 0.026143790849673203, 0.0, 0.0], [0.3082191780821918, 0.3082191780821918, 0.1541095890410959, 0.07534246575342465, 0.06506849315068493, 0.053082191780821915, 0.0273972602739726, 0.008561643835616438, 0.0, 0.0], [0.4438356164383562, 0.07945205479452055, 0.06575342465753424, 0.12876712328767123, 0.09315068493150686, 0.12602739726027398, 0.03561643835616438, 0.0273972602739726, 0.0, 0.0], [0.3333333333333333, 0.09259259259259259, 0.17037037037037037, 0.10740740740740741, 0.05925925925925926, 0.06296296296296296, 0.15925925925925927, 0.014814814814814815, 0.0, 0.0], [0.2148997134670487, 0.09455587392550144, 0.5243553008595988, 0.06017191977077364, 0.04297994269340974, 0.05157593123209169, 0.008595988538681949, 0.0028653295128939827, 0.0, 0.0], [0.04553903345724907, 0.7574349442379182, 0.09293680297397769, 0.046468401486988845, 0.012081784386617101, 0.03345724907063197, 0.011152416356877323, 0.0009293680297397769, 0.0, 0.0], [0.2937853107344633, 0.10451977401129943, 0.1016949152542373, 0.3983050847457627, 0.025423728813559324, 0.011299435028248588, 0.05649717514124294, 0.00847457627118644, 0.0, 0.0], [0.26495726495726496, 0.17094017094017094, 0.1282051282051282, 0.1794871794871795, 0.20512820512820512, 0.008547008547008548, 0.02564102564102564, 0.017094017094017096, 0.0, 0.0], [0.4479166666666667, 0.17708333333333334, 0.125, 0.06597222222222222, 0.06944444444444445, 0.03819444444444445, 0.06944444444444445, 0.006944444444444444, 0.0, 0.0], [0.23030303030303031, 0.06666666666666667, 0.07878787878787878, 0.05454545454545454, 0.048484848484848485, 0.09090909090909091, 0.42424242424242425, 0.006060606060606061, 0.0, 0.0], [0.2421875, 0.0859375, 0.046875, 0.3359375, 0.078125, 0.1640625, 0.0390625, 0.0078125, 0.0, 0.0], [0.23674911660777384, 0.053003533568904596, 0.13074204946996468, 0.04240282685512368, 0.29328621908127206, 0.07420494699646643, 0.1625441696113074, 0.007067137809187279, 0.0, 0.0], [0.359375, 0.1484375, 0.125, 0.109375, 0.109375, 0.0859375, 0.0625, 0.0, 0.0, 0.0], [0.28421052631578947, 0.02456140350877193, 0.3684210526315789, 0.2596491228070175, 0.02456140350877193, 0.028070175438596492, 0.010526315789473684, 0.0, 0.0, 0.0], [0.3, 0.02727272727272727, 0.18181818181818182, 0.38181818181818183, 0.05909090909090909, 0.01818181818181818, 0.031818181818181815, 0.0, 0.0, 0.0], [0.25190839694656486, 0.08396946564885496, 0.2366412213740458, 0.061068702290076333, 0.29770992366412213, 0.04580152671755725, 0.022900763358778626, 0.0, 0.0, 0.0], [0.4105691056910569, 0.10569105691056911, 0.08943089430894309, 0.10569105691056911, 0.22764227642276422, 0.04878048780487805, 0.012195121951219513, 0.0, 0.0, 0.0], [0.42398286937901497, 0.15417558886509636, 0.16274089935760172, 0.12419700214132762, 0.09207708779443255, 0.03640256959314775, 0.006423982869379015, 0.0, 0.0, 0.0], [0.459214501510574, 0.1782477341389728, 0.09365558912386707, 0.11178247734138973, 0.08157099697885196, 0.06948640483383686, 0.006042296072507553, 0.0, 0.0, 0.0], [0.45535714285714285, 0.125, 0.13392857142857142, 0.07321428571428572, 0.2017857142857143, 0.008928571428571428, 0.0017857142857142857, 0.0, 0.0, 0.0], [0.4027777777777778, 0.1388888888888889, 0.06944444444444445, 0.2743055555555556, 0.027777777777777776, 0.04861111111111111, 0.03819444444444445, 0.0, 0.0, 0.0], [0.22234891676168758, 0.09806157354618016, 0.5427594070695553, 0.09578107183580388, 0.037628278221208664, 0.0034207525655644243, 0.0, 0.0, 0.0, 0.0], [0.2727272727272727, 0.20909090909090908, 0.08484848484848485, 0.3787878787878788, 0.03333333333333333, 0.01818181818181818, 0.0030303030303030303, 0.0, 0.0, 0.0], [0.20149253731343283, 0.17164179104477612, 0.13432835820895522, 0.23134328358208955, 0.12686567164179105, 0.13432835820895522, 0.0, 0.0, 0.0, 0.0], [0.2826086956521739, 0.21195652173913043, 0.11413043478260869, 0.2826086956521739, 0.07065217391304347, 0.03804347826086957, 0.0, 0.0, 0.0, 0.0], [0.21839080459770116, 0.08620689655172414, 0.06896551724137931, 0.3735632183908046, 0.15517241379310345, 0.09770114942528736, 0.0, 0.0, 0.0, 0.0], [0.48264984227129337, 0.06624605678233439, 0.056782334384858045, 0.056782334384858045, 0.05362776025236593, 0.28391167192429023, 0.0, 0.0, 0.0, 0.0], [0.5017421602787456, 0.18815331010452963, 0.03832752613240418, 0.07317073170731707, 0.1951219512195122, 0.003484320557491289, 0.0, 0.0, 0.0, 0.0], [0.46226415094339623, 0.27358490566037735, 0.1320754716981132, 0.07075471698113207, 0.05188679245283019, 0.009433962264150943, 0.0, 0.0, 0.0, 0.0], [0.6190476190476191, 0.08571428571428572, 0.08571428571428572, 0.13333333333333333, 0.0380952380952381, 0.0380952380952381, 0.0, 0.0, 0.0, 0.0], [0.0410958904109589, 0.2808219178082192, 0.08904109589041095, 0.2945205479452055, 0.2899543378995434, 0.0045662100456621, 0.0, 0.0, 0.0, 0.0], [0.3425692695214106, 0.08816120906801007, 0.2141057934508816, 0.055415617128463476, 0.2972292191435768, 0.0025188916876574307, 0.0, 0.0, 0.0, 0.0], [0.3153153153153153, 0.10810810810810811, 0.3963963963963964, 0.14414414414414414, 0.02702702702702703, 0.009009009009009009, 0.0, 0.0, 0.0, 0.0], [0.23369565217391305, 0.30978260869565216, 0.16847826086956522, 0.25, 0.03804347826086957, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4888888888888889, 0.17037037037037037, 0.21481481481481482, 0.08888888888888889, 0.037037037037037035, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3403141361256545, 0.1518324607329843, 0.2513089005235602, 0.21465968586387435, 0.041884816753926704, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1927710843373494, 0.37349397590361444, 0.24096385542168675, 0.1566265060240964, 0.03614457831325301, 0.0, 0.0, 0.0, 0.0, 0.0], [0.22935779816513763, 0.28440366972477066, 0.3853211009174312, 0.045871559633027525, 0.05504587155963303, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.275, 0.2, 0.2375, 0.0375, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4819277108433735, 0.03614457831325301, 0.1686746987951807, 0.27710843373493976, 0.03614457831325301, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4657534246575342, 0.1506849315068493, 0.0821917808219178, 0.3013698630136986, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.49137931034482757, 0.13793103448275862, 0.3017241379310345, 0.06896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21621621621621623, 0.08108108108108109, 0.14864864864864866, 0.5540540540540541, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.14015151515151514, 0.14393939393939395, 0.03409090909090909, 0.6818181818181818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5892857142857143, 0.26785714285714285, 0.07142857142857142, 0.0625, 0.008928571428571428, 0.0, 0.0, 0.0, 0.0, 0.0], [0.46808510638297873, 0.2553191489361702, 0.2127659574468085, 0.06382978723404255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6363636363636364, 0.06293706293706294, 0.16783216783216784, 0.13286713286713286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.40816326530612246, 0.4897959183673469, 0.10204081632653061, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8363636363636363, 0.07272727272727272, 0.07272727272727272, 0.01818181818181818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7391304347826086, 0.14492753623188406, 0.08695652173913043, 0.028985507246376812, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8085106382978723, 0.12056737588652482, 0.07092198581560284, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3880597014925373, 0.3880597014925373, 0.22388059701492538, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3829787234042553, 0.5319148936170213, 0.06382978723404255, 0.02127659574468085, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.26881720430107525, 0.3978494623655914, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25301204819277107, 0.6385542168674698, 0.10843373493975904, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8297872340425532, 0.06382978723404255, 0.10638297872340426, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.08211143695014662, 0.9090909090909091, 0.008797653958944282, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3013698630136986, 0.6712328767123288, 0.0273972602739726, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6557377049180327, 0.3442622950819672, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.371900826446281, 0.5867768595041323, 0.04132231404958678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7241379310344828, 0.2413793103448276, 0.034482758620689655, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8421052631578947, 0.10526315789473684, 0.05263157894736842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5192307692307693, 0.4807692307692308, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9285714285714286, 0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7777777777777778, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.6666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9803921568627451, 0.0196078431372549, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9166666666666666, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9130434782608695, 0.08695652173913043, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2894736842105263, 0.7105263157894737, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9393939393939394, 0.06060606060606061, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
    2000: ([0.5248436473227437, 0.032842559651945345, 0.02388910289818948, 0.017295098683465137, 0.013830980489905055, 0.012106002583218147, 0.02277310733951191, 0.019578074369490835, 0.009454805012349595, 0.01603181437085042, 0.007202986562733679, 0.008120708798803562, 0.006367406131744125, 0.00626826947044028, 0.00702737304842401, 0.010497156194058598, 0.006574176882463575, 0.007013210668237747, 0.00810937889465455, 0.009593596338174979, 0.007211483990845437, 0.006460877840973466, 0.007123677233690603, 0.008542747728354219, 0.004999320205751059, 0.0054610137998232534, 0.011672633749518478, 0.006783780109220276, 0.005495003512270286, 0.006823434773741814, 0.0071293421857651085, 0.01011477192902948, 0.008619224581360042, 0.003192200493983821, 0.006290929278738302, 0.006511862409644014, 0.004483809566971063, 0.004067435589494913, 0.010542475810654641, 0.004741564886361061, 0.0035859146631619498, 0.004410165190002493, 0.0027928213727311866, 0.006038838911422809, 0.0025860506220117376, 0.002758831660284154, 0.003880492171036233, 0.00485486392785117, 0.00220366635698262, 0.0018496068523260294, 0.0037077111327638167, 0.006390065940042147, 0.0015720242006752622, 0.0024557567242981123, 0.0021186920758650384, 0.0025350660533411886, 0.00450080442319458, 0.001699485622351635, 0.0010111939452992229, 0.0022178287371688835, 0.001492714871632186, 0.0022064988330198726, 0.001277446692800979, 0.0013652534499558134, 0.0015465319163399877, 0.0017023180983888877, 0.003141215925313272, 0.002699349663501847, 0.0014842174435204278, 0.0014728875393714169, 0.0008100881466542793, 0.0013850807822165824, 0.0013114364052480116, 0.0008270830028777957, 0.0004928508304819741, 0.0004900183544447213, 0.0006684643447916431, 0.0006089823480093359, 0.0006627993927171376, 0.00031157236409779974, 0.001031021277559992, 0.0005608302553760396, 0.00028891255579977795, 0.0003653894088056015, 0.00040504407332713963, 0.0006684643447916431, 0.0002379279871292289, 0.0006514694885681267, 9.630418526659264e-05, 0.00016428361016065804, 0.00014445627789988897, 0.00015012122997439442, 0.00010763408941560355, 0.00015295370601164715, 5.3817044707801775e-05, 4.815209263329632e-05, 5.948199678230722e-05, 0.0, 0.0, 0.0] ,
    [[0.3044280741520278, 0.17695026849078496, 0.16322081005963462, 0.14042472813621523, 0.09609541541865674, 0.05688226881459295, 0.03341158692895113, 0.017836423001160312, 0.009012655495291292, 0.001737769502684908], [0.2944372574385511, 0.18180250107805088, 0.15929279862009488, 0.13445450625269512, 0.09279862009486847, 0.04941785252263907, 0.03648124191461837, 0.02630444156964209, 0.02285467874083657, 0.0021561017680034496], [0.2895423286696704, 0.19373962532606118, 0.17453165757647618, 0.1588807208916291, 0.06805786103865306, 0.05371116907754328, 0.027507706900640267, 0.025254920559639554, 0.008536874555371117, 0.00023713540431586437], [0.28922371437929906, 0.23157549950867998, 0.1912872584343269, 0.10301342941369145, 0.05977726826072716, 0.041107107762856206, 0.03930560104814936, 0.034719947592531934, 0.006223386832623649, 0.0037667867671143137], [0.36084374360024574, 0.1775547818963752, 0.1736637313127176, 0.11386442760597994, 0.07659225885725988, 0.03972967438050379, 0.0358386237968462, 0.01679295515052222, 0.005119803399549457, 0.0], [0.2767898923724848, 0.2477772578380908, 0.18671034160037436, 0.10645765091249415, 0.06855404773046327, 0.03532990173139916, 0.033926064576509124, 0.026204960224613945, 0.018249883013570427, 0.0], [0.48308457711442787, 0.17114427860696518, 0.11579601990049751, 0.11890547263681592, 0.04664179104477612, 0.03072139303482587, 0.015298507462686567, 0.012562189054726369, 0.005845771144278607, 0.0], [0.6187789351851852, 0.16174768518518517, 0.06901041666666667, 0.07465277777777778, 0.026475694444444444, 0.019675925925925927, 0.01461226851851852, 0.012297453703703705, 0.0027488425925925927, 0.0], [0.37177950868783705, 0.21420011983223486, 0.14619532654284004, 0.09257040143798682, 0.07938885560215699, 0.03505092869982025, 0.0329538645895746, 0.025464349910125823, 0.002396644697423607, 0.0], [0.4742049469964664, 0.12243816254416962, 0.16395759717314487, 0.07632508833922262, 0.0666077738515901, 0.03657243816254417, 0.027738515901060072, 0.0196113074204947, 0.01254416961130742, 0.0], [0.2874557609123083, 0.15650806134486828, 0.16515926071569012, 0.19937082186394023, 0.048368069209594966, 0.06449075894612662, 0.03814392449862367, 0.030279197797876523, 0.010224144710971293, 0.0], [0.2807813044994768, 0.24276246948029298, 0.15660969654691315, 0.11126613184513429, 0.09103592605510986, 0.05266829438437391, 0.024066968957098013, 0.0355772584583188, 0.005231949773282177, 0.0], [0.3429715302491103, 0.17126334519572953, 0.16637010676156583, 0.13745551601423486, 0.0947508896797153, 0.04448398576512456, 0.02402135231316726, 0.018683274021352312, 0.0, 0.0], [0.254405784003615, 0.22729326705829192, 0.25259828287392677, 0.09986443741527339, 0.051965657478535925, 0.05061003163126977, 0.04609127880704925, 0.012200632625395391, 0.004970628106642567, 0.0], [0.2361950826279726, 0.25957275292220877, 0.2543329302700524, 0.12293430068520758, 0.04877065699314793, 0.0354695687222894, 0.026199113260781944, 0.014510278113663845, 0.002015316404675534, 0.0], [0.36859147328656233, 0.3248785752833243, 0.11683756071235833, 0.0744738262277388, 0.03291958985429034, 0.04749055585536967, 0.024015110631408525, 0.010523475445223961, 0.0002698327037236913, 0.0], [0.32399827660491165, 0.18828091339939682, 0.1934510986643688, 0.14691943127962084, 0.08013787160706592, 0.023265833692373977, 0.025420077552778975, 0.01852649719948298, 0.0, 0.0], [0.3501615508885299, 0.14176090468497576, 0.2701938610662359, 0.0981421647819063, 0.04361873990306947, 0.049273021001615507, 0.026252019386106624, 0.02059773828756058, 0.0, 0.0], [0.318546978693678, 0.2315752706950751, 0.2553265805099546, 0.08278030038421237, 0.06147397834439399, 0.028641285365001747, 0.018861334264757248, 0.0027942717429269995, 0.0, 0.0], [0.5231768526719811, 0.22763507528786536, 0.08591674047829938, 0.07056392087392974, 0.030410392677886033, 0.020962503690581637, 0.03690581635665781, 0.0044286979627989375, 0.0, 0.0], [0.4662215239591516, 0.17871170463472114, 0.11468970934799685, 0.08523173605655932, 0.03652788688138256, 0.05420267085624509, 0.05498821681068342, 0.009426551453260016, 0.0, 0.0], [0.24024550635686104, 0.23805348531345902, 0.18719859710653222, 0.055677334502411226, 0.17360806663743972, 0.05918456817185445, 0.03857957036387549, 0.007452871547566856, 0.0, 0.0], [0.32445328031809145, 0.258051689860835, 0.22584493041749504, 0.07833001988071571, 0.04294234592445328, 0.04850894632206759, 0.018290258449304174, 0.0035785288270377734, 0.0, 0.0], [0.4986737400530504, 0.16047745358090185, 0.1289787798408488, 0.08885941644562334, 0.06465517241379311, 0.03149867374005305, 0.020225464190981434, 0.006631299734748011, 0.0, 0.0], [0.3059490084985836, 0.24475920679886687, 0.11444759206798867, 0.09971671388101983, 0.04475920679886686, 0.14390934844192635, 0.0339943342776204, 0.012464589235127478, 0.0, 0.0], [0.4118257261410788, 0.2474066390041494, 0.10840248962655602, 0.09906639004149377, 0.0700207468879668, 0.038381742738589214, 0.02437759336099585, 0.0005186721991701245, 0.0, 0.0], [0.5192914341179325, 0.19388497937393837, 0.09439456442611016, 0.07231254549866538, 0.08565882067459354, 0.026449890803203105, 0.008007765105556904, 0.0, 0.0, 0.0], [0.4254697286012526, 0.20167014613778705, 0.15615866388308977, 0.0709812108559499, 0.056784968684759914, 0.0651356993736952, 0.023799582463465554, 0.0, 0.0, 0.0], [0.45824742268041235, 0.13659793814432988, 0.0865979381443299, 0.11185567010309279, 0.08195876288659794, 0.08969072164948454, 0.03505154639175258, 0.0, 0.0, 0.0], [0.43129929431299296, 0.1087588210875882, 0.1523453715234537, 0.16770444167704443, 0.07513491075134911, 0.04234122042341221, 0.0224159402241594, 0.0, 0.0, 0.0], [0.4600715137067938, 0.21771950735001985, 0.11402463249900675, 0.08820023837902265, 0.04648390941597139, 0.03853794199443782, 0.03496225665474772, 0.0, 0.0, 0.0], [0.56482777933352, 0.10865303836460376, 0.20498459815177822, 0.053486418370204424, 0.03668440212825539, 0.024642957154858584, 0.006720806496779613, 0.0, 0.0, 0.0], [0.42195202103187646, 0.27571475517581334, 0.06638186000657247, 0.08971409792967466, 0.09168583634571147, 0.040749260598093986, 0.013802168912257641, 0.0, 0.0, 0.0], [0.30168589174800353, 0.21384205856255545, 0.08251996450754215, 0.17125110913930788, 0.06299911268855368, 0.15527950310559005, 0.012422360248447204, 0.0, 0.0, 0.0], [0.6042323277802791, 0.13462404322377308, 0.11166141377757767, 0.06348491670418731, 0.042323277802791534, 0.03872129671319226, 0.00495272399819901, 0.0, 0.0, 0.0], [0.40843845150065244, 0.13266637668551545, 0.09525880817746847, 0.08699434536755112, 0.19791213571117877, 0.07872988255763376, 0.0, 0.0, 0.0, 0.0], [0.4649399873657612, 0.07833228048010107, 0.15540113708149084, 0.13644977890082122, 0.07959570435881239, 0.08401768793430196, 0.0012634238787113076, 0.0, 0.0, 0.0], [0.28133704735376047, 0.18245125348189414, 0.2903899721448468, 0.04387186629526462, 0.12534818941504178, 0.0766016713091922, 0.0, 0.0, 0.0, 0.0], [0.2775389575497045, 0.14535196131112305, 0.06179473401397098, 0.18027941966684577, 0.3167651800107469, 0.018269747447608814, 0.0, 0.0, 0.0, 0.0], [0.4217443249701314, 0.1917562724014337, 0.1816009557945042, 0.06750298685782556, 0.12485065710872162, 0.012544802867383513, 0.0, 0.0, 0.0, 0.0], [0.23459715639810427, 0.20695102685624012, 0.12243285939968404, 0.17772511848341233, 0.21642969984202212, 0.04186413902053712, 0.0, 0.0, 0.0, 0.0], [0.3230571612074502, 0.2556197816313423, 0.15350032113037893, 0.12010276172125883, 0.1252408477842004, 0.0224791265253693, 0.0, 0.0, 0.0, 0.0], [0.22920892494929007, 0.18052738336713997, 0.1206896551724138, 0.21602434077079108, 0.2363083164300203, 0.017241379310344827, 0.0, 0.0, 0.0, 0.0], [0.37476547842401503, 0.099906191369606, 0.35412757973733583, 0.053939962476547844, 0.1125703564727955, 0.004690431519699813, 0.0, 0.0, 0.0, 0.0], [0.4819277108433735, 0.10405257393209201, 0.09200438116100766, 0.2497261774370208, 0.056955093099671415, 0.01533406352683461, 0.0, 0.0, 0.0, 0.0], [0.5061601642710473, 0.13655030800821355, 0.13039014373716631, 0.11293634496919917, 0.11293634496919917, 0.001026694045174538, 0.0, 0.0, 0.0, 0.0], [0.2759124087591241, 0.0708029197080292, 0.25547445255474455, 0.23357664233576642, 0.16423357664233576, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2969661610268378, 0.4352392065344224, 0.14294049008168028, 0.07526254375729288, 0.049591598599766626, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3444730077120823, 0.21079691516709512, 0.21465295629820053, 0.17352185089974292, 0.056555269922879174, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44104134762633995, 0.21745788667687596, 0.20367534456355282, 0.10872894333843798, 0.02909647779479326, 0.0, 0.0, 0.0, 0.0, 0.0], [0.26585179526355995, 0.5553857906799083, 0.05653170359052712, 0.11077158135981666, 0.01145912910618793, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1972517730496454, 0.5660460992907801, 0.08687943262411348, 0.13652482269503546, 0.013297872340425532, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2756756756756757, 0.42342342342342343, 0.13513513513513514, 0.15495495495495495, 0.010810810810810811, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23990772779700115, 0.33679354094579006, 0.1245674740484429, 0.28027681660899656, 0.01845444059976932, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3195187165775401, 0.42914438502673796, 0.12967914438502673, 0.11764705882352941, 0.004010695187165776, 0.0, 0.0, 0.0, 0.0, 0.0], [0.32849162011173183, 0.1329608938547486, 0.4223463687150838, 0.11173184357541899, 0.004469273743016759, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5047199496538703, 0.34675896790434235, 0.11390811831340465, 0.034612964128382634, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.42, 0.21333333333333335, 0.20333333333333334, 0.16333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4369747899159664, 0.30532212885154064, 0.15966386554621848, 0.09803921568627451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5644955300127714, 0.2669220945083014, 0.11749680715197956, 0.05108556832694764, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4781783681214421, 0.23719165085388993, 0.1555977229601518, 0.12903225806451613, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27471116816431324, 0.3080872913992298, 0.3594351732991014, 0.057766367137355584, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.37250554323725055, 0.28824833702882485, 0.2660753880266075, 0.07317073170731707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.487551867219917, 0.22406639004149378, 0.24066390041493776, 0.04771784232365145, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2967032967032967, 0.24725274725274726, 0.43772893772893773, 0.018315018315018316, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5174708818635607, 0.3560732113144759, 0.12312811980033278, 0.0033277870216306157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.18394950405770966, 0.6988277727682597, 0.11722272317403065, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6988457502623295, 0.2518363064008394, 0.04931794333683106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5076335877862596, 0.3148854961832061, 0.17748091603053434, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.49230769230769234, 0.40576923076923077, 0.10192307692307692, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6538461538461539, 0.15734265734265734, 0.1888111888111888, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4703476482617587, 0.45807770961145194, 0.07157464212678936, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7429805615550756, 0.23758099352051837, 0.019438444924406047, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6883561643835616, 0.2568493150684932, 0.0547945205479452, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6896551724137931, 0.27586206896551724, 0.034482758620689655, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7109826589595376, 0.2658959537572254, 0.023121387283236993, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4745762711864407, 0.5211864406779662, 0.00423728813559322, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.772093023255814, 0.22790697674418606, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7606837606837606, 0.23931623931623933, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8363636363636363, 0.16363636363636364, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6236263736263736, 0.37637362637362637, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8484848484848485, 0.15151515151515152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8921568627450981, 0.10784313725490197, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8604651162790697, 0.13953488372093023, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9370629370629371, 0.06293706293706294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9703389830508474, 0.029661016949152543, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9880952380952381, 0.011904761904761904, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ),
    math.inf: ([0.25945452693035187, 0.0160018427545637, 0.012672481384192177, 0.010301706762820655, 0.008544002409755967, 0.0077165347354805555, 0.010483324400108084, 0.010012890422548937, 0.00825164231070791, 0.010748220368639507, 0.006066915025094242, 0.006279540551674648, 0.006211323195230101, 0.006857173232218083, 0.00657633034919313, 0.007149533331266141, 0.0064248346615045915, 0.008355297254915859, 0.008140013909253197, 0.0067003619063650335, 0.005692162534496277, 0.006702133785753204, 0.006777438659750431, 0.007235469481592388, 0.006059827507541562, 0.006692388449118268, 0.009526509530496259, 0.006973231332143221, 0.007860942905616414, 0.007015756437459302, 0.007892836734603476, 0.010276900451386274, 0.007534917098193126, 0.0057338017001182726, 0.00730457277773102, 0.009398934214548016, 0.006833252860477788, 0.0064655878874325026, 0.00791321334756743, 0.008605132248647835, 0.01215952230131695, 0.008180767135181108, 0.008070024673420481, 0.009023295784255966, 0.011031721070746714, 0.008977226920163544, 0.008267589225201441, 0.00941931082751197, 0.006897040518451909, 0.007173453703006437, 0.0077014737606811105, 0.008911667382801252, 0.0080629371558678, 0.007062711241245808, 0.009059619311713452, 0.008132040452006432, 0.009331602797797553, 0.00883370468972177, 0.005940225648840084, 0.009432599922923246, 0.008148873306194047, 0.009642567630421397, 0.00808951534669035, 0.007179655280865032, 0.007458726284501814, 0.006816420006290172, 0.008736251323372418, 0.009772800765451896, 0.009258955742882583, 0.0066507492834962725, 0.007804242765194973, 0.009058733372019366, 0.011915002945749483, 0.009310340245139513, 0.0066206273338973816, 0.00693956562376799, 0.01538877248625686, 0.007911441468179261, 0.006971459452755051, 0.007823733438464844, 0.0073568432196820365, 0.011250548175185716, 0.006598478841545256, 0.006901470216922334, 0.007594275057696822, 0.006132474562456533, 0.005005559271580384, 0.0038830736791746587, 0.004779644649588703, 0.003483514877142313, 0.003036115331629376, 0.003307212878019393, 0.0030671232209223518, 0.0015238162738262406, 0.008127610753536007, 0.0016655666248798445, 0.000973647723799441, 0.0003410867822227341, 0.0003410867822227341, 2.0376612963955543e-05] ,
    [[0.2045530598447029, 0.12824303928866548, 0.12736889550567168, 0.12790840612173818, 0.11310942504558523, 0.1030977470309843, 0.08556023738467107, 0.060247628543526215, 0.03745842695094551, 0.012453134283509415], [0.20335511017606023, 0.12606577344701583, 0.11787177499723175, 0.1211383014062673, 0.09921381906765585, 0.0774554312922157, 0.08587088915956151, 0.07773225556416787, 0.05769017827483114, 0.0336064666149928], [0.18176733780760626, 0.12374161073825503, 0.1222734899328859, 0.1422678970917226, 0.09689597315436242, 0.10381711409395973, 0.0818652125279642, 0.07487416107382551, 0.048308165548098435, 0.02418903803131991], [0.16993464052287582, 0.15376676986584106, 0.13596491228070176, 0.11033711730306157, 0.07413140694874441, 0.07473340213278294, 0.08393532851737186, 0.08857929136566907, 0.0823873409012728, 0.026229790161678708], [0.20126503525508088, 0.11022397345499793, 0.13770219825798424, 0.10493571132310245, 0.11965989216092908, 0.07942762339278307, 0.059207797594359186, 0.06584404811281626, 0.09612194110327665, 0.025611779344670262], [0.14741676234213547, 0.14936854190585533, 0.12525832376578644, 0.09632606199770379, 0.09988518943742825, 0.08518943742824339, 0.10206659012629161, 0.08691159586681975, 0.0807118254879449, 0.026865671641791045], [0.3358404462097524, 0.1375813403194456, 0.09760838333474182, 0.12067945575931717, 0.09169272373869687, 0.057550916927237385, 0.05721287923603482, 0.05383250232400913, 0.03828276852869095, 0.00971858362207386], [0.3902849053264909, 0.11520084940718457, 0.07467704831003362, 0.09918598478145461, 0.061405061051141394, 0.08352503981596178, 0.06228986020173421, 0.059281543089718634, 0.04016988143691382, 0.013979826579366484], [0.1605110586214301, 0.12153747047455443, 0.12314794932359888, 0.1321666308782478, 0.10446639467468327, 0.09168992913893065, 0.08095340347863432, 0.10682843031994846, 0.07440412282585356, 0.004294610264118532], [0.2355753379492252, 0.08704253214638971, 0.11745796241345202, 0.10690735245631389, 0.12776129244971976, 0.10715463237718431, 0.07649192218925156, 0.07228816353445433, 0.06758984503791625, 0.0017309594460929772], [0.13040303738317757, 0.07841705607476636, 0.09258177570093458, 0.1601927570093458, 0.1035338785046729, 0.1244158878504673, 0.10499415887850468, 0.12353971962616822, 0.08192172897196262, 0.0], [0.12739841986455983, 0.11286681715575621, 0.09353837471783295, 0.08211060948081264, 0.11075056433408578, 0.10948081264108352, 0.10158013544018059, 0.1872178329571106, 0.07505643340857787, 0.0], [0.14705462844102124, 0.1190985594066467, 0.1210954214805306, 0.11653116531165311, 0.1540436456996149, 0.11253744116388532, 0.09998573669947226, 0.09328198545143346, 0.036371416345742404, 0.0], [0.11343669250645995, 0.1264857881136951, 0.16912144702842377, 0.11589147286821705, 0.09483204134366925, 0.11873385012919897, 0.09560723514211886, 0.09754521963824289, 0.06834625322997416, 0.0], [0.1263640037720598, 0.11774215276842247, 0.13929678027751582, 0.128654182944901, 0.10427051057523912, 0.09928600296376128, 0.09820827158830661, 0.13700660110467466, 0.04917149400511922, 0.0], [0.18550185873605948, 0.1821561338289963, 0.09851301115241635, 0.09913258983890955, 0.07992565055762081, 0.12366790582403965, 0.11821561338289963, 0.08661710037174722, 0.026270136307311027, 0.0], [0.1476833976833977, 0.09817981246552675, 0.1594043022614451, 0.13251516822945394, 0.12038058466629895, 0.09666298952013237, 0.1167953667953668, 0.1167953667953668, 0.011583011583011582, 0.0], [0.12649772028416922, 0.10094369632064468, 0.16498780617113773, 0.09362739900328704, 0.10444279503764183, 0.09797476407591985, 0.19467712861838618, 0.10995652634927368, 0.006892164139539816, 0.0], [0.20363517631693515, 0.14486286460600784, 0.14018284719198956, 0.11362646930779277, 0.07270352633870265, 0.12168045276447541, 0.11471484545058773, 0.08663474096647801, 0.00195907705703091, 0.0], [0.25479307153246067, 0.17360835647229936, 0.12098373661245537, 0.11238926351976729, 0.10908369694565649, 0.08356472299352109, 0.09837366124553748, 0.0465423773634801, 0.0006611133148221605, 0.0], [0.22070038910505838, 0.10334630350194553, 0.11750972762645914, 0.0890272373540856, 0.1070817120622568, 0.1368093385214008, 0.12124513618677042, 0.10428015564202335, 0.0, 0.0], [0.12121612690019828, 0.1458030403172505, 0.13126239259748843, 0.16074025115664242, 0.1702577660277594, 0.11222736285525446, 0.09081295439524124, 0.06768010575016524, 0.0, 0.0], [0.1477124183006536, 0.14875816993464053, 0.12392156862745098, 0.1288888888888889, 0.10379084967320261, 0.10980392156862745, 0.12718954248366013, 0.10993464052287581, 0.0, 0.0], [0.2586016897269499, 0.11742377862128077, 0.12917840088159668, 0.1077507040528958, 0.08754744704297784, 0.12293375780580384, 0.11338312721929718, 0.063181094649198, 0.0, 0.0], [0.11345029239766082, 0.10380116959064327, 0.08888888888888889, 0.16681286549707602, 0.17865497076023393, 0.15, 0.1631578947368421, 0.03523391812865497, 0.0, 0.0], [0.1580619539316918, 0.14601535610272703, 0.1437648927720413, 0.11530315064866296, 0.10272703203600742, 0.1437648927720413, 0.16190097961344982, 0.028461742123378344, 0.0, 0.0], [0.31163396261508414, 0.1515856040174835, 0.09485724913977495, 0.08862642983353483, 0.12061750209243932, 0.09913512508137264, 0.08555751883195388, 0.047986608388356736, 0.0, 0.0], [0.20797865582518105, 0.1511879049676026, 0.16249523567526364, 0.10621267945623174, 0.0977004192605768, 0.14674120188032017, 0.09960614915512642, 0.028077753779697623, 0.0, 0.0], [0.15834554265750028, 0.12171757015665502, 0.1406514144032458, 0.1344528344415643, 0.1913670686351854, 0.12645103121830273, 0.1252113152259664, 0.0018032232615800743, 0.0, 0.0], [0.17363303447404976, 0.12690996337921454, 0.14623058466978153, 0.14534663467609546, 0.12943553478974618, 0.09571915645914889, 0.17931557014774593, 0.0034095214042177044, 0.0, 0.0], [0.16904254125042092, 0.1720731844202492, 0.18419575709956224, 0.10730721742058592, 0.08979683466157817, 0.14468514984846784, 0.1328993152991357, 0.0, 0.0, 0.0], [0.24663793103448275, 0.11077586206896552, 0.15629310344827585, 0.15172413793103448, 0.12413793103448276, 0.12508620689655173, 0.0853448275862069, 0.0, 0.0, 0.0], [0.20235155790711346, 0.16743092298647855, 0.13004115226337448, 0.11169900058788948, 0.14944150499706055, 0.1676660787771899, 0.0713697824808936, 0.0, 0.0, 0.0], [0.12453646477132262, 0.14941285537700866, 0.21029048207663784, 0.1665636588380717, 0.14323238566131025, 0.14894932014833126, 0.05701483312731768, 0.0, 0.0, 0.0], [0.287689508793208, 0.20121285627653124, 0.14942389326864766, 0.11837477258944815, 0.0960582171012735, 0.11740448756822316, 0.029836264402668285, 0.0, 0.0, 0.0], [0.13959845414270902, 0.16815911018946178, 0.12583655386935622, 0.1776793288717127, 0.1699500424168159, 0.14101234800640966, 0.07776416250353474, 0.0, 0.0, 0.0], [0.17580707895760406, 0.13237391417088032, 0.12679891092959938, 0.15208090237261765, 0.11914948787760923, 0.17840010372099052, 0.11538960197069882, 0.0, 0.0, 0.0], [0.1590846807344478, 0.16511372978898328, 0.19320361742943273, 0.1704576596327761, 0.16812825431625103, 0.11194847903535216, 0.03206357906275692, 0.0, 0.0, 0.0], [0.16099417823555753, 0.11632333184057322, 0.14655172413793102, 0.19267801164352888, 0.283922973578146, 0.0960591133004926, 0.003470667263770712, 0.0, 0.0, 0.0], [0.1968495830330485, 0.175332029239164, 0.17368475239369915, 0.23494286008442294, 0.11942757129620096, 0.09183568413466488, 0.007927519818799546, 0.0, 0.0, 0.0], [0.09420765027322404, 0.10025500910746812, 0.08794171220400729, 0.14586520947176684, 0.13377049180327868, 0.437959927140255, 0.0, 0.0, 0.0, 0.0], [0.21648256443578082, 0.15811132770197098, 0.14825644357808102, 0.10872861165258826, 0.13666883257526533, 0.23175222005631363, 0.0, 0.0, 0.0, 0.0], [0.146119222746734, 0.1289933033263805, 0.12789548797892195, 0.18498188604676694, 0.20452299923152925, 0.20748710066966736, 0.0, 0.0, 0.0, 0.0], [0.18762886597938144, 0.15189003436426116, 0.17172312223858616, 0.12528227785959745, 0.18487972508591066, 0.17859597447226314, 0.0, 0.0, 0.0, 0.0], [0.15427240603919049, 0.20599100546097013, 0.13957597173144876, 0.2838901381304208, 0.10512367491166077, 0.11114680372630903, 0.0, 0.0, 0.0, 0.0], [0.20151978683509325, 0.14082700088818711, 0.13411625382413894, 0.3146156123556696, 0.15158393368202902, 0.05733741241488207, 0.0, 0.0, 0.0, 0.0], [0.1738105443634805, 0.14487783969138449, 0.18066866695242179, 0.3117231033004715, 0.17702528932704673, 0.011894556365195028, 0.0, 0.0, 0.0, 0.0], [0.17513167795334839, 0.20090293453724606, 0.22065462753950338, 0.2786869826937547, 0.11013920240782543, 0.014484574868322046, 0.0, 0.0, 0.0, 0.0], [0.17636480411046884, 0.2138728323699422, 0.19087989723827875, 0.15080282594733463, 0.17867694283879254, 0.08940269749518305, 0.0, 0.0, 0.0, 0.0], [0.19883907620106211, 0.23811288131406694, 0.2095837964678276, 0.16351735210571816, 0.18982339137952328, 0.00012350253180190194, 0.0, 0.0, 0.0, 0.0], [0.21269987346140573, 0.26481076728402164, 0.17335787415161624, 0.1495456114114805, 0.1995858736914759, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23451635351426584, 0.34267819862809423, 0.1376876429068496, 0.1726811810319117, 0.11243662391887861, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21294363256784968, 0.26568508955059883, 0.257554114932425, 0.1632787605757609, 0.10053840237336556, 0.0, 0.0, 0.0, 0.0, 0.0], [0.15955845459106874, 0.3070747616658304, 0.15855494229804315, 0.29327646763672854, 0.08153537380832915, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1532368472521025, 0.20545667905339332, 0.19127713671034619, 0.2880891844318404, 0.1619401525523176, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2420743000326833, 0.23074408977012748, 0.2901187493190979, 0.1897810218978102, 0.04728183898028108, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2964967245798918, 0.21465869173075097, 0.3023829868033798, 0.16823317193582074, 0.01822842495015665, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27499749272891383, 0.15244208203790993, 0.22003811052050948, 0.34349613880252733, 0.009026175910139404, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27755406413124534, 0.21938851603281134, 0.2116331096196868, 0.2841163310961969, 0.007307979120059657, 0.0, 0.0, 0.0, 0.0, 0.0], [0.34695219310603925, 0.3214990138067061, 0.2211890673429135, 0.10763595378979994, 0.0027237719545411855, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2968036529680365, 0.30495759947814743, 0.2121113285496847, 0.18612741900413132, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.28831312017640576, 0.37183020948180817, 0.2534913634693128, 0.08636530687247336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2608695652173913, 0.233490307742854, 0.3497973934946884, 0.15584273354506625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.34587857847976305, 0.2653010858835143, 0.3202122408687068, 0.0686080947680158, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.32961159282575125, 0.3284238033020549, 0.29611592825751276, 0.045848675614681075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.29737457759292957, 0.38055627761892385, 0.26865089680270343, 0.0534182479854432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2888145218537674, 0.3424601967346111, 0.34124328161444073, 0.027481999797180814, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.40096092829299246, 0.24839089837730033, 0.3392258181488532, 0.011422355180853957, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2537556214716295, 0.2517462443785284, 0.48732178738876664, 0.007176346761075495, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3837751431996803, 0.28799786865592114, 0.32463034501132276, 0.003596643133075796, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3626972414576002, 0.37359518674083325, 0.26370757180156656, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.37574572127139366, 0.4541809290953545, 0.17007334963325182, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4373559372444048, 0.44516320916053237, 0.11748085359506283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5423922352269483, 0.3803406603863355, 0.07726710438671615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.521744948481199, 0.40840358624381107, 0.06985146527498996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5082343929528916, 0.42920975360653646, 0.06255585344057193, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3742084052964882, 0.61237766263673, 0.013413932066781807, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5792833146696529, 0.4006718924972004, 0.020044792833146696, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5320879400177914, 0.4582539077392299, 0.009658152242978777, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5335749065790963, 0.46348091948816666, 0.0029441739327369493, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5215558766859345, 0.4784441233140655, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4303488463658556, 0.5696511536341444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6345327604726101, 0.3654672395273899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7224646983311939, 0.2775353016688062, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6444237050863276, 0.35557629491367243, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9274776076278533, 0.07252239237214678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.904070796460177, 0.09592920353982301, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9621263974446727, 0.0378736025553274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9341983317886933, 0.06580166821130677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9888097660223805, 0.011190233977619531, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] )},
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

