#!/usr/bin/env python3
import sys
import time
import argparse
import numpy as np
from multiprocessing import cpu_count, Pool


def main(args):
	sys.stdout.write("Start analysis: " + time.strftime("%a,%d %b %Y %H:%M:%S") + "\n")
	sys.stdout.flush()
	error_type, error_prob = extract_error_rate(args.er_sub, args.er_ins, args.er_del)
	bp5_list, pro5_list, bp3_list, pro3_list = extract_read_completeness(args.cpt_5end, args.cpt_3end)
	input_gpd_fl = args.input_gpd
	output_gpd_fl = args.output
	dic_iso_seq, iso_list = parse_transcriptome_fa(args.input_fa)
	p = Pool(processes=args.cpu)
	csize = 100
	results = p.imap(func=generate_simulated_reads,
					 iterable=generate_tx(input_gpd_fl, dic_iso_seq, iso_list, error_type, error_prob, bp5_list,
										  pro5_list, bp3_list, pro3_list), chunksize=csize)

	for res in results:
		if not res:
			continue
		output_gpd_fl.write(res + "\n")
	output_gpd_fl.close()

	sys.stdout.write("Finish analysis: " + time.strftime("%a,%d %b %Y %H:%M:%S") + "\n")
	sys.stdout.flush()


def extract_error_rate(err_sub, err_ins, err_del):
	error_type = ["no", "sub", "ins", "del"]
	if err_sub + err_ins + err_del > 1.0:
		sys.stdout.write("The overall error rate (substitution+insertion+deletion) is > 1.0 !!! END PROGRAM.")
		sys.exit()
	else:
		error_prob = [(1.0 - (err_sub + err_ins + err_del)), err_sub, err_ins, err_del]
	return error_type, error_prob


def mutate_read(read_seq, error_type, error_prob):
	read_seq = read_seq.upper()

	dic_error_no = {"A": "A", "C": "C", "G": "G", "T": "T"}
	dic_error_sub = {"A": np.random.choice(["C", "G", "T"]), "C": np.random.choice(["A", "G", "T"]),
					 "G": np.random.choice(["A", "C", "T"]), "T": np.random.choice(["A", "C", "G"])}
	dic_error_ins = {"A": "A" + np.random.choice(["A", "C", "G", "T"]),
					 "C": "C" + np.random.choice(["A", "C", "G", "T"]),
					 "G": "G" + np.random.choice(["A", "C", "G", "T"]),
					 "T": "T" + np.random.choice(["A", "C", "G", "T"])}
	dic_error_del = {"A": "", "C": "", "G": "", "T": ""}

	dic_error = {"no": dic_error_no, "sub": dic_error_sub, "ins": dic_error_ins, "del": dic_error_del}
	read_seq_new = ""
	for nuc in read_seq:
		nuc_new = dic_error[np.random.choice(error_type, p=error_prob)][nuc]
		read_seq_new += nuc_new
	return read_seq_new


def extract_read_completeness(pro_fl_5end, pro_fl_3end):
	bp5_list = []
	pro5_list = []
	for line in pro_fl_5end:
		bp5, pro5 = line.strip().split("\t")
		if int(bp5) != 0:
			bp5_list.append(int(bp5))
			pro5_list.append(float(pro5))
	bp3_list = []
	pro3_list = []
	for line in pro_fl_3end:
		bp3, pro3 = line.strip().split("\t")
		if int(bp3) != 0:
			bp3_list.append(int(bp3))
			pro3_list.append(float(pro3))

	if sum(pro5_list) > 1.0 or sum(pro3_list) > 1.0:
		if sum(pro5_list) > 1.0:
			sys.stdout.write(
				"The overall probability of incompleteness of 5 end is >1.0 !!! Please check the 5 end incompleteness file !")
		if sum(pro3_list) > 1.0:
			sys.stdout.write(
				"The overall probability of incompleteness of 3 end is >1.0 !!! Please check the 3 end incompleteness file !")
		sys.exit()
	else:
		bp5_list.append(0)
		pro5_list.append(float(1.0) - sum(pro5_list))
		bp3_list.append(0)
		pro3_list.append(float(1.0) - sum(pro3_list))
	pro_fl_5end.close()
	pro_fl_3end.close()
	return bp5_list, pro5_list, bp3_list, pro3_list


def mutate_read_ends(read_seq, bp5_list, pro5_list, bp3_list, pro3_list):
	del5 = np.random.choice(bp5_list, p=pro5_list)
	del3 = np.random.choice(bp3_list, p=pro3_list)
	if del5 == 0:
		if del3 == 0:
			read_seq_new = read_seq
		else:
			read_seq_new = read_seq[:-del3]
	else:
		if del3 == 0:
			read_seq_new = read_seq[del5:]
		else:
			read_seq_new = read_seq[del5:-del3]
	return read_seq_new


def parse_transcriptome_fa(iso_fa_fl):
	dic_iso_seq = {}
	iso = ""
	seq_list = [""]
	iso_list = []
	for line in iso_fa_fl:
		if line.startswith(">"):
			dic_iso_seq[iso] = "".join(seq_list)
			seq_list = []
			iso = line.strip()[1:]  # get isoform ID + allele ID
			iso_list.append(iso)
		else:
			seq_list.append(line.strip())
	dic_iso_seq[iso] = "".join(seq_list)
	del dic_iso_seq[""]
	iso_fa_fl.close()
	return dic_iso_seq, iso_list


def generate_simulated_reads(inputs):
	(line, z, dic_iso_seq, iso_list, error_type, error_prob, bp5_list, pro5_list, bp3_list, pro3_list) = inputs
	gene, iso, chrom, strand, tss, tts, cds_tss, cds_tts, exon_count, exon_start_set, exon_end_set, total_read_count, allele1_read_count, allele2_read_count = line.rstrip(
		"\n").split("\t")[:14]
	if int(total_read_count) != 0:
		if allele1_read_count != 0:
			lr_idx1 = 0
			iso1 = iso + " Allele1"
			read_seq1 = dic_iso_seq[iso1]
			simu_fa_all_lines_list1 = []
			for i in range(0, int(allele1_read_count)):
				simu_fa_seq_line_list1 = []
				read_seq_muta1 = mutate_read(read_seq1, error_type, error_prob)
				read_seq_muta_end1 = mutate_read_ends(read_seq_muta1, bp5_list, pro5_list, bp3_list, pro3_list)
				if read_seq_muta_end1 != "":
					lr_idx1 += 1
					simu_fa_name_line1 = ">LR" + str(iso_list.index(iso1) + 1) + "." + str(lr_idx1) + " " + iso1 + "\n"
					for j in range(0, len(read_seq_muta_end1), 80):
						simu_fa_seq_line_list1.append(read_seq_muta_end1[j:j + 80])
					simu_fa_line1 = simu_fa_name_line1 + "\n".join(simu_fa_seq_line_list1)
					simu_fa_all_lines_list1.append(simu_fa_line1)
			if simu_fa_all_lines_list1 != []:
				simu_fa_all_lines1 = "\n".join(simu_fa_all_lines_list1)
			else:
				simu_fa_all_lines1 = ""
		else:
			simu_fa_all_lines1 = ""
		if allele2_read_count != 0:
			lr_idx2 = 0
			iso2 = iso + " Allele2"
			read_seq2 = dic_iso_seq[iso2]
			simu_fa_all_lines_list2 = []
			for i in range(0, int(allele2_read_count)):
				simu_fa_seq_line_list2 = []
				read_seq_muta2 = mutate_read(read_seq2, error_type, error_prob)
				read_seq_muta_end2 = mutate_read_ends(read_seq_muta2, bp5_list, pro5_list, bp3_list, pro3_list)
				if read_seq_muta_end2 != "":
					lr_idx2 += 1
					simu_fa_name_line2 = ">LR" + str(iso_list.index(iso2) + 1) + "." + str(lr_idx2) + " " + iso2 + "\n"
					for j in range(0, len(read_seq_muta_end2), 80):
						simu_fa_seq_line_list2.append(read_seq_muta_end2[j:j + 80])
					simu_fa_line2 = simu_fa_name_line2 + "\n".join(simu_fa_seq_line_list2)
					simu_fa_all_lines_list2.append(simu_fa_line2)
			if simu_fa_all_lines_list2 != []:
				simu_fa_all_lines2 = "\n".join(simu_fa_all_lines_list2)
			else:
				simu_fa_all_lines2 = ""
		else:
			simu_fa_all_lines2 = ""
		if simu_fa_all_lines1 == "" and simu_fa_all_lines2 == "":
			return None
		elif simu_fa_all_lines1 == "" and simu_fa_all_lines2 != "":
			return simu_fa_all_lines2
		elif simu_fa_all_lines1 != "" and simu_fa_all_lines2 == "":
			return simu_fa_all_lines1
		else:
			simu_fa_all_lines = simu_fa_all_lines1 + "\n" + simu_fa_all_lines2
			return simu_fa_all_lines
	else:
		return None


def generate_tx(input_fl, dic_iso_seq, iso_list, error_type, error_prob, bp5_list, pro5_list, bp3_list, pro3_list):
	z = 0
	for line in input_fl:
		z += 1
		yield line, z, dic_iso_seq, iso_list, error_type, error_prob, bp5_list, pro5_list, bp3_list, pro3_list


def do_inputs():
	parser = argparse.ArgumentParser(description="Simulate ASE-specific reads.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-g', '--input_gpd', type=argparse.FileType('r'), required=True,
						help="Input: gpd file with total read count, allele1 read count, allele2 read count in last 3 columns")
	parser.add_argument('-t', '--input_fa', type=argparse.FileType('r'), required=True,
						help="Input: ASE-specific transcriptome fasta file")
	parser.add_argument('-5', '--cpt_5end', type=argparse.FileType('r'), required=True,
						help="Input: 5'end completeness probability file")
	parser.add_argument('-3', '--cpt_3end', type=argparse.FileType('r'), required=True,
						help="Input: 3'end completeness probability file")
	parser.add_argument('-o', '--output', type=argparse.FileType('w'), required=True,
						help="Output: fasta file, simulated reads")
	parser.add_argument('-s', '--er_sub', type=float, default=0.05, help="Error rate: substitution")
	parser.add_argument('-i', '--er_ins', type=float, default=0.025, help="Error rate: insertion")
	parser.add_argument('-d', '--er_del', type=float, default=0.025, help="Error rate: deletion")
	parser.add_argument('-p', '--cpu', type=int, default=cpu_count(), help="Number of threads")
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = do_inputs()
	main(args)
