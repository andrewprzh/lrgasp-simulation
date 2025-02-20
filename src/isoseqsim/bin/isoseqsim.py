#!/usr/bin/env python3

# NOTE: this file was modified by the LRGASP consortium comparing to the original IsoSeqSim source code
# Contributor: Andrey Prjibelski

import argparse
import sys
import os
import time
from subprocess import call


def main():
    args = do_inputs()
    tempdir = setup_tempdir(args.tempdir)
    udir = os.path.dirname(os.path.realpath(__file__)) + "/../utilities"
    sys.stdout.write("### Start analysis at " + time.strftime("%a,%d %b %Y %H:%M:%S") + "\n")
    sys.stdout.flush()
    if args.mode == "normal":  # normal mode
        sys.stdout.write("## Mode: normal\n")
        sys.stdout.write("# Step1: convert gtf to gpd\n")
        sys.stdout.flush()
        cmd1 = udir + "/py_isoseqsim_gtf2gpd.py -i " + args.annotation + " -o " + tempdir + "/normal_annotation.gpd"
        call(cmd1.split())
        sys.stdout.write("# Step2: generate transcriptome fasta file\n")
        sys.stdout.flush()
        cmd2 = udir + "/py_isoseqsim_gpd2fa_normal.py -a " + args.genome + " -g " + tempdir + "/normal_annotation.gpd" + " -o " + tempdir + "/normal_transcriptome.fa"
        call(cmd2.split())
        if args.expr is None:
            sys.stdout.write("# Step3: generate expression matrix based on Negative Binomial distribution\n")
            sys.stdout.flush()
            cmd3 = udir + "/py_isoseqsim_generate_expr_matrix.py -i " + tempdir + "/normal_annotation.gpd" + " -n " + args.nbn + " -p " + args.nbp + " -o " + args.transcript
            call(cmd3.split())
        else:
            sys.stdout.write("# Step3: generate expression matrix based on input abundance file\n")
            sys.stdout.flush()
            cmd3 = udir + "/py_isoseqsim_generate_expr_matrix_by_fixed_count.py -i " + tempdir + "/normal_annotation.gpd" + " -e " + args.expr + " -o " + args.transcript + " --read_number " + str(args.read_number)
            call(cmd3.split())

        sys.stdout.write("# Step4: simulate Iso-Seq reads\n")
        sys.stdout.flush()
        polya_opt = ' --polya' if args.polya else ''
        keep_isoforom_ids = ' --keep_isoform_ids' if args.keep_isoform_ids else ''
        cmd4 = udir + "/py_isoseqsim_simulate_reads_normal.py -g " + args.transcript + " -t " + tempdir + "/normal_transcriptome.fa" + " -5 " + args.c5 + " -3 " + args.c3 + " -o " + args.output + " -s " + args.es + " -i " + args.ei + " -d " + args.ed + " -p " + args.cpu + polya_opt + keep_isoforom_ids
        call(cmd4.split())

    elif args.mode == "fusion":  # fusion mode
        sys.stdout.write("## Mode: fusion\n")
        sys.stdout.write("# Step1: convert gtf to gpd\n")
        sys.stdout.flush()
        cmd1 = udir + "/py_isoseqsim_gtf2gpd.py -i " + args.annotation + " -o " + tempdir + "/normal_annotation.gpd"
        call(cmd1.split())
        sys.stdout.write("# Step2: generate fusion event\n")
        sys.stdout.flush()
        cmd2 = udir + "/py_isoseqsim_generate_fusion_event.py -i " + tempdir + "/normal_annotation.gpd" + " -o " + tempdir + "/fusion_annotation.gpd" + " -n " + args.fc
        call(cmd2.split())
        sys.stdout.write("# Step3: generate fusion transcript-specific transcriptome fasta file\n")
        sys.stdout.flush()
        cmd3 = udir + "/py_isoseqsim_gpd2fa_fusion.py -a " + args.genome + " -g " + tempdir + "/fusion_annotation.gpd" + " -o " + tempdir + "/fusion_transcriptome.fa"
        call(cmd3.split())
        sys.stdout.write("# Step4: generate expression matrix based on Negative Binomial distribution\n")
        sys.stdout.flush()
        cmd4 = udir + "/py_isoseqsim_generate_expr_matrix.py -i " + tempdir + "/fusion_annotation.gpd" + " -n " + args.nbn + " -p " + args.nbp + " -o " + args.transcript
        call(cmd4.split())
        sys.stdout.write("# Step5: simulate Iso-Seq reads\n")
        sys.stdout.flush()
        cmd5 = udir + "/py_isoseqsim_simulate_reads_fusion.py -g " + args.transcript + " -t " + tempdir + "/fusion_transcriptome.fa" + " -5 " + args.c5 + " -3 " + args.c3 + " -o " + args.output + " -s " + args.es + " -i " + args.ei + " -d " + args.ed + " -p " + args.cpu
        call(cmd5.split())

    elif args.mode == "ase":  # ASE mode
        sys.stdout.write("## Mode: ase\n")
        sys.stdout.write("# Step1: convert gtf to gpd\n")
        sys.stdout.flush()
        cmd1 = udir + "/py_isoseqsim_gtf2gpd.py -i " + args.annotation + " -o " + tempdir + "/normal_annotation.gpd"
        call(cmd1.split())
        sys.stdout.write("# Step2: generate haplotype-phased genome\n")
        sys.stdout.flush()
        if args.vcf and args.id:
            cmd2 = udir + "/py_isoseqsim_generate_haplotype_fa.py -g " + args.genome + " -v " + args.vcf + " -i " + args.id + " -o " + tempdir + "/ase_haplotype.fa"
        else:
            sys.stderr.write("Both --vcf and --id parameters are required for running 'ase' mode.")
            sys.exit()
        call(cmd2.split())
        sys.stdout.write("# Step3: generate ASE-specific transcriptome fasta file\n")
        sys.stdout.flush()
        cmd3 = udir + "/py_isoseqsim_gpd2fa_ase.py -a " + tempdir + "/ase_haplotype.fa" + " -g " + tempdir + "/normal_annotation.gpd" + " -o " + tempdir + "/ase_transcriptome.fa"
        call(cmd3.split())
        sys.stdout.write(
            "# Step4: generate expression matrix based on Negative Binomial distribution for total read count and Truncated Normal distribution for allele-specific read count\n")
        sys.stdout.flush()
        cmd4 = udir + "/py_isoseqsim_generate_expr_matrix_ase.py -i " + tempdir + "/normal_annotation.gpd" + " -n " + args.nbn + " -p " + args.nbp + " -l " + args.tnl + " -u " + args.tnu + " -m " + args.tnm + " -s " + args.tns + " -o " + args.transcript
        call(cmd4.split())
        sys.stdout.write("# Step5: simulate Iso-Seq reads\n")
        sys.stdout.flush()
        cmd5 = udir + "/py_isoseqsim_simulate_reads_ase.py -g " + args.transcript + " -t " + tempdir + "/ase_transcriptome.fa" + " -5 " + args.c5 + " -3 " + args.c3 + " -o " + args.output + " -s " + args.es + " -i " + args.ei + " -d " + args.ed + " -p " + args.cpu
        call(cmd5.split())

    elif args.mode == "apa":
        sys.stdout.write("## Mode: apa\n")  # APA mode
        sys.stdout.write("# Step1: convert gtf to gpd\n")
        sys.stdout.flush()
        cmd1 = udir + "/py_isoseqsim_gtf2gpd.py -i " + args.annotation + " -o " + tempdir + "/normal_annotation.gpd"
        call(cmd1.split())
        sys.stdout.write("# Step2: generate APA event\n")
        sys.stdout.flush()
        cmd2 = udir + "/py_isoseqsim_generate_apa_event.py -i " + tempdir + "/normal_annotation.gpd" + " -o " + tempdir + "/apa_annotation.gpd" + " -g " + args.genome + " -d " + args.dis_pa
        call(cmd2.split())
        sys.stdout.write("# Step3: generate APA-specific transcriptome fasta file\n")
        sys.stdout.flush()
        cmd3 = udir + "/py_isoseqsim_gpd2fa_apa.py -a " + args.genome + " -g " + tempdir + "/apa_annotation.gpd" + " -o " + tempdir + "/apa_transcriptome.fa"
        call(cmd3.split())
        sys.stdout.write("# Step4: generate expression matrix based on Negative Binomial distribution\n")
        sys.stdout.flush()
        cmd4 = udir + "/py_isoseqsim_generate_expr_matrix.py -i " + tempdir + "/apa_annotation.gpd" + " -n " + args.nbn + " -p " + args.nbp + " -o " + args.transcript
        call(cmd4.split())
        sys.stdout.write("# Step5: simulate Iso-Seq reads\n")
        sys.stdout.flush()
        cmd5 = udir + "/py_isoseqsim_simulate_reads_apa.py -g " + args.transcript + " -t " + tempdir + "/apa_transcriptome.fa" + " -5 " + args.c5 + " -3 " + args.c3 + " -o " + args.output + " -s " + args.es + " -i " + args.ei + " -d " + args.ed + " -p " + args.cpu
        call(cmd5.split())
    elif args.mode == "ats":
        sys.stdout.write("## Mode: ats\n")  # ATS mode
        sys.stdout.write("# Step1: convert gtf to gpd\n")
        sys.stdout.flush()
        cmd1 = udir + "/py_isoseqsim_gtf2gpd.py -i " + args.annotation + " -o " + tempdir + "/normal_annotation.gpd"
        call(cmd1.split())
        sys.stdout.write("# Step2: generate ATS event\n")
        sys.stdout.flush()
        cmd2 = udir + "/py_isoseqsim_generate_ats_event.py -i " + tempdir + "/normal_annotation.gpd" + " -o " + tempdir + "/ats_annotation.gpd" + " -g " + args.genome + " -d " + args.dis_tss
        call(cmd2.split())
        sys.stdout.write("# Step3: generate ATS-specific transcriptome fasta file\n")
        sys.stdout.flush()
        cmd3 = udir + "/py_isoseqsim_gpd2fa_ats.py -a " + args.genome + " -g " + tempdir + "/ats_annotation.gpd" + " -o " + tempdir + "/ats_transcriptome.fa"
        call(cmd3.split())
        sys.stdout.write("# Step4: generate expression matrix based on Negative Binomial distribution\n")
        sys.stdout.flush()
        cmd4 = udir + "/py_isoseqsim_generate_expr_matrix.py -i " + tempdir + "/ats_annotation.gpd" + " -n " + args.nbn + " -p " + args.nbp + " -o " + args.transcript
        call(cmd4.split())
        sys.stdout.write("# Step5: simulate Iso-Seq reads\n")
        sys.stdout.flush()
        cmd5 = udir + "/py_isoseqsim_simulate_reads_ats.py -g " + args.transcript + " -t " + tempdir + "/ats_transcriptome.fa" + " -5 " + args.c5 + " -3 " + args.c3 + " -o " + args.output + " -s " + args.es + " -i " + args.ei + " -d " + args.ed + " -p " + args.cpu
        call(cmd5.split())
    else:
        sys.stderr.write("Error: please choose the one mode from ['normal','fusion','ase','apa']")
        sys.exit()
    sys.stdout.write("### Finish analysis at " + time.strftime("%a,%d %b %Y %H:%M:%S") + "\n")
    sys.stdout.flush()


def setup_tempdir(tempd):
    if not os.path.exists(tempd):
        os.makedirs(tempd.rstrip('/'))
    return tempd.rstrip('/')


def do_inputs():
    parser = argparse.ArgumentParser(description="IsoSeqSim: Iso-Seq reads simulator",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='IsoSeqSim (version 0.1)')
    parser.add_argument('-m', '--mode', type=str, choices=["normal", "fusion", "apa", "ase", "ats"], default="normal",
                        help="Mode choice to perform")
    parser.add_argument('-g', '--genome', type=str, required=True, help="Input: genome sequence file, FASTA format")
    parser.add_argument('-a', '--annotation', type=str, required=True, help="Input: gene annotation file, GTF format")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output: simulated reads file, FASTA format. For more details, see README")
    parser.add_argument('-t', '--transcript', type=str, required=True,
                        help="Output: annotation file to show the simulated transcripts with read counts, modified GenePred table format. For more details, see README")
    parser.add_argument('--tempdir', type=str, required=True,
                        help="Temporary directory for saving intermediate files. You can delete it after finish running. (e.g., ./temp)")
    parser.add_argument('--nbn', type=str, default='10',
                        help="Average read count per transcript to simulate (i.e., the parameter 'n' of the Negative Binomial distribution)")
    parser.add_argument('--nbp', type=str, default='0.5',
                        help="The parameter 'p' of the Negative Binomial distribution")
    parser.add_argument('--cpu', type=str, default='1', help="Number of thread")

    group1 = parser.add_argument_group('Error pattern and transcript completeness options')
    group1.add_argument('--c5', type=str, required=True,
                        help="5'end completeness of transcripts. Tab-spit file: first column is the number of deleted nucleotides in comparison with reference gene annotation; second column is its frequency. Total frequency must be <= 1.0. Can choose from './utilities/5_end_completeness.*.tab' under the IsoSeqSim package")
    group1.add_argument('--c3', type=str, required=True,
                        help="3'end completeness of transcripts. Tab-spit file: first column is the number of deleted nucleotides in comparison with reference gene annotation; second column is its frequency. Total frequency must be <= 1.0. Can choose from './utilities/3_end_completeness.*.tab' under the IsoSeqSim package")
    group1.add_argument('--es', type=str, default='0.017',
                        help="Error rate: substitution. For more choices, see README")
    group1.add_argument('--ei', type=str, default='0.011', help="Error rate: insertion. For more choices, see README")
    group1.add_argument('--ed', type=str, default='0.022', help="Error rate: deletion. For more choices, see README")
    group1.add_argument('--expr', type=str, default=None, help="Expression file")
    parser.add_argument("--read_number", "-n", help="number of reads to generate (in millions)", default=1.0, type=float)
    group1.add_argument('--polya', default=False, action='store_true',
                        help="append polyA tails to transcripts before mutating")
    group1.add_argument('--keep_isoform_ids', default=False, action='store_true',
                        help="keep origin isoform ids in read names")

    group2 = parser.add_argument_group('Fusion mode options')
    group2.add_argument('--fc', type=str, default='all',
                        help="Number of fusion transcripts to generate. Note: total # of generated fusion transcripts = (# of gene)*(# of gene - 1)")

    group3 = parser.add_argument_group('APA mode options')
    group3.add_argument('--dis_pa', type=str, default='50', help="Distance (bp) bewteen two adjacent polyA sites")

    group4 = parser.add_argument_group('ATS mode options')
    group4.add_argument('--dis_tss', type=str, default='50', help="Distance (bp) bewteen two adjacent TSSs")

    group5 = parser.add_argument_group('ASE mode options')
    group5.add_argument('--vcf', type=str,
                        help="Input: phased genotype file, vcf format with header line to show the sample ID")
    group5.add_argument('--id', type=str,
                        help="Sample ID of your interest, can be found in the header line starts with '#CHROM' of vcf file")
    group5.add_argument('--tnl', type=str, default="0.0",
                        help="ASE score distribution: Prameter of the Truncated Normal distribution, lower boundary")
    group5.add_argument('--tnu', type=str, default="1.0",
                        help="ASE score distribution: Parameter of the Truncated Normal distribution, upper boundary")
    group5.add_argument('--tnm', type=str, default="0.5",
                        help="ASE score distribution: Parameter of the Truncated Normal distribution, mean value")
    group5.add_argument('--tns', type=str, default="0.1",
                        help="ASE score distribution: Parameter of the Truncated Normal distribution, stdandard variation")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
