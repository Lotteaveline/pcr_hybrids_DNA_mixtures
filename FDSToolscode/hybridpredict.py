#!/usr/bin/env python3

#
# Copyright (C) 2021 Jerry Hoogenboom
#
# This file is part of FDSTools, data analysis tools for Massively
# Parallel Sequencing of forensic DNA markers.
#
# FDSTools is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# FDSTools is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FDSTools.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Predict background profiles of new alleles based on a model of stutter
occurrence obtained from stuttermodel.

This tool can be used to compute background noise profiles for alleles
for which no reference samples are available.  The profiles are
predicted using a model of stutter occurrence that must have been
created previously using stuttermodel.  A list of sequences should be
given; bgpredict will predict a background noise profile for each of the
provided sequences separately.  The prediction is based completely on
the provided stutter model.

The predicted background noise profiles obtained from bgpredict can be
combined with the output of bgestimate and/or bghomstats using bgmerge.

It is possible to use an entire forensic case sample as the SEQS input
argument of bgpredict to obtain a predicted background noise profile for
each sequence detected in the sample.  When the background noise
profiles thus obtained are combined with those obtained from bgestimate,
bgcorrect may subsequently produce 'cleaner' results if the sample
contained alleles for which no reference samples were available.
"""
import argparse
import sys
#import numpy as np  # Only imported when actually running this tool.

from errno import EPIPE
import pickle
import pandas as pd

from ..lib.hybrids import obtain_all_hybrids_from_file
from ..lib.cli import add_sequence_format_args
from ..lib.io import get_column_ids
from ..lib.seq import PAT_SEQ_RAW, SEQ_SPECIAL_VALUES, ensure_sequence_format, reverse_complement,\
                      mutate_sequence, get_repeat_pattern

__version__ = "1.1.0"


# Default values for parameters are specified below.


# Default minimum R2 score.
# This value can be overridden by the -t command line option.
_DEF_MIN_R2 = 0.



def predict_hybrids(hybrid_model, seqsfile, outfile, library):
    
    model = pickle.load(open(hybrid_model, 'rb'))

    outfile.write("\t".join(("marker", "allele", "sequence", "tmean", "tools")) + "\n")
    columns = ['highest_reads_par', 'lowest_reads_par', \
            'ratio_parents', 'len_hybrid', 'len_longest_par', \
            'len_shortest_par', 'begin_pos_overlap', 'end_pos_overlap', \
            'len_overlap', 'longest_c_stretch', 'len_overlap_parA', \
            'len_overlap_parB', 'hybrid_CG_counts', 'hybrid_AT_counts', \
            'overlap_CG_counts', 'overlap_AT_counts', 'hybrid_A_counts', \
            'hybrid_C_counts', 'hybrid_G_counts', 'hybrid_T_counts', \
            'overlap_A_counts', 'overlap_C_counts', 'overlap_G_counts', \
            'overlap_T_counts', 'MT_hybrid', 'MT_overlap', 'hybrid_mol_weight']

    hybrid_features_dict,_ = obtain_all_hybrids_from_file(seqsfile, False)
    for marker, align_hybrid_parents in hybrid_features_dict.items():
        for hybrid_features_and_seqs in align_hybrid_parents:

            hybrid_features = hybrid_features_and_seqs[0]
            parent_highest_reads_seq = hybrid_features_and_seqs[2]
            parent_lowest_reads_seq = hybrid_features_and_seqs[3]
            hybridseq = hybrid_features_and_seqs[4]

            highest_reads_par = hybrid_features[1]
            lowest_reads_par = hybrid_features[2]
            
            hybrid_features.pop(0)

            features = pd.DataFrame([hybrid_features], columns=columns)
            y_pred = model.predict(features)

            reads_per_par = y_pred[0]*(highest_reads_par+lowest_reads_par)/2*100
        
            pred_perc_highest_par = str(reads_per_par/highest_reads_par)
            pred_perc_lowest_par = str(reads_per_par/lowest_reads_par)


            outfile.write("\t".join((marker, parent_highest_reads_seq, \
                    hybridseq, pred_perc_highest_par, "hybridmodel")) + "\n")
            outfile.write("\t".join((marker, parent_lowest_reads_seq, \
                    hybridseq, pred_perc_lowest_par, "hybridmodel")) + "\n")

    outfile.close()
#predict_hybrids
     

def add_arguments(parser):
    parser.add_argument("hybridmodel", metavar="HYBR",
        type=argparse.FileType("tr", encoding="UTF-8"),
        help="file containing a trained hybrid model")
    parser.add_argument("seqs", metavar="SEQS", type=argparse.FileType("tr", encoding="UTF-8"),
        help="file containing the sequences for which a profile should be predicted")
    parser.add_argument("outfile", metavar="OUT", nargs="?", default=sys.stdout,
        type=argparse.FileType("tw", encoding="UTF-8"),
        help="the file to write the  to (default: write to stdout)")
    add_sequence_format_args(parser, default_format="raw", force=True)
#add_arguments


def run(args):
    # Import numpy now.
    global np
    import numpy as np

    try:
        predict_hybrids(args.stuttermodel, args.seqs, args.outfile, args.library)
    except IOError as e:
        if e.errno == EPIPE:
            return
        raise
#run