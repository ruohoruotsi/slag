""" Functions for building different types of melody to chord
finite-state models """

import copy
import glob
import json
import librosa
import marl
import music21
import numpy as np
import os
import sys

import pyjams

import hamilton.train.data as traindata
import hamilton.train.utils as trainutils
from hamilton.core import fsm
from hamilton.core import fileutils
from hamilton.core import utils as coreutils

DEBUG = False

FACTOR_FST_EPS_WEIGHT = 1000.0

DISAMBIG_SYMBOL = '#'

NGRAM_SMOOTH_METHOD_KN = 'kneser_ney'
NGRAM_SMOOTH_METHOD_WB = 'witten_bell'
# NGRAM_SMOOTH_METHOD_KATZ = 'katz'
NGRAM_SMOOTH_METHOD_ABS = 'absolute'
NGRAM_SMOOTH_METHOD_NONE = 'unsmoothed'

# NGRAM_SMOOTH_METHOD = NGRAM_SMOOTH_METHOD_KN
NGRAM_SMOOTH_METHOD = NGRAM_SMOOTH_METHOD_WB
# NGRAM_SMOOTH_METHOD = NGRAM_SMOOTH_METHOD_NONE

# n-gram parameters
NGRAM_REL_ENT_THRESH = 1e-10
NGRAM_WB_K = 1.0 # Witten-Bell k parameter
NGRAM_NUM_BINS = 10 # for absolute discounting, Kneser-Ney?
NGRAM_DISCOUNT_D = 10.0  # for absolute discounting, Kneser-Ney?

BUILD_2ND_ORDER_ERGODIC = False

WEIGHT_CURVE_EXP = -3.0
# WEIGHT_CURVE_EXP = None

# TODO: clean all this stuff up!
PY_SITE_PACKS_DIR = \
    '/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages'
MOZART_CORPUS_DIR = \
    os.path.join(PY_SITE_PACKS_DIR,\
        'music21-1.9.3-py2.7.egg','music21','corpus','mozart')


def build_ngram(ngram_filename,\
    order,\
    symbol_table,\
    chord_seqs_filename,\
    dataset,\
    build_ergodic_lm,\
    smooth_method=NGRAM_SMOOTH_METHOD,\
    rel_ent_thresh=NGRAM_REL_ENT_THRESH,\
    witten_bell_k=NGRAM_WB_K,\
    kn_num_bins=NGRAM_NUM_BINS,\
    kn_discount_D=NGRAM_DISCOUNT_D):

    """ function to build an OpenGRM n-gram

    Parameters
    ----------
    ngram_filename: string
        name of compiled ngram (in .fst format)
    order: int
        n-gram order
    symbol_table: SymbolTable object
        symbols table
    chord_seqs_filename: string
        name of file containing chord sequences
    dataset: data.Dataset object
        object representing training data for fst
    build_ergodic_lm: boolean
        build ergodic or sequence FST        
    smooth_method: string
        smoothing method for smoothing method
        (see ngrammake documentation)
    rel_ent_thresh: float
        relative entropy threshold parameter
        for n-gram pruning
    witten_bell_k: float
        k parameter for Witten-Bell smoothing
    kn_num_bins: int
        number of bins for Absolute and Kneser-Ney smoothing
    kn_discount_D: float
        discount parameter for Absolute and Kneser-Ney smoothing

    Returns
    -------
    fst: FST object
        the n-gram, as an FST
    """

    print 'building chord sequence'
    if chord_seqs_filename is None:
        chord_seqs_filename = marl.fileutils.temp_file('txt')
    if build_ergodic_lm:
        # print 'NOT USING REPEATING CHORDS'
        chord_seqs = dataset.get_harmonic_sequences(repeat_type=True)
    else:
        chord_seqs = dataset.get_harmonic_sequences(repeat_type=True)
    chord_seqs = [' '.join(s) for s in chord_seqs]
    fileutils.write_text_file(data=chord_seqs, fname=chord_seqs_filename)

    # NGRAM_SMOOTH_METHOD='kneser_ney'
    print 'smoothing method:',smooth_method
    if smooth_method == NGRAM_SMOOTH_METHOD_WB:
        print 'witten_bell_k=',witten_bell_k
    elif smooth_method == NGRAM_SMOOTH_METHOD_KN:
        print 'discount_D=',kn_discount_D
        print 'bins=',kn_num_bins

    print 'n-gram order:',order

    cmd = [os.path.join(fsm.FST_BIN_DIR,'farcompilestrings')]
    # cmd.append('--generate_keys=8')
    cmd.append('--symbols='+symbol_table.filename)
    cmd.append('--keep_symbols=1')
    cmd.append(chord_seqs_filename)
    cmd.append('|')
    cmd.append(os.path.join(fsm.FST_BIN_DIR,'ngramcount'))
    cmd.append('--order='+str(order))
    cmd.append('|')
    cmd.append(os.path.join(fsm.FST_BIN_DIR,'ngrammake'))
    if smooth_method is not None:
        cmd.append('--method=' + smooth_method)
        # if smooth_method == NGRAM_SMOOTH_METHOD_KATZ:
        #     cmd.append('--backoff=true')
        if smooth_method == NGRAM_SMOOTH_METHOD_WB:
            if witten_bell_k is None:
                witten_bell_k = 1.0
            cmd.append('--witten_bell_k='+str(witten_bell_k))
        elif smooth_method == NGRAM_SMOOTH_METHOD_KN:
            if kn_discount_D is None:
                kn_discount_D = 1.0
            if kn_num_bins is None:
                kn_num_bins = 5
            cmd.append('--discount_D='+str(kn_discount_D))
            cmd.append('--bins='+str(kn_num_bins))

    # cmd.append('--method=presmoothed')
    cmd.append('|')
    # cmd.append(os.path.join(FST_BIN_DIR,'ngramshrink'))
    # cmd.append('-method=relative_entropy')
    # cmd.append('-theta='+str(rel_ent_thresh))
    # cmd.append('|')
    cmd.append(os.path.join(fsm.FST_BIN_DIR,'fstarcsort'))

    dbg_msg = ' '.join(cmd)
    print '[build_ngram]',dbg_msg

    stdout,stderr = coreutils.launch_process(command=cmd,\
        outfile_name=ngram_filename,outfile_type='binary')

    if not stderr == '':
        print 'ERROR: unable to build n-gram',ngram_filename
        print 'command:',' '.join(cmd)
        print 'error:',stderr
        return None

    fst = fsm.FST(filename=ngram_filename,\
        isyms_table=symbol_table,\
        osyms_table=symbol_table)
    fst.map_to_log()
    fst.replace_infinite_weights()
    
    return fst

def generate_ngram_symbols_file(chord_seqs_filename,symbols_filename):
    """ Generate the n-gram/FST output symbols file.

    Parameters
    ----------
    chord_seqs_filename: string
        name of file with chord sequence data
    symbols_filename: string
        name of file containing chord symbols

    Returns
    -------
    None

    """

    cmd = [os.path.join(fsm.FST_BIN_DIR,'ngramsymbols')]
    cmd.append(chord_seqs_filename)

    stdout,stderr = coreutils.launch_process(command=cmd,\
        outfile_name=symbols_filename,\
        outfile_type='text')

    if not stderr == '':
        print 'ERROR: unable to build n-gram symbols file',symbols_filename
        print 'command:',' '.join(cmd)
        print 'error:',stderr
        return

def build_LoG(lang_mod_filename,\
    ngram_filename,\
    LoG_filename,
    ngram_order,\
    isyms_filename,\
    osyms_filename,\
    dataset,\
    chord_seqs_filename=None,\
    unseen_transition_weight=None,\
    omit_ngram=False,\
    witten_bell_k=None,\
    kn_num_bins=None,\
    kn_discount_D=None,\
    compose_fsts=True):
    """ Build the LoG (L composed with G) FST
    from the dataset

    Parameters
    ----------
    lang_mod_filename: string
        name of compiled language model FST file
    ngram_filename: string
        name of compiled language model FST file
    ngram_order: int
        order of n-gram
    isyms_filename: string
        name of input symbols file
    osyms_filename: string
        name of output symbols file
    dataset: data.Dataset object
        object representing training data for fst
    chord_seqs_filename: string
        name of file containing chord sequences for
        training n-gram
    roman_to_label_fst: FST object
        Roman numeral to chord label mapping FST (optional)
    unseen_transition_weight: float
        weight for unseen transitions in ergodic melody-to-chord FST
        if None, build regular FST instead of ergodic FST
    omit_ngram: boolean
        if True, then return FST only (i.e., don't compose with ngram)
    witten_bell_k: float
        k parameter for Witten Bell smoothing
    kn_num_bins: int
        number of bins for Absolute and Kneser-Ney smoothing
    kn_discount_D: float
        discount parameter for Absolute and Kneser-Ney smoothing
    compose_fsts: boolean
        compose L and G into LoG, or use separately (i.e., compose melody
        with min(det(L)), compose that result with G)

    Returns
    -------
    fst: FST object
        FST resulting from composition of L and G FST's
    """

    if omit_ngram:
        print '\t\tOMITTING N-GRAM'

    # build the language model
    print 'building language model'
    build_ergodic_lm = (unseen_transition_weight is not None)
    if build_ergodic_lm:
        print 20*'.'
        print 'building ergodic FST (L):'
        if BUILD_2ND_ORDER_ERGODIC:
            print 'building 2nd ORDER ergodic FST:',lang_mod_filename
            L_fst = build_2nd_order_ergodic_fst(fst_filename=lang_mod_filename,\
                isyms_filename=isyms_filename,\
                osyms_filename=osyms_filename,\
                dataset=dataset,\
                unseen_transition_weight=unseen_transition_weight)
        else:
            L_fst = build_ergodic_fst(fst_filename=lang_mod_filename,\
                isyms_filename=isyms_filename,\
                osyms_filename=osyms_filename,\
                dataset=dataset,\
                unseen_transition_weight=unseen_transition_weight)
        L_fst.compile()
    else:
        print 20*'.'        
        print 'building sequence FST (L):'
        L_fst = build_L_fst(fst_filename=lang_mod_filename,\
            isyms_filename=isyms_filename,\
            osyms_filename=osyms_filename,\
            dataset=dataset)

    # build the n-gram
    print 20*'.'
    print 'building n-gram (G):'
    ngram = build_ngram(ngram_filename=ngram_filename,\
        order=ngram_order,\
        symbol_table=L_fst.osyms_table,\
        chord_seqs_filename=chord_seqs_filename,\
        dataset=dataset,\
        build_ergodic_lm=build_ergodic_lm,\
        witten_bell_k=witten_bell_k,\
        kn_num_bins=kn_num_bins,\
        kn_discount_D=kn_discount_D)
    # ngram.compile()

    # build the final L composed with G FST
    if omit_ngram:
        print 'omitting n-gram'
        LoG_fst = L_fst
    elif compose_fsts:
        print 20*'.'
        print 'composing L and G'
        LoG_fst = fsm.compose(L_fst,ngram,LoG_filename)
    else:
        print 20*'.'
        print 'using separate L and G FSTs'
        print 'building L...'
        L_fst.build_m2c(do_closure=True,do_determinize_twice=True)        
        print 'removing disambiguation symbol from L...'
        L_fst.substitute_disambig_symbol(disambig_sym=traindata.DISAMBIG_SYMBOL)
        L_fst.load_from_compiled()

    # determinize,minimize,remove disambiguation symbols
    if build_ergodic_lm:
        # print 'doing closure...'
        # LoG_fst.closure()
        pass
    elif compose_fsts:
        print 20*'.'        
        print 'building L o G FST'        
        LoG_fst.build_m2c(do_closure=False,do_determinize_twice=True)
        print 'removing disambiguation symbol from L o G...'
        LoG_fst.substitute_disambig_symbol(disambig_sym=traindata.DISAMBIG_SYMBOL)
        LoG_fst.load_from_compiled()

    if compose_fsts:
        return LoG_fst,None
    else:
        return L_fst,ngram


def build_L_fst(fst_filename,\
    isyms_filename,\
    osyms_filename,\
    dataset):
    """ Build a language model FST
    (sometimes referred to as L).

    Parameters
    ----------
    fst_filename: string
        name of compiled FST file
    isyms_filename: string
        name of input symbols file
    osyms_filename: string
        name of output symbols file
    dataset: data.Dataset object
        object representing training data for fst

    Returns
    -------
    fst: FST object
        FST with the melodic and harmonic sequences

    """

    input_seqs, output_seqs = dataset.get_disambiguated_seqs()

    output_seqs = traindata.expand_harmonic_seqs(melodic_seqs=input_seqs,\
        harmonic_seqs=output_seqs)

    # make symbols tables and FST
    isyms_table = fsm.SymbolTable(filename=isyms_filename)
    isyms_table.set_symbols_from_seq(sym_seqs=input_seqs)
    isyms_table.write_file()

    osyms_table = fsm.SymbolTable(filename=osyms_filename)
    chord_seqs = dataset.get_harmonic_sequences()
    osyms_table.set_symbols_from_seq(sym_seqs=chord_seqs)
    osyms_table.write_file()

    if WEIGHT_CURVE_EXP is None:
        print 'using weight 1.0 for all sequences'
        weights = len(input_seqs) * [1.0]
    else:
        print 'using weight curve exponent:',WEIGHT_CURVE_EXP

        weights = []
        # compute weights to favor longer sequences
        max_w = float(dataset.melodic_context_len)
        for iseq in input_seqs:
            N = float(len(iseq))
            # w = weight_curve[N-1]
            w = 1.0 + (N/max_w)**(WEIGHT_CURVE_EXP)
            weights.append(w)

    fst = fsm.BranchFST(filename=fst_filename,\
        isyms_table=isyms_table,osyms_table=osyms_table)

    fst.build(input_seqs=input_seqs,output_seqs=output_seqs,weights=weights,add_epsilons=False)
    print 'compiling L...'
    fst.compile()
    print 'performing closure on L...'
    fst.closure()
    fst.arcsort(sort_type='olabel')

    return fst


def build_ergodic_fst(fst_filename,\
    isyms_filename,\
    osyms_filename,\
    dataset,\
    unseen_transition_weight):
    """ Build an ergodic language model FST using a flower FST.

    Parameters
    ----------
    fst_filename: string
        name of compiled FST file
    isyms_filename: string
        name of input symbols file
    osyms_filename: string
        name of output symbols file
    dataset: data.Dataset object
        object representing training data for fst
    unseen_transition_weight: float
        weight for unseen transitions 

    Returns
    -------
    fst: FST object
        FST with the melodic and harmonic sequences

    """

    # input_seqs, output_seqs = dataset.get_disambiguated_seqs()
    input_seqs, output_seqs = dataset.get_melody_harmony_sequence_pairs()

    sym_seqs = [item for sublist in input_seqs for item in sublist]

    # make symbols tables and FST
    isyms_table = fsm.SymbolTable(filename=isyms_filename)
    isyms_table.set_symbols_from_seq(sym_seqs=sym_seqs)
    isyms_table.write_file()

    osyms_table = fsm.SymbolTable(filename=osyms_filename)
    chord_seqs = dataset.get_harmonic_sequences()
    osyms_table.set_symbols_from_seq(sym_seqs=chord_seqs)
    osyms_table.write_file()

    # compute the weights for the FST

    weights_map = {}

    # create weights for existing transitions
    for iseq,osym in zip(input_seqs,output_seqs):
        for isym in iseq:

            if not (isym,osym) in weights_map:
                weights_map[isym,osym] = 1.0
            else:
                w = weights_map[isym,osym]
                weights_map[isym,osym] = coreutils.log_plus(w,1.0)

    # assemble the sequences, and set any unseen transitions
    fst_isyms = []
    fst_osyms = []
    fst_wts = []

    isyms = ['pc'+str(n) for n in range(12)]

    osyms = osyms_table.get_symbols()
        
    for isym in isyms:
        if isym == fsm.EPSILON_LABEL:
            continue

        for osym in osyms:
            if osym == fsm.EPSILON_LABEL:
                continue

            fst_isyms.append(isym)
            fst_osyms.append(osym)
            if (isym,osym) in weights_map:
                fst_wts.append(weights_map[isym,osym])
            else:
                fst_wts.append(unseen_transition_weight)

    fst = fsm.FlowerFST(filename=fst_filename,\
        isyms_table=isyms_table,\
        osyms_table=osyms_table)

    fst.build(input_symbols=fst_isyms,\
        output_symbols=fst_osyms,\
        weights=fst_wts)

    return fst

def build_2nd_order_ergodic_fst(fst_filename,\
    isyms_filename,\
    osyms_filename,\
    dataset,\
    unseen_transition_weight):
    """ Build a 2nd order ergodic language model FST using a flower FST.

    Parameters
    ----------
    fst_filename: string
        name of compiled FST file
    isyms_filename: string
        name of input symbols file
    osyms_filename: string
        name of output symbols file
    dataset: data.Dataset object
        object representing training data for fst
    unseen_transition_weight: float
        weight for unseen transitions 

    Returns
    -------
    fst: FST object
        FST with the melodic and harmonic sequences

    """

    # compute the weights for the FST
    weights_map = {}

    osyms_set = set()

    # create weights for existing transitions
    for song in dataset.songs:

        iseq,oseq = song.get_equal_length_sequences()

        for i in range(len(iseq)-1):
            isym = iseq[i] + iseq[i+1]
            osym = oseq[i+1]
            osyms_set.add(osym)

            if not (isym,osym) in weights_map:
                weights_map[isym,osym] = 1.0
            else:
                w = weights_map[isym,osym]
                weights_map[isym,osym] = coreutils.log_plus(w,1.0)

    pcsyms = ['pc'+str(n) for n in range(12)]
    isyms = ['pc'+str(n) + 'pc'+str(m) for n in range(12) for m in range(12)]

    osyms = list(osyms_set)

    # assemble the sequences, and set any unseen transitions
    fst_isyms = []
    fst_osyms = []
    fst_wts = []

    for isym in isyms:
        if isym == fsm.EPSILON_LABEL:
            continue

        for osym in osyms:
            if osym == fsm.EPSILON_LABEL:
                continue

            fst_isyms.append(isym)
            fst_osyms.append(osym)
            if (isym,osym) in weights_map:
                fst_wts.append(weights_map[isym,osym])
            else:
                fst_wts.append(unseen_transition_weight)

    isyms_table = fsm.SymbolTable(filename=isyms_filename)
    isyms_table.set_symbols(symbols=isyms) #,add_epsilon_label=False)
    isyms_table.write_file()

    osyms_table = fsm.SymbolTable(filename=osyms_filename)
    osyms_table.set_symbols(symbols=osyms) #,add_epsilon_label=False)
    osyms_table.write_file()

    # FST to map second order symbols to chord symbols
    fname = 'Flower.fst'
    print 'building:',fname

    fst = fsm.FlowerFST(filename=fname,\
        isyms_table=isyms_table,\
        osyms_table=osyms_table)
    fst.build(input_symbols=fst_isyms,\
        output_symbols=fst_osyms,\
        weights=fst_wts)
    fst.compile()

    # final 2nd order ergodic FST
    print 'building:',fst_filename

    return fst


def build_chromagrams_from_chorales(time_res=0.0232,\
    bpm=60,\
    dur_beats=None,\
    out_dir=None,\
    transpose_to_c=True):
    """ Build all the possible chromagrams for all the chorales in 
    the music21 corpus """

    bach_flist = music21.corpus.getBachChorales()
    chromagrams = []
    for fn in bach_flist:
        print fn
        fileb = fileutils.filebase(fn)

        stream = music21.corpus.parse(fn)

        if transpose_to_c:
            print 'transposing...'
            key_pc,mode = trainutils.get_key_and_mode(music21_stream=stream)
            key_pc = int(key_pc)
            if mode == 'minor':
                key_pc = (key_pc+3)%12
            stream.transpose(value=-key_pc,inPlace=True)


        n_parts = len(stream.parts)
        idx = set(range(n_parts))
        for n in range(n_parts):
            curridx = idx - set([n])
            currparts = [stream.parts[i] for i in curridx]
            newstream = music21.stream.Stream(currparts)
            cg = m21_stream_to_chromagram(stream=newstream,time_res=time_res,\
                bpm=bpm,dur_beats=dur_beats)
            chromagrams.append(cg)

            if not out_dir is None:
                cgfname = fileb + '_mel' + str(n) + '.pkl'
                print 'writing:',cgfname
                filepath = os.path.join(out_dir,cgfname)
                fileutils.write_pickle_file(data=cg,fname=filepath)

    return chromagrams




def m21_stream_to_chromagram(stream,time_res=0.0232,bpm=60,dur_beats=None):
    """ convert music21 stream to a chromagram using the specified time resolution and BPM """

    frame_dur_beats = ( float(bpm) / 60.0 ) * time_res

    if dur_beats is None:
        dur_beats = stream.duration.quarterLength

    tree = music21.stream.timespans.streamToTimespanCollection(stream)

    frame_times_beats = np.arange(0.0,dur_beats+frame_dur_beats,frame_dur_beats)
    n_frames = len(frame_times_beats)
    chromagram = np.zeros((12,n_frames))

    for t in range(n_frames):
        beat = frame_times_beats[t]
        vert = tree.getVerticalityAt(beat)
        pcs = [n.pitchClass for n in vert.pitchSet]
        chromagram[pcs,t] = 1.0

    return chromagram

# ---------------------------------------------
# code to build a Roman Numeral to chord label
# fst, which isn't needed any more
# ---------------------------------------------
def build_roman_to_label_fst(roman_numerals,\
    isyms_table,\
    osyms_filename='labels.syms',\
    key=0,\
    fst_filename='RtoC.fst'):
    # isyms_filename='roman.syms',\
    # osyms_filename='labels.syms'):
    """ Build a (flower) transducer that maps from roman numerals
    to regular chord symbols, given a key.

    Parameters
    ----------
    roman_numerals: list of strings
        list of Roman numeral symbols
    key: int
        pitch class of key
    fst_filename: string
        name of compiled FST file
    isyms_filename: string
        name of input symbols file
    osyms_filename: string
        name of output symbols file

    Returns
    -------
    fst: FST object
        flower FST that maps from Roman numerals to chord labels

    """

    # roman_numerals = symbols_table.get_symbols()

    romans = roman_numerals
    if fsm.EPSILON_LABEL in romans:
        romans.remove(fsm.EPSILON_LABEL)
    chord_labels = coreutils.get_chord_labels_from_roman(romans,key)

    osyms_table = fsm.SymbolTable(filename=osyms_filename)
    osyms_table.set_symbols_from_seq(sym_seqs=chord_labels)
    osyms_table.write_file()

    fst = None
    # fst = fsm.FST(filename=fst_filename,\
    #     isyms_table=isyms_table,osyms_table=osyms_table)

    # build_flower_fst(fst=fst,\
    #     input_seqs=roman_numerals,\
    #     output_seqs=chord_labels)

    # fst.compile()

    return fst


# ------------------------------------------------------------------------------------------
# hacky code to build rhythm fst
# ------------------------------------------------------------------------------------------

def build_full_rhythm_fst(full_fst_filename,\
    syms_filename,\
    input_sequences,\
    output_sequences):
    """ Build a rhythm FST.

    Parameters
    ----------
    fst_filename: string
        name of compiled FST file
    syms_filename: string
        name of symbols file (same for input and output)
    input_sequences: list of list of strings
        list of quantized input rhythms
    output_sequences: list of list of strings
        list of quantized output rhythms

    Returns
    -------
    fst: FST object
        FST to map input to output rhythm sequences
    """

    # input_seqs, output_seqs = dataset.get_disambiguated_seqs()

    # output_seqs = traindata.expand_harmonic_seqs(melodic_seqs=input_seqs,\
    #     harmonic_seqs=output_seqs)

    # make symbols tables and FST
    all_sequences = copy.deepcopy(input_sequences)
    all_sequences.extend(output_sequences)

    syms_table = fsm.SymbolTable(filename=syms_filename)
    # gross hack
    all_syms = ['k' + str(i) for i in range(33)]
    syms_table.set_symbols_from_seq(sym_seqs=all_syms)
    # syms_table.set_symbols_from_seq(sym_seqs=all_sequences)
    syms_table.write_file()

    # osyms_table = fsm.SymbolTable(filename=osyms_filename)
    # osyms_table.set_symbols_from_seq(sym_seqs=output_sequences)
    # osyms_table.write_file()

    weights = len(input_sequences) * [1.0]

    fst_filename = 'rhythmL.fst'
    ngram_filename = 'rhythmG.fst'

    fst = fsm.BranchFST(filename=fst_filename,\
        isyms_table=syms_table,osyms_table=syms_table)

    # for iseq,oseq in zip(input_sequences,output_sequences):
    #     # if not len(iseq) == len(oseq):
    #     #     print iseq
    #     #     print oseq
    #     #     print            
    #     if '#' in ' '.join(iseq):
    #         print iseq
    #         print oseq
    #         print

    fst.build(input_seqs=input_sequences,\
        output_seqs=output_sequences,\
        weights=weights,\
        add_epsilons=False,\
        make_factor_transducer=False)

    print 'compiling L...'
    fst.compile()
    print 'performing closure on L...'
    fst.closure()
    fst.arcsort(sort_type='olabel')

    print 'building rhythm sequences'
    rhythm_seqs_filename = marl.fileutils.temp_file('txt')
    rhythm_seqs = [' '.join(s) for s in all_sequences]
    fileutils.write_text_file(data=rhythm_seqs, fname=rhythm_seqs_filename)

    order = 2
    print 'n-gram order:',order

    cmd = [os.path.join(fsm.FST_BIN_DIR,'farcompilestrings')]
    # cmd.append('--generate_keys=8')
    cmd.append('--symbols='+syms_table.filename)
    cmd.append('--keep_symbols=1')
    cmd.append(rhythm_seqs_filename)
    cmd.append('|')
    cmd.append(os.path.join(fsm.FST_BIN_DIR,'ngramcount'))
    cmd.append('--order='+str(order))
    cmd.append('|')
    cmd.append(os.path.join(fsm.FST_BIN_DIR,'ngrammake'))
    # if smooth_method is not None:
    #     cmd.append('--method=' + smooth_method)
    #     # if smooth_method == NGRAM_SMOOTH_METHOD_KATZ:
    #     #     cmd.append('--backoff=true')
    #     if smooth_method == NGRAM_SMOOTH_METHOD_WB:
    #         if witten_bell_k is None:
    #             witten_bell_k = 1.0
    #         cmd.append('--witten_bell_k='+str(witten_bell_k))
    #     elif smooth_method == NGRAM_SMOOTH_METHOD_KN:
    #         if kn_discount_D is None:
    #             kn_discount_D = 1.0
    #         if kn_num_bins is None:
    #             kn_num_bins = 5
    #         cmd.append('--discount_D='+str(kn_discount_D))
    #         cmd.append('--bins='+str(kn_num_bins))

    # cmd.append('--method=presmoothed')
    cmd.append('|')
    # cmd.append(os.path.join(FST_BIN_DIR,'ngramshrink'))
    # cmd.append('-method=relative_entropy')
    # cmd.append('-theta='+str(rel_ent_thresh))
    # cmd.append('|')
    cmd.append(os.path.join(fsm.FST_BIN_DIR,'fstarcsort'))

    dbg_msg = ' '.join(cmd)
    print '[build_ngram]',dbg_msg

    stdout,stderr = coreutils.launch_process(command=cmd,\
        outfile_name=ngram_filename,outfile_type='binary')

    if not stderr == '':
        print 'ERROR: unable to build n-gram',ngram_filename
        print 'command:',' '.join(cmd)
        print 'error:',stderr
        return None

    ngram = fsm.FST(filename=ngram_filename,\
        isyms_table=syms_table,\
        osyms_table=syms_table)
    ngram.map_to_log()
    ngram.replace_infinite_weights()

    full_fst = fsm.compose(fst,ngram,full_fst_filename)
    full_fst.build_m2c(do_closure=True,do_determinize_twice=True)        
    print 'removing disambiguation symbol from full FST...'
    full_fst.substitute_disambig_symbol(disambig_sym=traindata.DISAMBIG_SYMBOL)
    full_fst.load_from_compiled()

    return full_fst



def build_fsa(duration_seqs,\
    fsa_filename='rhythm_ngram.fst',\
    syms_filename='rhythm.syms'):
    """ build the FSA from rhythm sequences """
    # make symbols tables and FST

    print 'DO NOT USE'
    return None

    print 'quantizing duration sequences'
    train_seqs = quantize_duration_seqs(duration_seqs=duration_seqs,\
        codebook=codebook)

    print 'building rhythm FSA'
    syms_table = fsm.SymbolTable(filename=syms_filename)
    syms_table.set_symbols_from_seq(sym_seqs=codebook.values())
    syms_table.write_file()

    fsa = fsm.FST(filename=fst_filename,\
        isyms_table=syms_table,osyms_table=syms_table)

    M.build_phoneme_fst(fst=fsa,input_seqs=train_seqs,output_seqs=train_seqs)

    print 'compiling/determinizing/minimizing'
    fsa.compile()
    fsa.determinize()
    fsa.minimize()

    inv_cb_filename = 'rhythm_codebook.json'

    print 'saving inverse codebook to',inv_cb_filename
    with open(inv_cb_filename,'w') as fp:
        json.dump(inverse_codebook,fp)

    return fsa


def disambiguate_rhythm_symbols(io_pairs_dict):
    input_seqs = []
    output_seqs = []

    disambig_cnt = 0

    for input_str,output_str in io_pairs_dict.items():
        unique_outputs = set(output_str)

        if len(unique_outputs)>1:
            for unique_out in unique_outputs:
                input_seq = input_str.split()
                input_seq.append(DISAMBIG_SYMBOL+str(disambig_cnt))
                disambig_cnt += 1

                output_seq = unique_out.split()
                output_seq = [fsm.EPSILON_LABEL if s == '_' else s for s in output_seq]
                output_seq.append(fsm.EPSILON_LABEL)

                input_seqs.append(input_seq)
                output_seqs.append(output_seq)
        else:
            input_seq = input_str.split()
            output_seq = output_str[0].split()
            output_seq = [fsm.EPSILON_LABEL if s == '_' else s for s in output_seq]
            # output_seq.append(fsm.EPSILON_LABEL)

            input_seqs.append(input_seq)
            output_seqs.append(output_seq)

    return input_seqs,output_seqs    

def build_rhythm_io_pairs_dict(input_seqs,output_seqs):

    io_pairs_dict = {}

    for input_seq,output_seq in zip(input_seqs,output_seqs):

        in_seg = ' '.join(input_seq)
        output_seq = [s if not s is None else '_' for s in output_seq]
        out_seg = ' '.join(output_seq)

        if in_seg in io_pairs_dict:
            io_pairs_dict[in_seg].append(out_seg)
        else:
            io_pairs_dict[in_seg] = [out_seg]

    return io_pairs_dict



def quantize_dur(dur,codebook):
    """ Quantize a single duration """

    cb_keys = codebook.keys()
    cb_vals = np.abs(np.array(cb_keys) - dur)
    cb_idx = np.argmin(cb_vals)
    k = cb_keys[cb_idx]
    return codebook[k]

    # if not dur in codebook:
    #     return None

    # return codebook[dur]


# def quantize_seq(seq,codebook):
#     if type(seq) == list:
#         return map(quantize_seq,seq)
#     else:
#         return quantize_dur(seq)

# def quantize_duration_sequences(durations):
#     """ Quantize duration sequences """

#     quant_seq = map(quantize_seq,durations)
#     return quant_seq

# def build_train_pairs_from_segmented_duration(duration_seqs,codebook):
#     """ build quantized training pairs from 
#     segmented (e.g., by measure) duration sequences"""


    # training_pairs = {}
    # training_pairs['input'] = []
    # training_pairs['output'] = []

    # for song_seq in duration_seqs:

    #     for part_num,part_seq in enumerate(song_seq):

    #         # quant_seq_0 =[quantize_dur(d,codebook) for d in part_seq[0]] 
    #         # n_parts = len(part_seq)

    #         # # for meas_idx,measure_seq in enumerate(part_seq):
    #         # for i in range(1,n_parts):
    #         #     measure_seq = part_seq[i]
    #         #     quant_seq = [quantize_dur(d,codebook) for d in measure_seq]

    #         #     # curr_meas_seqs.append(curr_seq)                
    #         #     training_pairs['input'].append(quant_seq_0)
    #         #     training_pairs['output'].append(quant_seq)

    #         # training_seqs[part_num].append(curr_meas_seq)

    #         curr_meas_seqs = []

    #         print part_seq
    #         for meas_seq in part_seq:
    #             quant_seq = [quantize_dur(d,codebook) for d in meas_seq]
    #             curr_meas_seqs.append(quant_seq)

    #         lengths = [len(s) for s in curr_meas_seqs]
    #         print len(curr_meas_seqs)
    #         input_idx = np.argmax(lengths)
    #         print input_idx
    #         input_seq = curr_meas_seqs[input_idx]
    #         out_idx = set(range(len(curr_meas_seqs))) - set([input_idx])

    #         for i in out_idx:
    #             training_pairs['input'].append(quant_seq[input_idx])
    #             training_pairs['output'].append(quant_seq[i])



    # return training_pairs

def build_codebook(n_decimal_places=5,prefix='k'):
    """ quantize the durations """

    durations = get_data_for_files_in_dir()

    # alldurs = set()
    # for durs in durations:
    #     for d in set(durs.tolist()):
    #         alldurs.add(d)

    filtdurs = set()
    for d in durations:
        if d<=0 or d>8.0:
            continue
        # d = np.around(d,decimals=n_decimal_places)
        d = coreutils.truncate_num(d,n_decimals=n_decimal_places)
        filtdurs.add(d)
    # del alldurs

    # return filtdurs
    codebook = {}
    inv_codebook = {}

    i = 0
    for d in filtdurs:
        code = prefix + str(i)
        i += 1
        codebook[d] = code
        inv_codebook[code] = d

    return codebook,inv_codebook

def get_file_list(basedir=MOZART_CORPUS_DIR):
    """ get a list of all files to read from a
    music21 corpus/composer directory

    assume that basedir is a composer, so that everything
    below is a directory containing files (e.g. musicXML)

    """

    filelist = []

    print 'get files in:'
    print basedir
    dlist = glob.glob(os.path.join(basedir,'*'))
    for d in dlist:
        if os.path.isdir(d):
            tmplist = glob.glob(os.path.join(d,'*.*'))
            files = [t for t in tmplist if os.path.isfile(t)]
            filelist.extend(files)

    return filelist

def reject_dur_seq(dur_seq,codebook,inv_codebook,tol=1e-2):
    quant_seq = [quantize_dur(d,codebook) for d in dur_seq]
    dequant_seq = [inv_codebook[q] for q in quant_seq]

    return (np.sum(dur_seq) - np.sum(dequant_seq))>tol

def sanity_check_durations(durations,codebook,inv_codebook):
    cnt = 0
    s = 0
    for iseq,oseq in zip(durations['input'],durations['output']):
        if reject_dur_seq(iseq,codebook,inv_codebook) or \
            reject_dur_seq(oseq,codebook,inv_codebook):
            cnt += 1
        # qdur = [quantize_dur(d,codebook) for d in iseq]
        # unqdur = [inv_codebook[q] for q in qdur]
        # qdiff = np.sum(iseq) - np.sum(unqdur)
        # # if not np.sum(iseq) == np.sum(unqdur):
        # if qdiff > 1e-2: 
        #     print '.',
        #     cnt += 1
        #     s += qdiff

        # qdur = [quantize_dur(d,codebook) for d in oseq]
        # unqdur = [inv_codebook[q] for q in qdur]
        # qdiff = np.sum(oseq) - np.sum(unqdur)
        # # if not np.sum(oseq) == np.sum(unqdur):
        # if qdiff > 1e-2: 
        #     print '+',
        #     cnt += 1
        #     s += qdiff

    print
    print 'count:',cnt
    print 'sum:',s


def quantize_duration_pairs(duration_pairs,codebook,inv_codebook,segment_by_beat=True,add_additional=True):
    """ Quantize the duration pairs """

    quant_input_seqs = []
    quant_output_seqs = []

    cnt = 0

    for input_seq,output_seq in zip(duration_pairs['input'],duration_pairs['output']):
        if cnt % 1000 == 0:
            print cnt
        cnt += 1

        if input_seq == [] or output_seq == []:
            continue

        if reject_dur_seq(dur_seq=input_seq,codebook=codebook,inv_codebook=inv_codebook) or \
            reject_dur_seq(dur_seq=output_seq,codebook=codebook,inv_codebook=inv_codebook):
            continue

        if segment_by_beat:
            quant_in_seqs,quant_out_seqs =\
                quantize_and_segment_one_duration_pair(input_seq=input_seq,\
                    output_seq=output_seq,\
                    codebook=codebook)
            for qiseq,qoseq in zip(quant_in_seqs,quant_out_seqs):
                quant_input_seqs.append(qiseq)
                quant_output_seqs.append(qoseq)
        else:
            quant_in_seq,quant_out_seq =\
                quantize_one_duration_pair(input_seq=input_seq,\
                    output_seq=output_seq,\
                    codebook=codebook)
            quant_input_seqs.append(quant_in_seq)
            quant_output_seqs.append(quant_out_seq)


    if add_additional:
        add_iseqs,add_oseqs = add_additonal_sequences()
        for iseq,oseq in zip(add_iseqs,add_oseqs):
            quant_input_seqs.append(iseq)
            quant_output_seqs.append(oseq)



    return quant_input_seqs,quant_output_seqs

def add_additonal_sequences():
    quant_input_seqs = []
    quant_output_seqs = []

    qiseq = 24*['k6']
    qoseq = len(qiseq) * [None]
    qoseq[0] = 'k1'
    quant_input_seqs.append(qiseq)
    quant_output_seqs.append(qoseq)        

    qiseq = 4*['k1']
    qoseq = len(qiseq) * [None]
    qoseq[0] = 'k4'
    quant_input_seqs.append(qiseq)
    quant_output_seqs.append(qoseq)        

    qiseq = ['k9','k14']
    qoseq = len(qiseq) * [None]
    qoseq[0] = 'k3'
    quant_input_seqs.append(qiseq)
    quant_output_seqs.append(qoseq)        


    qiseq = ['k13','k14']
    qoseq = len(qiseq) * [None]
    qoseq[0] = 'k4'
    quant_input_seqs.append(qiseq)
    quant_output_seqs.append(qoseq)        

    qiseq = ['k2','k11']
    qoseq = len(qiseq) * [None]
    qoseq[0] = 'k19'
    quant_input_seqs.append(qiseq)
    quant_output_seqs.append(qoseq)

    qiseq = ['k23', 'k31']
    qoseq = len(qiseq) * [None]
    qoseq[0] = 'k1'
    quant_input_seqs.append(qiseq)
    quant_output_seqs.append(qoseq)

    qiseq = ['k19','k1']
    qoseq = len(qiseq) * [None]
    qoseq[0] = 'k24'
    quant_input_seqs.append(qiseq)
    quant_output_seqs.append(qoseq)

    qiseq = 4*['k22']
    qoseq = len(qiseq) * [None]
    qoseq[0] = 'k14'
    quant_input_seqs.append(qiseq)
    quant_output_seqs.append(qoseq)

    return quant_input_seqs,quant_output_seqs

def find_sequence_alignment(input_seq,output_seq):
    """ Find the sequence alignment """
    istart = input_seq[:-1]
    istart.insert(0,0.0)
 
    intervals_from = np.zeros((len(input_seq),2))
    intervals_from[:,0] = np.cumsum(istart)
    intervals_from[:,1] = np.cumsum(input_seq)

    iend = output_seq[:-1]
    iend.insert(0,0.0)

    intervals_to = np.zeros((len(output_seq),2))
    intervals_to[:,0] = np.cumsum(iend)
    intervals_to[:,1] = np.cumsum(output_seq)

    intervals_map = librosa.util.match_intervals(intervals_from=intervals_from,\
        intervals_to=intervals_to)

    return intervals_map

def quantize_and_segment_one_duration_pair(input_seq,output_seq,codebook):
    """ Quantize and segment duration pair (segment so that there's only 
        one output symbol per quantized output sequence) """

    intervals_map = find_sequence_alignment(input_seq=input_seq,\
        output_seq=output_seq)

    change_flags = np.diff(intervals_map).tolist()
    change_flags.insert(0,1)

    quant_input_seqs = []
    quant_output_seqs = []

    curr_in_seq = []
    curr_out_seq = []

    for input_dur,map_idx,change in \
        zip(input_seq,intervals_map.tolist(),change_flags):

        input_q = quantize_dur(dur=input_dur,codebook=codebook)

        # same map index as previous
        if change == 0:
            output_q = None
            curr_in_seq.append(input_q)
            curr_out_seq.append(output_q)
        # new map index
        else:
            if not curr_in_seq == [] and not curr_out_seq == []:
                quant_input_seqs.append(curr_in_seq)
                quant_output_seqs.append(curr_out_seq)

            output_q = quantize_dur(dur=output_seq[map_idx],codebook=codebook)
            curr_in_seq = [input_q]
            curr_out_seq = [output_q]

    return quant_input_seqs,quant_output_seqs


def quantize_one_duration_pair(input_seq,output_seq,codebook):
    """ Quantize one pair of input/output duration sequences """

    intervals_map = find_sequence_alignment(input_seq=input_seq,\
        output_seq=output_seq)

    quant_input_seqs = []
    quant_output_seqs = []

    prev_idx = -1
    for input_dur,map_idx in zip(input_seq,intervals_map.tolist()):
        input_q = quantize_dur(dur=input_dur,codebook=codebook)
        if not map_idx == prev_idx:
            output_q = quantize_dur(dur=output_seq[map_idx],codebook=codebook)
            prev_idx = map_idx
        else:
            output_q = None

        quant_input_seqs.append(input_q)
        quant_output_seqs.append(output_q)

    return quant_input_seqs,quant_output_seqs

def get_data_for_files_in_dir(basedir=MOZART_CORPUS_DIR,segmented_data=True,all_pairs=True):
    """ get the durations data for everything in directory """
    dur_pairs = {}
    dur_pairs['input'] = []
    dur_pairs['output'] = []

    filelist = get_file_list(basedir=basedir)
    N = len(filelist)
    i = 1
    for f in filelist:
        print i,'/',N
        i += 1
        try:
            if segmented_data:
                dur_seqs = get_segmented_data_for_one_file(filename=f)
            else:
                dur_seqs = get_data_for_one_file(filename=f)
        except IndexError:
            print 'ERROR with file:',f
            continue

        if segmented_data:
            # durations.append(d)
            for meas_seqs in dur_seqs:
                if all_pairs:
                    n_seqs = len(meas_seqs)
                    print '# seqs:',n_seqs
                    for i in range(n_seqs):
                        for j in range(n_seqs):
                            if i == j:
                                continue
                            dur_pairs['input'].append(meas_seqs[i])
                            dur_pairs['output'].append(meas_seqs[j])

                else:
                    lengths = [len(s) for s in meas_seqs]
                    input_idx = np.argmax(lengths)

                    print input_idx

                    input_seq = meas_seqs[input_idx]
                    out_idx = set(range(len(meas_seqs))) - set([input_idx])

                    for i in out_idx:
                        dur_pairs['input'].append(meas_seqs[input_idx])
                        dur_pairs['output'].append(meas_seqs[i])

                        # dur_pairs['input'].append(meas_seqs[i])
                        # dur_pairs['output'].append(meas_seqs[input_idx])

        else:
            print 'doing nothing!'
            return None
            # durations.extend(d)

        if DEBUG:
            break

    return dur_pairs

def get_data_for_one_file(filename,\
    filter_time_sig=True,\
    win_size_meas=4,
    hop_size_meas=1):
    """ get the rhythm data for a single file """

    try:
        print 'loading:',filename
        score = music21.converter.parse(filename)
        if filter_time_sig and not valid_time_signatures(score=score):
            return None

    except Exception, details:
        print 'ERROR processing',filename
        print details
        print sys.exc_info()[0]
        return None

    print 'getting durations'
    # n_parts = len(score.parts)
    # print n_parts
    # parts_durs = [[] for _ in range(n_parts)]
    # parts_durs = []
    durations = []
    for part in score.parts:
        # times = get_onsest_from_stream(stream=part)
        # durs = get_durations_from_onsets(onset_times=times)
        durs = get_durations_from_stream(stream=part)
        durations.extend(durs)

    # return parts_durs
    return durations

def get_segmented_data_for_one_file(filename,\
    filter_time_sig=True):
    """ get the rhythm data for a single file """

    try:
        print 'loading:',filename
        score = music21.converter.parse(filename)
        if filter_time_sig and not valid_time_signatures(score=score):
            return None

    except Exception, details:
        print 'ERROR processing',filename
        print details
        print sys.exc_info()[0]
        return None

    print 'getting durations'
    # n_parts = len(score.parts)
    # print n_parts
    # parts_durs = [[] for _ in range(n_parts)]
    # parts_durs = []
    durations = []
    for part in score.parts:
        # times = get_onsest_from_stream(stream=part)
        # durs = get_durations_from_onsets(onset_times=times)
        durs = get_segmented_durations_from_stream(stream=part)
        durations.append(durs)

    # return parts_durs
    return durations


def get_segmented_durations_from_stream(stream):
    """ get the durations from a single 
    music21 stream grouped by measure"""

    durations = []
    tree = music21.stream.timespans.streamToTimespanCollection(stream,\
        flatten=True,\
        classList=(music21.note.Note,music21.chord.Chord,music21.note.Rest))
    n_meas = tree[-1].measureNumber+1

    measures = [[] for _ in range(n_meas)]

    for note in tree:
        meas_num = note.measureNumber
        measures[meas_num].append(note)

    for i in range(n_meas):
        measure = measures[i]

        if len(measure) == 0:
            durations.append([])
            continue

        times = []
        # curr_meas_beats = i*4.0
        # next_meas_beats = (i+1)*4.0

        times = [n.startOffset for n in measure]
        # if i == n_meas-1:
        #     for n in measure:
        #         print n.measureNumber,n.startOffset

        # times.insert(0,curr_meas_beats)

        delta_t = np.diff(times)
        delta_t = [t for t in delta_t if t>0.0]
        n = measure[-1]
        t = n.stopOffset - n.startOffset
        if t>0.0:
            delta_t.append(t)
        # if not n.stopOffset == next_meas_beats:
        #     delta_t.append(next_meas_beats - n.stopOffset)

        durations.append(delta_t)

    return durations

def get_durations_from_stream(stream):
    """ get the durations from a single 
    music21 stream"""

    durations = []
    tree = music21.stream.timespans.streamToTimespanCollection(stream)
    n_meas = len(stream)

    measures = [[] for _ in range(n_meas)]

    for note in tree:
        m = note.measureNumber
        measures[m].append(note)

    for measure in measures:
        times = [n.stopOffset - n.startOffset for n in measure]
        durations.extend(times)

    return durations

def valid_time_signatures(score,beatCount=4,denom=4):
    for timesig in score.getTimeSignatures():
        if not timesig.beatCount == 4 and not timesig.denominator == 4:
            return False

    return True



def build_rhythm_fst_from_data():
    """ Do all the steps to build the full rhythm FST """

    
    print 'Building...'
    print

    # get codebook
    home_dir = os.getenv('HOME')
    codebook_fname = 'rhythm_fwd_cb.json'
    codebook_path = os.path.join(home_dir,'Dropbox','work','src','hamilton',codebook_fname)
    inv_codebook_fname = 'rhythm_codebook.json'
    inv_codebook_path = os.path.join(home_dir,'Dropbox','work','src','hamilton',inv_codebook_fname)

    tmp_codebook = fileutils.read_json_file(fname=codebook_path)
    inv_codebook = fileutils.read_json_file(fname=inv_codebook_path)

    # hack to make sure codebook keys are floats, not strings
    codebook = {}
    for k,v in tmp_codebook.items():
        codebook[float(k)] = v

    # get training sequences
    print '\n'
    print 80*'.'
    print 'get durations'
    durations = get_data_for_files_in_dir()

    print '\n'
    print 80*'.'
    print 'quantize durations'
    quant_input_seqs,quant_output_seqs = quantize_duration_pairs(durations,codebook,inv_codebook)

    quant_iseqs = []
    quant_oseqs = []

    print 'filter non-matching duration sequences'
    for q_in,q_out in zip(quant_input_seqs,quant_output_seqs):
        durs_in = [inv_codebook[str(s)] for s in q_in]
        durs_out = [inv_codebook[str(s)] if not s is None else 0.0 for s in q_out]
        if np.sum(durs_in) == np.sum(durs_out):
            quant_iseqs.append(q_in)
            quant_oseqs.append(q_out)
        else:
            print '.',

    iopairs = build_rhythm_io_pairs_dict(quant_iseqs,quant_oseqs)
    input_seqs,output_seqs = disambiguate_rhythm_symbols(iopairs)

    # for iseq,oseq in zip(input_seqs,output_seqs):
    #     txti = ' '.join(iseq)
    #     txto = ' '.join(oseq)
    #     if '#' in txti:
    #         print txti
    #         print txto
    #         print

    print '\n'
    print 80*'.'
    print 'build FST'
    fst = build_full_rhythm_fst(full_fst_filename='rhythmTST.fst',\
        syms_filename='rhythmTST.syms',\
        input_sequences=input_seqs,\
        output_sequences=output_seqs)

    return fst

def main():
    beat_quant_level = 0.5
    lambda_max = 14
    # lambda_max = None

    # phi = -14.0
    phi = None

    print 40*'--'
    print 'building dataset'
    print 40*'--'

    if phi is None:
        print 'lambda_max:',lambda_max
    else:
        print 'phi:',phi
    print 'repeating melodic events using beat quant level:',beat_quant_level

    jamsdir = os.path.expanduser('~/Dropbox/testjams')
    mapfn = os.path.expanduser('~/Dropbox/work/src/hamilton/roman2majmindim.json')
    dataset = traindata.Dataset(jams_dir=jamsdir,\
        chord_label_map_file=mapfn,\
        melodic_context_len=lambda_max,\
        transpose_to_12_keys=False)

    if not beat_quant_level is None:
        dataset.repeat_melodic_events(beat_quant_level=beat_quant_level)

    print
    print 40*'--'
    print 'building FSTs'
    print 40*'--'

    if phi is not None:
        fst_suffix = '_erg2'
    else:
        if not beat_quant_level is None:
            tmp = str(beat_quant_level)
            tmp = tmp.replace('.','')
            fst_suffix = '_bq'+ tmp + '_L' + str(lambda_max)
        else:
            fst_suffix = '_L' + str(lambda_max)

    print 'using beat quant level:',beat_quant_level

    Lfn = 'L' + fst_suffix + '.fst'
    Gfn = 'G' + fst_suffix + '.fst'
    LoGfn = 'LoG' + fst_suffix + '.fst'
    isymsfn = 'isyms' + fst_suffix + '.syms'
    osymsfn = 'osyms' + fst_suffix + '.syms'
    chord_seqs_fn = 'chords' + fst_suffix + '.txt'

    fsts = build_LoG(lang_mod_filename=Lfn,\
        ngram_filename=Gfn,\
        ngram_order=2,\
        isyms_filename=isymsfn,\
        osyms_filename=osymsfn,\
        dataset=dataset,\
        chord_seqs_filename=chord_seqs_fn,\
        LoG_filename=LoGfn,\
        unseen_transition_weight=phi)

    print '\ndone'

if __name__ == '__main__':
    main()    
    # fst = build_rhythm_fst_from_data()

