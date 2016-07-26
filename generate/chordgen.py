""" Generate accompaniment from melodic sequences, etc. """

import copy
import os
import time
import numpy as np
import scipy

import hamilton.core.data as coredata
import hamilton.core.fsm as F
import hamilton.core.utils as CU

GEN_METHOD_LOG_PROB = 'log_prob'
GEN_METHOD_UNIFORM = 'uniform'
GEN_METHOD_SHORTEST = 'shortest'

BUILD_2ND_ORDER_ERGODIC = False


def build_rand_output_path_fst(fst,\
    path_fst_filename,\
    rand_method):
    """ Build a single output path from the
    specified FST.

    Parameters
    ----------
    fst: FST object
        the FST to find an output path through
    path_fst_filename: string
        the filename of the output path FST
    rand_method: string
        method to generate random accompaniment
        sequences (see fstrandgen documentation)
    """

    seed = int(1000000*np.random.random()) #int(10000*time.time())
    cmd = [os.path.join(F.FST_BIN_DIR,'fstrandgen')]
    cmd.append('--select='+rand_method)
    cmd.append('--seed='+str(seed))
    cmd.append(fst.filename)
    cmd.append('|')
    cmd.append(os.path.join(F.FST_BIN_DIR,'fsttopsort'))
    cmd.append('|')
    cmd.append(os.path.join(F.FST_BIN_DIR,'fstrmepsilon'))

    stdout,stderr = CU.launch_process(command=cmd,\
        outfile_name=path_fst_filename,\
        outfile_type='binary')

    if not stderr == '':
        print 'ERROR: unable to build path FST',path_fst_filename
        print 'command:',' '.join(cmd)
        print 'error:',stderr
        return None

    path_fst = F.FST(filename=path_fst_filename,\
        isyms_table=fst.isyms_table,\
        osyms_table=fst.osyms_table)
    path_fst.load_from_compiled()

    return path_fst

def build_nshortest_paths_fst(fst,\
        path_fst_filename,\
        npaths):
    """ given an FST, return an FST that has the n-shortest paths """

    cmd = [os.path.join(F.FST_BIN_DIR,'fstmap')]
    cmd.append('--map_type=to_standard')
    cmd.append(fst.filename)
    cmd.append('|')
    cmd.append(os.path.join(F.FST_BIN_DIR,'fstshortestpath'))
    cmd.append('--nshortest=' + str(npaths))
    cmd.append('|')
    cmd.append(  os.path.join(F.FST_BIN_DIR,'fsttopsort') )
    cmd.append('|')
    cmd.append(  os.path.join(F.FST_BIN_DIR,'fstrmepsilon') )
    cmd.append('|')
    cmd.append(os.path.join(F.FST_BIN_DIR,'fstmap'))
    cmd.append('--map_type=to_log')

    stdout,stderr = CU.launch_process(command=cmd,\
        outfile_name=path_fst_filename,\
        outfile_type='binary')

    if not stderr == '':
        print 'ERROR: unable to build FST',path_fst_filename
        print 'command:',' '.join(cmd)
        print 'error:',stderr
        return None

    path_fst = F.LinearChainFST(filename=path_fst_filename,\
        isyms_table=fst.isyms_table, \
        osyms_table=fst.osyms_table,\
        overwrite_isyms_file=False)
    path_fst.map_to_log()
    path_fst.load_from_compiled()

    print 'total weight for',path_fst_filename,'=',path_fst.total_weight()

    return path_fst

def segment_sequences_by_harmony(melodic_seq,\
    harmonic_seq):
    """ segment the melodic and harmonic sequences into
    regions of static harmony

    Parameters
    ----------
    melodic_seq: list of strings
        list of melodic symbols
    harmonic_seq: list of strings
        list of harmonic symbols

    Returns
    -------
    seg_idx: list of (int,int) tuples
        list of segment (start,end) indices (inclusive)
        for each segment
    """

    assert len(melodic_seq) == len(harmonic_seq)
    assert not melodic_seq[0] == F.EPSILON_LABEL

    seq_len = len(melodic_seq)

    seg_idx = []
    start_i = 0

    for i in range(seq_len):
        note = melodic_seq[i]
        chord = harmonic_seq[i]

        if note == F.EPSILON_LABEL and chord == F.EPSILON_LABEL:
            continue
        elif not chord == F.EPSILON_LABEL:
            seg_idx.append((start_i,i))
            start_i = i+1

    return seg_idx

def get_rand_gen_rhythm_seq(rhythm_fst,\
    codebook,\
    method=GEN_METHOD_LOG_PROB,\
    path_fst_filename='tmp_path.fst',\
    pref_tot_dur=0.1):
    """ generate a random rhythm sequence 

    Parameters
    ----------
    rhythm_fst: FST object
        FST used to generate accompaniment rhythms
    codebook: dictionary
        symbol->duration codebook
    method: string
        method used to randomly generate a path
    path_fst_filename: string
        filename of path FST (i.e. file containing result 
        of composition of input sequence FST and output sequence FST)
    pref_tot_dur: float
        preferred total duration of output sequence (typically set this
        to the total duration of the input sequence)

    Returns
    -------
    in_dur_seq: list of floats
        list of input sequence durations (as floats)
    out_dur_seq: list of floats
        list of output sequence durations (as floats)
    """

    out_dur_seq = []
    max_iter = 20
    curr_iter = 0
    while np.sum(out_dur_seq)<pref_tot_dur and curr_iter < max_iter:
        # print 'sum=',np.sum(out_dur_seq)

        # print 'iter #',curr_iter,'/',max_iter
        path_fst = build_rand_output_path_fst(fst=rhythm_fst,\
            path_fst_filename=path_fst_filename,\
            rand_method=method)

        path_fst.load_from_compiled()
        _,out_rhy_seq = path_fst.get_input_output_syms()

        out_dur_seq.extend([codebook[s] for s in out_rhy_seq if not s == F.EPSILON_LABEL])
        curr_iter += 1

    return out_dur_seq

def get_rhythm_seq(in_duration_syms,\
    rhythm_fst,\
    codebook,\
    method=GEN_METHOD_UNIFORM,\
    path_fst_filename='tmp_path.fst'):
    """ build a rhythm sequence 

    Parameters
    ----------
    in_duration_syms: list of strings
        list of quantized duration symbols (e.g. 'k0') 
        representing durations of input sequence
    rhythm_fst: FST object
        FST used to generate accompaniment rhythms
    codebook: dictionary
        symbol->duration codebook
    method: string
        method used to randomly generate a path
    path_fst_filename: string
        filename of path FST (i.e. file containing result 
        of composition of input sequence FST and output sequence FST)
    pref_tot_dur: float
        preferred total duration of output sequence (typically set this
        to the total duration of the input sequence)

    Returns
    -------
    in_dur_seq: list of floats
        list of input sequence durations (as floats)
    out_dur_seq: list of floats
        list of output sequence durations (as floats)
    """

    tmp_fst = build_melody_fst(melody_seq=in_duration_syms,\
        isyms_filename=rhythm_fst.isyms_table,\
        melody_fst_filename='tmp_rhythm_1.fst')

    all_paths_fst = F.compose(tmp_fst,rhythm_fst,'tmp_all_paths.fst')

    path_fst = build_rand_output_path_fst(fst=all_paths_fst,\
                path_fst_filename=path_fst_filename,\
                rand_method=method)

    in_rhy_seq,out_rhy_seq = path_fst.get_input_output_syms()

    in_dur_seq = [codebook[s] for s in in_rhy_seq if not s == F.EPSILON_LABEL]
    out_dur_seq = [codebook[s] for s in out_rhy_seq if not s == F.EPSILON_LABEL]

    return in_dur_seq,out_dur_seq

def get_harmonic_seq(melody_sequence,\
    full_fst,\
    method=GEN_METHOD_LOG_PROB,\
    npaths=1,\
    melody_fst_filename='melfst.fst',\
    out_harmony_fst_filename='harmfst.fst',\
    path_fst_filename='path.fst',\
    build_dir='.',\
    return_path_weight=False):

    """ Build the harmonic sequence from an input symbol
    sequence and a LoG FST.

    Parameters
    ----------
    melody_sequence: core.data.Sequence object
        Sequence object representing melody
    full_fst: FST object
        full FST model (composition of L and G,
        determinized/minimized, disambiguation symbols removed)
    method: string
        method to generate random accompaniment
        sequences (see fstrandgen documentation)
    npaths: int
        number of paths to generate (used only when using
        shortest paths method)
    melody_fst_filename: string
        name of compiled melody FST
    out_harmony_fst_filename: string
        name of compiled FST file consisting of output
        harmony sequence
    build_dir: string
        build directory (for FSTs)
    return_path_weight: Boolean
        return the path weight with harmonic sequence (as a tuple)
    Returns
    -------
    output_sequence: core.data.Sequence object
        Sequence object representing harmony, with timing information
        from melody sequence object
    """

    print 'build path fst'
    path_fst = build_path_fst(melody_sequence=melody_sequence,\
        full_fst=full_fst,\
        path_fst_filename=path_fst_filename,\
        melody_fst_filename=melody_fst_filename,\
        out_harmony_fst_filename=out_harmony_fst_filename,\
        method=method,\
        npaths=npaths,\
        build_dir=build_dir)

    if path_fst is None:
        print 'ERROR: unable to build path_fst'
        if return_path_weight:
            return None,np.inf
        else:
            return None

    song = coredata.song_from_fst(melodic_sequence=melody_sequence,\
        harmonic_fst=path_fst)

    if return_path_weight:
        w = path_fst.total_weight()
        return song.harmonic_sequence,w
    else:
        return song.harmonic_sequence

def get_harmonic_seq_separate_fsts(melody_sequence,\
    L_fst,\
    G_fst,\
    method=GEN_METHOD_LOG_PROB,\
    npaths=1,\
    melody_fst_filename='melfst.fst',\
    out_harmony_fst_filename='harmfst.fst'):

    """ Build the harmonic sequence from an input symbol
    sequence and separate L (i.e. note->chord FST) 
    and G (i.e. chord model) FSTs.

    Parameters
    ----------
    melody_sequence: core.data.Sequence object
        Sequence object representing melody
    L_fst: FST object
        L FST model (i.e. note->chord model), 
        determinized/minimized, disambiguation symbols removed
    G_fst: FST object
        G FST model (i.e. n-gram chord model)
    method: string
        method to generate random accompaniment
        sequences (see fstrandgen documentation)
    npaths: int
        number of paths to generate (used only when using
        shortest paths method)
    melody_fst_filename: string
        name of compiled melody FST
    out_harmony_fst_filename: string
        name of compiled FST file consisting of output
        harmony sequence

    Returns
    -------
    output_sequence: core.data.Sequence object
        Sequence object representing harmony, with timing information
        from melody sequence object
    """

    path_fst = build_path_fst_separate_fsts(melody_sequence=melody_sequence,\
        L_fst=L_fst,\
        G_fst=G_fst,\
        melody_fst_filename=melody_fst_filename,\
        out_harmony_fst_filename=out_harmony_fst_filename,\
        method=method,
        npaths=npaths)

    if path_fst is None:
        print 'ERROR: unable to build path_fst'
        return None

    song = coredata.song_from_fst(melodic_sequence=melody_sequence,\
        harmonic_fst=path_fst)

    return song.harmonic_sequence


def build_path_fst_separate_fsts(melody_sequence,\
    L_fst,\
    G_fst,\
    melody_fst_filename='melfst.fst',\
    out_harmony_fst_filename='harmfst.fst',\
    method=GEN_METHOD_LOG_PROB,
    npaths=1):

    """ Build the harmonic sequence from an input symbol
    sequence and a LoG FST.

    Parameters
    ----------
    melody_sequence: core.data.Sequence object
        Sequence object representing melody
    L_fst: FST object
        L FST model (i.e. note->chord model), 
        determinized/minimized, disambiguation symbols removed
    G_fst: FST object
        G FST model (i.e. n-gram chord model)
    method: string
        melody_fst_filename: string
        name of compiled melody FST
    out_harmony_fst_filename: string
        name of compiled FST file consisting of output
        harmony sequence
    method: string
        method to generate random accompaniment
        sequences (see fstrandgen documentation)
    npaths: int
        number of paths to generate (used only when using
        shortest paths method)

    Returns
    -------
    output_sequence: core.data.Sequence object
        Sequence object representing harmony, with timing information
        from melody sequence object
    """

    if BUILD_2ND_ORDER_ERGODIC:
        mel_seq = melody_sequence.get_label_pairs()
    else:
        mel_seq = melody_sequence.get_labels()

    melfst = build_melody_fst(melody_seq=mel_seq, \
        isyms_filename=L_fst.isyms_table,\
        melody_fst_filename=melody_fst_filename)

    # outfst = F.compose(melfst,full_fst,out_harmony_fst_filename)
    # form (M o L)
    MoG_fn = 'tmp_MoG.fst'
    MoGfst = F.compose(melfst,L_fst,MoG_fn)
    
    # print '******* determinizing and minimizing M o G'
    # MoGfst.determinize()
    # MoGfst.minimize()

    # form (M o G) o G
    outfst = F.compose(MoGfst,G_fst,out_harmony_fst_filename)

    if method == GEN_METHOD_LOG_PROB or method == GEN_METHOD_UNIFORM:
        path_fst = build_rand_output_path_fst(fst=outfst,\
            path_fst_filename='path.fst',\
            rand_method=method)
    elif method == GEN_METHOD_SHORTEST:
        path_fst = build_nshortest_paths_fst(fst=outfst,\
            path_fst_filename='path.fst',\
            npaths=npaths)
    else:
        print 'ERROR: unknown sequence generation method',method
        path_fst = None

    del outfst
    del melfst
    del MoGfst

    return path_fst

def build_path_fst(melody_sequence,\
    full_fst,\
    path_fst_filename='path.fst',\
    melody_fst_filename='melfst.fst',\
    out_harmony_fst_filename='harmfst.fst',\
    method=GEN_METHOD_LOG_PROB,
    npaths=1,\
    build_dir='.'):

    """ Build the harmonic sequence from an input symbol
    sequence and a LoG FST.

    Parameters
    ----------
    melody_sequence: core.data.Sequence object
        Sequence object representing melody
    full_fst: FST object
        full FST model (composition of L and G,
        determinized/minimized, disambiguation symbols removed)
    melody_fst_filename: string
        name of compiled melody FST
    out_harmony_fst_filename: string
        name of compiled FST file consisting of output
        harmony sequence
    method: string
        method to generate random accompaniment
        sequences (see fstrandgen documentation)
    npaths: int
        number of paths to generate (used only when using
        shortest paths method)
    build_dir: string
        build directory (for FSTs)

    Returns
    -------
    output_sequence: core.data.Sequence object
        Sequence object representing harmony, with timing information
        from melody sequence object
    """

    if BUILD_2ND_ORDER_ERGODIC:
        mel_seq = melody_sequence.get_label_pairs()
    else:
        mel_seq = melody_sequence.get_labels()

    melody_fst_filename = os.path.join(build_dir,melody_fst_filename)
    print 'build melody fst',melody_fst_filename

    melfst = build_melody_fst(melody_seq=mel_seq, \
        isyms_filename=full_fst.isyms_table,\
        melody_fst_filename=melody_fst_filename)

    out_harmony_fst_filename = os.path.join(build_dir,out_harmony_fst_filename)

    print 'compose melody with full fst'
    print 'out harmony fst',out_harmony_fst_filename
    outfst = F.compose(melfst,full_fst,out_harmony_fst_filename)

    # path_fst_filename = os.path.join(build_dir,'path.fst')
    path_fst_filename = os.path.join(build_dir,path_fst_filename)

    print 'build path fst... '
    print 'path fst',path_fst_filename

    if method == GEN_METHOD_LOG_PROB or method == GEN_METHOD_UNIFORM:
        path_fst = build_rand_output_path_fst(fst=outfst,\
            path_fst_filename=path_fst_filename,\
            rand_method=method)
    elif method == GEN_METHOD_SHORTEST:
        path_fst = build_nshortest_paths_fst(fst=outfst,\
            path_fst_filename=path_fst_filename,\
            npaths=npaths)
    else:
        print 'ERROR: unknown sequence generation method',method
        path_fst = None

    del outfst
    del melfst

    return path_fst

def build_melody_fst(melody_seq,\
    isyms_filename,\
    melody_fst_filename='melodic_seq.fst'):
    """ Build an FST from a sequence of melodic symbols.

    Parameters
    ----------
    melody_seq: list of strings
        list of melodic sequences
    isyms_filename: string
        name of input symbols file
    melody_fst_filename: string
        name of compiled FST
    """

    mel_fst = F.LinearChainFST(filename=melody_fst_filename,\
        isyms_table=isyms_filename,\
        osyms_table=isyms_filename)
    mel_fst.build(input_symbols=melody_seq)
    mel_fst.rmepsilon()

    return mel_fst
