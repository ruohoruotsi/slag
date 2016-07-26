""" Common utility functions for building/evaluating models """


import copy
import glob
import numpy as np

from music21 import analysis as m21A

import hamilton.core.utils as coreutils
# from pyjams import pyjams
import pyjams # assuming src_dir/pyjams is in python path


def get_key_and_mode(music21_stream,\
    return_relative_major=True):
    """ Get the pitch class of the key and
    the mode of the music21 Stream object.

    Parameters
    ----------
    music21_stream: music21 Stream object
        song to estimate key of
    return_relative_major: boolean
        If minor mode, indicate whether or not to
        return relative major (e.g., if key is A minor,
        return C major)

    Returns
    -------
    key: int
        pitch class of key
    mode: string
        either 'major' or 'minor'     
    """

    key = m21A.discrete.analyzeStream(\
        music21_stream, 'Krumhansl')

    pitchObj,mode = key.pitchAndMode
    keypc = pitchObj.pitchClass
    
    if return_relative_major and mode == 'minor':
        keypc = (keypc+3) % 12
        mode = 'major'

    return keypc,mode

def import_jams(jams_dir,\
    ignore_filelist):
    """ import jams files from specified directory.

    Parameters
    ----------
    jams_dir: str
        path to directory containing jams files
    ignore_filelist: list of strings
        list of file names to ignore

    Returns
    -------
    jams: list of jams objects
        list of jams objects representing files in jams_dir
    """

    filelist = glob.glob(jams_dir+'/*.jams')
    jams = []

    for fname in filelist:
        if fname in ignore_filelist:
            continue
        jam = pyjams.load(fname)
        jams.append(jam)

    return jams

def get_measure_times_from_jam(jam,\
    beat_annot_num=0):
    """ Get the measure times from the jam file

    Parameters
    ----------
    jam: JAMS object
        JAMS object containing beat annotation
    beat_annot_num: int
        beat annoation index number

    Returns
    -------
    measure_times: list of floats
        list of times in seconds of each measure.
        measure_times[m] = t, where m = measure number (starting from 0) and
        t is time in seconds
    """

    # some measure times may be missing, 
    # I think measure 0 missing usually??
    meas_times = [d.time.value for \
        d in jam.beat[beat_annot_num].data]
    meas_nums = [d.label.secondary_value for \
        d in jam.beat[beat_annot_num].data]
    max_meas_num = meas_nums[-1]
    measure_times = (max_meas_num+1)*[0.0]

    for m,t in zip(meas_nums,meas_times):
        measure_times[m] = t

    return measure_times

def get_beat_sync_melodic_chunks(chords,\
    notes,\
    duration,\
    key_normalize=True,\
    beats_per_meas=0.25):
    """ Chunk key normalized melody by beat. Default is 4 beats per measure
    (not necessarily the best assumption, but hopefully it will work).

    Parameters
    ----------
    chords: JAMS RangeAnnotation (jams "chord[i].data" field)
        chords
    notes: JAMS RangeAnnotation (jams "note[i].data" field)
        melody notes
    duration: float
        duration of the song in measures
    key_normalize: boolean
        flag to indicate whether or not to transpose
        melody note values to key of C major
    beats_per_meas: float
        number of beats per measure, e.g. 0.25 corresponds to 4 beats/measure
        (4/4 time)

    Returns
    -------
    chunks: list of list of JAMS RangeAnnotations
        each list of RangeAnnotations contains the melody notes for
        the corresponding beat, which ranges from 0.0 to duration,
        at intervals specified by beats_per_meas.
    """

    if key_normalize:
        # first, key-normalize (which requires looking at harmonic context)
        notes = key_normalize_melody(notes=notes, chords=chords)

    # now group by beat
    # beats = np.arange(0.0, duration+beats_per_meas, beats_per_meas)
    beats = np.arange(0.0, duration, beats_per_meas)
    n_beats = len(beats)
    chunks = [ [] for _ in range(n_beats) ]

    # TODO: should do this more efficiently....
    for i in range(n_beats-1):
        for note in notes:
            if note.start.value >= beats[i] and \
            note.start.value < beats[i+1]:
                chunks[i].append(note)
            # elif note.end.value > beats[i]:
            elif note.start.value >= beats[i+1]:
                break

    return chunks

def key_normalize_melody(notes,chords):
    """ Key normalize a melody. Put key-normalized pitch class value into
    secondary_value field.

    Parameters
    ----------
    chords: JAMS RangeAnnotation (jams "chord[i].data" field)
        chords
    notes: JAMS RangeAnnotation (jams "note[i].data" field)
        melody notes

    Returns
    -------
    key_norm_notes: JAMS RangeAnnotation (jams "note[i].data" field)
        melody notes, with key normalized pitch class in secondary_value field
    """

    key_norm_notes = copy.deepcopy(notes)

    for note in key_norm_notes:
        # find chord occurring during current note, only check note start
        # times since note end times are approximated
        # TODO: figure out a more efficient way to do this
        for chord in chords:
            if note.start.value >= chord.start.value and \
            note.start.value <= chord.end.value:
                key_pc = chord.label.secondary_value
                tr_pc = coreutils.transpose_pc_to_c(pc=note.label.value, key=key_pc)
                note.label.secondary_value = tr_pc
                break

    return key_norm_notes


