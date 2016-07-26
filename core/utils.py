""" Various utility functions """

import hamilton.core.fileutils as U

import os
import subprocess

#TODO: get rid of this dependency if possible!!
try:
    import music21
except ImportError:
    print 'Unable to import music21'



import numpy as np
import scipy.linalg
import scipy.spatial


HARM_DISS_WEIGHTS = np.array( [90.0, 30.0, 15.0, 12.0, 9.0, 50.0] )
HARM_DISS_NORM = HARM_DISS_WEIGHTS.sum()

EPS = 1e-20

NUM_MIDI_OCTS = 12

PITCH_CLASSES = ['C', 'Db', 'D', 'Eb', 'E', 'F',
                 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']



# REAL_BOOK_CHORDS = {
#     '100100100000' : 'dim',
#     '100100100100' : 'dim7',
#     '100101100100' : 'dim7(11)',

#     '100101100010' : 'hdim7(11)',
#     '100100100010' : 'hdim7',

#     '101101010110' : 'min13',
#     '101100010000' : 'min(9)',
#     '100101010010' : 'min7(11)',
#     '101101000110' : 'min11',                     #???
#     '101101010010' : 'min11',
#     '100100010010' : 'min7',

#     '101010010110' : '9(13)',
#     '101010010010' : '9',
#     '100110010010' : '7#9',
#     '110010010010' : '7b9',
#     '100110001010' : '7+#9',
#     '100010011010' : '7(b13)',
#     '110010100010' : '7b5b9',     #<---??

#     '100010001001' : '+maj7',
#     '100010010001' : 'maj7',
#     '100010010000' : 'maj',
#     '100010100001' : 'maj7b5',
#     '101010010000' : 'maj(9)',
#     '100010110001' : 'maj7#11',
#     '110010010001' : 'maj7(b9)',

#     '110010010000' : 'maj(b9)',

#     '100011001000' : '+(11)',
#     '101010001000' : '+(9)',
#     '101010001001' : '+maj9',


#     '101100101000' : '?',                       #???
#     '101000100100' : '?',                       #???
#     '100100000100' : '?',                       #???
#     '100101101010' : '?',                       #???
#     '100100101010' : '?',                       #???
#     '101001001000' : '?',                       #???
#     '100001010001' : '?',                       #???
#     '100101001010' : '?',                      #['Chord Symbol Cannot Be Identified']
#     '100101001000' : '?',                       #???
#     '101010100100' : '?',                       #???
#     '101000100110' : '?',                       #???
#     '101011000100' : '?',                       #???
#     '100000100100' : '?',                       #???

#     '101010100101' : 'maj9b5(13)',              #???
#     '100101011010' : 'maj(9,b13)',              #???


#     '101001000100' : '?',                       #Dmin7
#     '100100001000' : '?',                       #<--------really an Ab major 
#     '100100011000' : '?',                       #<--------really an Ab maj7

#     '101100010010' : 'min9',
#     '101011001000' : '+(9,11)',
#     '101001100100' : '?',                       #???
#     '101000011000' : 'sus2(#5)',                #???
#     '100001001000' : '?',                       #???
#     '100010001010' : '7+',
#     '100011010000' : 'maj(11)',
#     '101010110101' : 'maj13#11',
#     '101000110000' : 'sus2(#11)',
#     '101011010110' : '13',
#     '100010101000' : '?',                       #<---- Ab+7
#     '110101010010' : 'm7(b9,11)',               #???
#     '101100011000' : 'm(9,b13)',                #???
#     '100100101000' : '?',                       #<----- Ab7
#     '101010010001' : 'maj9',
#     '101000100101' : '?',                       #???
#     '100110101000' : '?',                       #??? <--- looks like another Ab chord
#     '101100100100' : '?',                       #???
#     '100100010001' : 'min(maj7)',
#     '100101000100' : '?',                       #F7
#     '100100010000' : 'min',
#     '110010001010' : '7+(b9)',
#     '100011000010' : '?',                       #???
#     '100101000101' : '?',                       #<---looks like F7(#11)
#     '100100010100' : 'min6',
#     '101100001000' : '?',                       #??? Ab(#11)?
#     '101000100010' : '?',                       # F#(#11)??
#     '100010010010' : '7',
#     '101011000101' : '?',                       #???
#     '101101000100' : '?',                       #???
#     '100010110010' : '7(#11)',
#     '110001001000' :  '?',                      #Dbmaj7
#     '101011010101' :  'maj13'    
# }

INTERVALS = {'maj':         '100010010000',\
             'min':         '100100010000',\
             'maj7':        '100010010001',\
             'min7':        '100100010010',\
             '7':           '100010010010',\
             'maj6':        '100010010100',\
             'min6':        '100100010100',\
             'dim':         '100100100000',\
             'aug':         '100010001000',\
             'sus4':        '100001010000',\
             'sus2':        '101000010000',\
             'hdim7':       '100100100010',\
             'dim7':        '100100100100',\
             '7(b5)':       '100010100010',\
             '(b9)' :       '110010010000',\
             'maj(b9)' :    '110010010000',\
             '(9)' :        '101010010000',\
             'min7(9,13)' : '101100010110',\
             'sus(b9)' :    '110001010000',\
             '7(b9)' :      '110010010010',\
             'min6(9)' :    '101100010100',\
             '6(9)' :       '100010010100',\
             'maj6(9)' :    '100010010100',\
             'min(b6)' :    '100100011000',\
             'min7(11)' :   '100101010010',\
             '(#11)' :      '100010110000',\
             'min(13)' :    '100100010100',\
             'maj(11)' :    '100011010000',\
             '(9)' :        '101010010000',\
             '7(9,13)' :    '101010010110',\
             '9(13)' :      '101010010110',\
             'aug7(b9)' :   '110010010010',\
             'maj9' :       '101010010001',\
             '6' :          '100010010100',\
             '9' :          '101010010010',\
             '9(#11)' :     '101010110010',\
             'maj7(#11)' :  '100010110001',\
             '7(13)' :      '100010010110',\
             'maj7(#5)' :   '100010001001',\
             '7(#5)' :      '100010001010',\
             'min13' :      '101101010110',\
             'min(maj7,13)':'100100010101',\
             'min9' :       '101100010010',\
             'min9(13)' :   '101100010110',\
             'min(9)' :     '101100010000',\
             'sus7' :       '100001010010',\
             '13' :         '101011010110',\
             '13(b9)' :     '110011010110',\
             '13(#11)' :    '101010110110',\
             'min11' :      '101101010010',\
             'min7(b6)' :   '100100011010',\
             'min7(#5)' :   '100100001010',\
             'aug7' :       '100010001010',\
             'maj7(b9)' :   '110010010001',\
             '7(#9)' :      '100110010010',\
             '7(b13)' :     '100010011010',\
             'min(maj7)' :  '100100010001',\
             '7(#11)' :     '100010110010',\
             'maj(9)':      '101010010000'
              }

QUALITIES = INTERVALS.keys()

# QUALITIES = ['maj', 'min', 'maj7', 'min7', '7', 'maj6', 'min6',\
#              'dim', 'aug', 'sus4', 'sus2', 'hdim7', 'dim7', \
#              '7(b5)', '(b9)', 'min7(9,13)', 'sus(b9)', '7(b9)', 'min6(9)',\
#              '6(9)', 'min(b6)', 'min7(11)', '(#11)', 'min(13)', 'maj(11)', '(9)',\
#              '7(9,13)', 'aug7(b9)', 'maj9', '6', '9', 'maj7(#11)',\
#              '7(13)', 'maj7(#5)', 'min13', 'min(maj7,13)', 'min9', 'sus7',\
#              '13', 'min11', 'min7(b6)', 'min7(#5)', 'aug7', 'maj7(b9)', '7(#9)',\
#              '7(b13)', 'min(maj7)', '7(#11)', 'maj(9)']

CHORD_TYPE_TO_3_TRIAD_MAP = {
    'maj'   : 'maj', 
    '7'     : 'maj', 
    'maj7'  : 'maj',
    'aug'   : 'maj', 
    'sus4'  : 'maj', 
    'min'   : 'min',
    'min7'  : 'min',
    'dim'   : 'dim',
    'hdim7' : 'dim',
    '7(b5)' : 'maj', 
    '(b9)'  :   'maj',
    'min7(9,13)' : 'min',
    'sus(b9)' : 'maj',
    '7(b9)' : 'maj',
    'min6(9)' : 'min',
    '6(9)'  : 'maj',
    'min(b6)'   : 'min',
    'min7(11)'  : 'min',
    '(#11)' : 'maj',
    'min(13)'   : 'min',
    'maj(11)'   : 'maj',
    '(9)'   : 'maj',
    '7(9,13)' : 'maj',
    'aug7(b9)'  : 'maj',
    'maj9' : 'maj',
    '6' : 'maj',
    '9' : 'maj',
    'maj7(#11)' : 'maj',
    '7(13)' : 'maj',
    'maj7(#5)' : 'maj',
    'min13' : 'min',
    'min(maj7,13)' : 'min',
    'min9' : 'min',
    'sus7' : 'maj',
    '13' : 'maj',
    'min11' : 'min',
    'min7(b6)' : 'min',
    'min7(#5)' : 'min',
    'aug7' : 'maj',
    'maj7(b9)' : 'maj',
    '7(#9)' : 'maj',
    '7(b13)' : 'maj',
    'min(maj7)' : 'min',
    '7(#11)' : 'maj',
    'maj(9)' : 'maj',
    'min(9)' : 'min',
    '9(13)' : 'maj',
    '7(#5)' : 'maj',
    'maj(b9)' : 'maj',
    '13(b9)' : 'maj',
    'min9(13)' : 'min',
    '13(#11)' : 'maj',
    'dim7' : 'dim',
    'maj6' : 'maj',
    'maj6(9)' : 'maj',
    'sus2' : 'maj',
    'min6' : 'min',
    '9(#11)' : 'maj'
    }

# build chord map
ntypes = len(INTERVALS.keys())
CHORD_MAP = np.zeros((ntypes,12))
for ci,templ in enumerate(INTERVALS.values()):
    for ti,templ_val in enumerate(templ):
        CHORD_MAP[ci,ti] = float(templ_val)

ENHARMONIC_MAP = {'A':'A', 'Ab':'Ab', 'A#':'Bb', 'B':'B', 'Bb':'Bb', \
                'B#':'C', 'C':'C', 'Cb':'B', 'C#':'Db', 'D':'D', \
                'Db':'Db', 'D#':'Eb', 'E':'E', 'Eb':'Eb', 'E#':'F', \
                'F':'F', 'Fb':'E', 'F#':'Gb', \
                'G':'G', 'Gb':'Gb', 'G#':'Ab', \
                'Abb':'G', 'Bbb':'A', 'Cbb':'Bb', \
                'Dbb':'C', 'Ebb':'D', 'Fbb':'Eb', \
                'Gbb':'F', 'A##':'B', 'B##':'Db', \
                'C##':'D', 'D##':'E', 'E##':'F#', \
                'F##':'Ab', 'G##':'A'}

NOTE_TO_PC_MAP = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']

def reduce_chord_type(chord,\
    type_map=CHORD_TYPE_TO_3_TRIAD_MAP):
    """ Reduce a complex chord type down to a simpler type 

    Parameters
    ----------
    chord: string
        chord label in 'root:type' format
    type_map: dictionary
        dictionary that maps chord types, e.g. maps 'hdim7' to 'dim'
    """

    if not ':' in chord:
        return chord
    root,chordtype = chord.split(':')
    newchord = root + ':' + type_map[chordtype]

    return newchord

def midi_note_to_pc(n):
    """ convert a MIDI note number to corresponding pitch class

    Parameters
    ----------
    n: integer
        MIDI note number (0-127)

    Returns
    -------
    pc: integer
        pitch class (0-11)
    """

    return n%12

def midi_note_to_oct(n):
    """ convert a MIDI note number to corresponding pitch class

    Parameters
    ----------
    n: integer
        MIDI note number (0-127)

    Returns
    -------
    octave: integer
        octave number (0-10)
    """

    octave = int(n) / NUM_MIDI_OCTS
    return octave

def pc_oct_to_midi_note(pc,octave,prev_note=None):
    """ convert a pitch class and an octave value to
    corresponding MIDI note number.

    Parameters
    ----------
    pc: integer
        pitch class (0-11)
    octave: integer
        octave number (0-10)

    Returns
    -------
    n: integer
        MIDI note number (0-127)
    """

    if not prev_note is None:
        n0 = octave*NUM_MIDI_OCTS + pc
        n1 = (octave-1)*NUM_MIDI_OCTS + pc
        if np.abs(prev_note-n0) < np.abs(prev_note-n1):
            n = n0
        else:
            n = n1
    else:
        n = octave*NUM_MIDI_OCTS + pc
    return n

def transpose_midi_note_to_c(note,key):
    """ transpose MIDI note to the key of C major,
    given the specified key

    Parameters
    ----------
    note: integer
        MIDI note (0-127)
    key: integer
        pitch class of key (0-11)

    Returns
    -------
    transposed_n: integer
        transposed MIDI note (0-127)
    """

    pc = midi_note_to_pc(n=note)
    octave = midi_note_to_oct(n=note)
    newpc = transpose_pc_to_c(pc=pc, key=key)
    transposed_n = pc_oct_to_midi_note(pc=newpc, octave=octave)

    return transposed_n

def transpose_pc_from_c(pc,to_key):
    """ transpose pitch class to the specified key

    Parameters
    ----------
    pc: integer or string
        pitch class (0-11) or 
        'pcn' where n is a pitch class preceded by 'pc'
    key: integer
        pitch class of key (0-11)

    Returns
    -------
    transposed_pc: integer or string
        transposed pitch class (0-11) or 'pcn' where n
        is transposed pitch class
    """

    if type(pc) == str:
        is_string = True
        pc = pc.replace('pc','')
        pc = int(pc)
    else:
        is_string = False

    pc_transp = (pc+to_key)%12

    if is_string:
        return 'pc' + str(pc_transp)
    else:
        return pc_transp

def transpose_pc_to_c(pc, key):
    """ transpose pitch class to the key of C major,
    given the specified key

    Parameters
    ----------
    pc: integer
        pitch class (0-11)
    key: integer
        pitch class of key (0-11)

    Returns
    -------
    transposed_pc: integer
        transposed pitch class (0-11)
    """

    return (pc-key)%12

def transpose_chord_from_c(chord,to_key):
    """ transpose chord to the specified key

    Parameters
    ----------
    chord: string
        chord label, in "standard" (Harte?) format, e.g. "C:maj"
    to_key: integer
        pitch class of key (0-11)

    Returns
    -------
    transposed_chord: string
        chord label, in "standard" (Harte?) format, e.g. "C:maj",
        transposed to specified key
    """

    root,qual = chord.split(':')
    root = ENHARMONIC_MAP[root]
    try:
        rootpc = NOTE_TO_PC_MAP.index(root)
    except ValueError:
        print '\tunable to find ',root
        return chord
    tr_rootpc = (rootpc + to_key) % 12
    tr_root = NOTE_TO_PC_MAP[tr_rootpc]

    transposed_chord = tr_root + ':' + qual
    return transposed_chord

def transpose_chord_to_c(chord,from_key):
    """ transpose chord from the specified key to C major

    Parameters
    ----------
    chord: string
        chord label, in "standard" (Harte?) format, e.g. "C:maj"
    to_key: integer
        pitch class of key (0-11)

    Returns
    -------
    transposed_chord: string
        chord label, in "standard" (Harte?) format, e.g. "C:maj",
        transposed to key of C major
    """

    root,qual = chord.split(':')
    root = ENHARMONIC_MAP[root]
    try:
        rootpc = NOTE_TO_PC_MAP.index(root)
    except ValueError:
        print '\tunable to find ',root
        return chord
    tr_rootpc = transpose_pc_to_c(pc=rootpc, key=from_key)
    tr_root = NOTE_TO_PC_MAP[tr_rootpc]

    transposed_chord = tr_root + ':' + qual
    return transposed_chord

def beats_to_times(beat_values,\
    measure_times):
    """ Convert beat values to times in seconds 

    Parameters
    ----------
    beat_values: list of floats
        beat values in measures and fractions of measures to convert to seconds
    measure_times: list of floats
        list of times (in seconds) of occurrence of each measure
        (e.g., the time that measure m begins is measure_times[m]) 

    Returns
    -------
    beat_times: list of floats
        list of beat values in seconds
    """

    beat_times = []

    for b in beat_values:
        beat_frac = get_frac_part(b)
        beat_int = int(b)

        # is the beat the beginning of a measure or not
        # TODO: find a better way to deal with arbitrary floating point error
        if beat_frac < 0.0001:
            t = measure_times[beat_int]
        else:
            start_t = measure_times[beat_int]
            end_t = measure_times[beat_int+1]  # NOTE: assume that beat isn't after last measure
            t = start_t + beat_frac*(end_t-start_t)
        beat_times.append(t)

    return beat_times

def get_frac_part(n):
    """ get the fractional part from a float.
    For example, if n = 123.45, return 0.45.

    Parameters
    ----------
    n: float
        number

    Returns
    -------
    m: float
        fractional portion
    """

    return float(str(n-int(n)))

def get_onsets_from_cg(cg):
    """ return a numpy array of the indices of the chromagram that correspond
    to onsets.
    cg should be 12xT """

    # add a frame of zeros to beginning and take diff
    tmp = np.hstack( [np.zeros((12,1)), cg] )
    onsets = np.diff(tmp.sum(axis=0))
    onsets[onsets<0.0] = 0.0

    return onsets.nonzero()

def mask_cg(cg,idx):
    """ mask a chromagram using the provided indices. Note that the
    indices refer to frames of chromagram to *preserve* not remove.
    cg should be 12xT """

    maskcg = np.zeros(cg.shape)
    maskcg[:,idx] = cg[:,idx]

    return maskcg

def convert_m21_corpus_to_midi(corpus_name,midi_dir):
    """ convert each song in a music21 corpus to a MIDI file """

    corpus_stream = music21.corpus.parse(corpus_name)
    corpus_name = corpus_name.replace('/','_')

    n_scores = len(corpus_stream.scores)
    # write some files
    for i in range(n_scores):
        mfname = os.path.join(midi_dir,corpus_name + '_' + str(i))
        mf = music21.midi.translate.streamToMidiFile(corpus_stream.scores[i])
        mf.open(mfname,'wb')
        mf.write()
        mf.close()

def beat_to_idx(beat,\
    beats_per_measure,\
    tempo,\
    time_res):
    """ convert a beat in measure format to an index 

    Parameters
    ----------
    beat: float
        time in beat format (measure#.beat, where by default 1 beat = 0.25)
    beats_per_measure: int
        number of beats in a measure (e.g. 4 for 4/4 time signature)
    tempo: float
        tempo in beats per minute
    time_res: float
        time resolution (length of frame in seconds)

    Returns
    -------
    n: int
        index number of frame corresponding to particular beat, given
        other parameters
    """

    t = beats_per_measure * beat / ( float(tempo) / 60.0 )
    n = int( np.round(t/time_res) )
    return n

def beat_sync_chord_seq_to_chromagram(beat_sync_seq,\
    convert_from_roman=True,
    beats_per_measure=4,\
    bpm=60.0,\
    time_res=0.1):
    """ convert a beat synchronous chord sequence to a chromagram """

    numbeats = len(beat_sync_seq)
    ntimeframes = beat_to_idx(beat=numbeats,\
        beats_per_measure=beats_per_measure,\
        bpm=bpm,\
        time_res=time_res)
    cg = np.zeros((12,ntimeframes))

    for b in range(numbeats-1):
        start_i = beat_to_idx(beat=b,beats_per_measure=beats_per_measure,bpm=bpm,time_res=time_res)
        end_i = beat_to_idx(beat=b+1,beats_per_measure=beats_per_measure,bpm=bpm,time_res=time_res)
        c = beat_sync_seq[b]
        if convert_from_roman and not c == 'R':
            c = roman_numeral_to_label(roman=c, key_pc=0)[0]  # why a list?
        vec = chord_symbol_to_pitch_vector(chord=c)
        print start_i,end_i,c
        cg[:,start_i:end_i] = vec.reshape((12,1))

    return cg

def beat_aligned_chord_seq_to_chromagram(beat_aligned_seq,beats_per_measure=4,bpm=60.0,time_res=0.1):
    """ convert beat-aligned chord sequence to a chromagram """

    end_beat = beat_aligned_seq[-1]['end_beat']
    N = beat_to_idx(beat=end_beat,beats_per_measure=beats_per_measure,bpm=bpm,time_res=time_res)

    cg = np.zeros((12,N))

    for m in beat_aligned_seq:
        start_i = beat_to_idx(beat=m['start_beat'],beats_per_measure=beats_per_measure,bpm=bpm,time_res=time_res)
        end_i = beat_to_idx(beat=m['end_beat'],beats_per_measure=beats_per_measure,bpm=bpm,time_res=time_res)
        vec = chord_symbol_to_pitch_vector(chord=m['chord'])
        cg[:,start_i:end_i] = vec.reshape((12,1))

    return cg

def add_beat_strength_tag(note,beat):
    """ return the beat strength tag:
        if beat is downbeat (n.0) or 3rd beat (n.5) of measure -> 's' (strong)
        if beat is 2nd (n.25) or 4th (n.75) beat of measure -> 'm' (medium)
        otherwise, -> 'w' (weak)

        NOTE that we're ignoring time signature here!!
        (probably should fix that)
    """

    b = beat - np.floor(beat)

    if b == 0.0 or b == 0.5:
        tag = 's'
    elif b == 0.25 or b == 0.75:
        tag ='m'
    else:
        tag = 'w'

    return str(note) + '_' + tag + '_'

def roman_numeral_to_label(roman,key_pc):
    """ given a music21 RomanNumeral instance, return the full chord label
    Note that RomanNumeral should have key information
    so we can determine root """

    chord = music21.roman.RomanNumeral(roman)
    key = PITCH_CLASSES[key_pc]
    key = key.replace('b','-')
    chord.key = music21.key.Key(key)

    ctypes = roman_numeral_to_quality(chord)
    rname = chord.root().name
    rname = rname.replace('-','b')
    rname = ENHARMONIC_MAP[rname]
    lab = [ rname+':'+ctype for ctype in ctypes ]

    return lab

def build_roman_to_label_map(roman_numerals,key=0):
    """ Build a map from Roman numerals to chord labels in
    specified key.

    Parameters
    ----------
    roman_numerals: list of strings
        list of Roman numeral symbols
    key: int
        pitch class of key

    Returns
    -------
    rom_to_lab_map: dictionary
        dictionary that maps Roman numerals (dictionary keys)
        to chord labels (dictionary values) for the specified
        key
    """

    rom_to_lab_map = {}

    for r in roman_numerals:
        l = roman_numeral_to_label(roman=r,key_pc=key)
        # for some reason, roman_numeral_to_label() returns a list. why?????
        rom_to_lab_map[r] = l

    return rom_to_lab_map

def get_chord_labels_from_roman(roman_numerals,key):
    """ use Roman numerals to get chord labels """

    chord_labels = []

    for r in roman_numerals:
        l = roman_numeral_to_label(roman=r,key_pc=key)
        # for some reason, roman_numeral_to_label() returns a list. why?????
        chord_labels.append(l[0])

    return chord_labels

def roman_numeral_to_quality(roman):
    """ given a music21 RomanNumeral instance, return the chord quality.
    Note that RomanNumeral should have key information
    so we can determine root """

    # make a chroma vector from chord and transpose to 'C'
    pvec = np.zeros((12,))
    pcs = roman.pitchClasses
    pvec[pcs] = 1.0
    pvec = np.roll(pvec,-roman.root().pitchClass)

    i = find_closest_chord(chord_map=CHORD_MAP,vec=pvec)
    quals = [ INTERVALS.keys()[i] ]
    return quals

def quantize_beat(beat_val):
    """ quantize a beat value (float) """
    print '***** this isn\'t really working! use round_to instead! ***** '
    return None

    # TODO: allow different quanitzation levels (i.e., not just 0.0 and 0.5)
    frac = float(str(beat_val - np.floor(beat_val)))
    if  frac == 0.25 or frac == 0.5 or frac == 0.75:
        return beat_val
    else:
        return np.round(beat_val)

def round_to(number, precision):
    """ from:
    stackoverflow.com/questions/4265546/python-round-to-nearest-05
    """

    correction = 0.5 if number >= 0 else -0.5
    return int( number/precision+correction ) * precision

def truncate_num(number,n_decimals=5):
    """ truncate a number to  specified number of decimals """

    fac = 10**n_decimals
    tmp = int(number*fac)
    return float(tmp)/float(fac)

def find_closest_chord(chord_map,vec):
    """ find the index of the chord in the chord map closest to the
    chord represented by the vector  """

    nchords = chord_map.shape[0]
    min_i = -1
    min_d = np.inf
    for i in range(nchords):
        d = scipy.spatial.distance.cosine(chord_map[i,:],vec)
        if d < min_d:
            min_d = d
            min_i = i

    return min_i

def pitch_vector_to_chord_symbol(pvec):
    """ convert a pitch vector (list or numpy vector) to a chord symbol """

    print 'this is shit'
    if type(pvec) == list:
        pvec = np.array(pvec)

    c = music21.chord.Chord(list(pvec.nonzero()[0]))

    root = music21.harmony.chordSymbolFigureFromChord(c, False)[0]
    qual = music21.harmony.chordSymbolFigureFromChord(c, True)[0]
    qual = qual.replace(root,'')

    return qual

def get_root_pc(chord):
    """ Get the pitch class of the root of the chord,
    in standard (Harte?) format, e.g. "A:7"

    """

    if ':' in chord:
        root,_ = chord.split(':')
    else:
        root = chord
    root = ENHARMONIC_MAP[root]
    rootpc = PITCH_CLASSES.index(root)

    return rootpc

def melody_symbol_to_pitch_vector(note,\
    prefix=None):
    """ convert a melody symbol to a pitch vector """

    if type(note) == str and not prefix is None:
        note = note.replace(prefix,'')

    try:
        pc = int(note)
        pc = pc % 12

    except ValueError:
        print 'ERROR converting',note,'to pitch class'
        return None

    vec = np.zeros((12,))
    vec[pc] = 1.0

    return vec

def chord_symbol_to_pitch_vector(chord):
    """ convert a chord symbol to a pitch vector
    chord symbol should be of the format 'Root:qual' (e.g. 'C:maj').
    """

    if chord == 'R':
        vec = np.zeros((12,))
    else:
        root,qual = chord.split(':')

        root = ENHARMONIC_MAP[root]
        rootpc = PITCH_CLASSES.index(root)

        qual_idx = QUALITIES.index(qual)
        vec = CHORD_MAP[qual_idx,:]
        vec = np.roll(vec,rootpc)

    return vec

def zero_pad_cg(cg,begin_pad_len=0,end_pad_len=0):
    """
    Zero pad a chromagram (or tonnetz-gram) that's either 12xN or 6xN
    where N = number of frames.
    """
    n_feats = cg.shape[0]

    if begin_pad_len > 0:
        begin_pad = np.zeros((n_feats,begin_pad_len))
    if end_pad_len > 0:
        end_pad = np.zeros((n_feats,end_pad_len))

    cglist = []
    if begin_pad_len > 0 and end_pad_len > 0:
        cglist = [begin_pad,cg,end_pad]
    elif begin_pad_len > 0:
        cglist = [begin_pad,cg]
    elif end_pad_len > 0:
        cglist = [cg,end_pad]
    else:
        return cg

    return np.hstack(cglist)

def normalize_mat(mat,axis=0,method='max1'):
    """
    Normalize matrix.
    Use axis=0 if matrix is 12xT, otherwise use axis=1
    """

    if method == 'max1' or method is None:
        max_vals = np.amax(mat,axis=axis) + np.spacing(1)
        if axis==0:
            norm_mat = mat/max_vals[np.newaxis,:]
        elif axis==1:
            norm_mat = mat/max_vals[:,np.newaxis]
        else:
            print 'ERROR[Utils.normalize()]: invalid axis value:',axis
            return None
    elif method == 'L2' or method == 'l2':
        T = mat.shape[1]
        norm_mat = np.zeros(mat.shape)
        for t in range(T):
            norm_mat[:,t] = mat[:,t]/(scipy.linalg.norm(mat[:,t]) + \
                np.spacing(1))
    return norm_mat

def cosine_dist(X,Y):
    """ cosine distance between matrices X and Y """

    n = (X*Y).sum(axis=0)
    d = np.sqrt( np.sum( (X)**2,axis=0 ) ) *\
        np.sqrt( np.sum( (Y)**2,axis=0 ) ) + EPS
    dists = 1.0 - n/d

    return dists

def correlation_dist(X,Y):
    """ compute the correlation distance between matrices X and Y """
    Xd = X - np.mean(X,axis=0)
    Yd = Y - np.mean(Y,axis=0)
    n = (Xd*Yd).sum(axis=0)
    d = np.sqrt( np.sum( (Xd)**2,axis=0 ) ) * \
        np.sqrt( np.sum( (Yd)**2,axis=0 ) ) + EPS
    dists = 1.0 - n/d

    return dists

def fft2d_dist(X,Y):
    """ compute the 2d fft distance between matrices X and Y """
    # X1 = np.log1p( np.abs( np.fft.fft2(a=X) ) )
    # X2 = np.log1p( np.abs( np.fft.fft2(a=Y) ) )
    X1 = np.abs( np.fft.fft2(a=X) )
    X2 = np.abs( np.fft.fft2(a=Y) )

    d = ( 1.0 / X1.size ) * np.sqrt(np.sum((X1-X2)**2))
    # d = np.sqrt(np.sum((X1-X2)**2))

    return d

def launch_process(command,outfile_name=None,outfile_type='binary'):
    """ Launch a process (potentially with pipes) using
    the args list (list of strings, starting with executable,
    followed by any arguments)
    """

    # need to split up commands separated by pipes into separate lists
    # e.g. [ 'ls', '|', 'grep', 'txt' ] => [ ['ls'], ['grep','txt'] ]
    command_list = []

    sub_cmd_list = []
    for curr_cmd in command:
        if curr_cmd == '|':
            command_list.append( sub_cmd_list )
            sub_cmd_list = []
        # # NOTE: assume re-direct to a file is last command
        # (aside from file name)
        # # AND that file is BINARY!
        # elif curr_cmd = '>':
        #   command_list.append( sub_cmd_list )
        #   outfilename = command[-1]
        #   break
        else:
            sub_cmd_list.append( curr_cmd )
    command_list.append( sub_cmd_list )

    # if output being redirected to a file, create it
    outfile_type = outfile_type.lower()
    if outfile_type == 'binary' or outfile_type == 'bin' or outfile_type == 'b':
        outfile_type = 'b'
    elif outfile_type == 'text' or outfile_type == 'txt' or outfile_type == 't':
        outfile_type = 't'
    elif outfile_name is not None:
        print 'ERROR: unknown output file type',outfile_type
        return None

    if outfile_name is not None:
        fh = open(outfile_name,'w'+outfile_type)

    # now launch all the processes
    procs = []

    nprocs = len(command_list)

    # TODO: find a better solution here
    if len(command_list) == 1:
        if outfile_name is not None:
            proc = subprocess.Popen(command_list[0], \
                stdout=fh, stderr=subprocess.PIPE)
        else:
            proc = subprocess.Popen(command_list[0], \
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_text,stderr_text = proc.communicate()
    else:
        for i,command in enumerate(command_list):
            if i == 0:
                procs = [ subprocess.Popen(command,stdout=subprocess.PIPE) ]
            elif i == nprocs-1 and outfile_name is not None:
                procs.append( subprocess.Popen(command, \
                    stdin=procs[-1].stdout, stderr=subprocess.PIPE, \
                    stdout=fh,\
                    shell=True) )
            else:
                procs.append( subprocess.Popen(command, \
                    stdin=procs[-1].stdout, stderr=subprocess.PIPE, \
                    stdout=subprocess.PIPE,\
                    shell=True) )

        procs[0].stdout.close()
        stdout_text,stderr_text = procs[-1].communicate()

    if outfile_name is not None:
        fh.close()

    return (stdout_text,stderr_text)


def build_maj_min_dim_templates(out_fname=None):
    """ build a 12x36 matrix of binary templates for major, minor,
    and dimiinished triads. """

    templates = np.zeros((12,36))

    # C major
    templates[0,0] = 1.0
    templates[4,0] = 1.0
    templates[7,0] = 1.0

    # c minor
    templates[0,12] = 1.0
    templates[3,12] = 1.0
    templates[7,12] = 1.0

    # c dim
    templates[0,24] = 1.0
    templates[3,24] = 1.0
    templates[6,24] = 1.0


    # now transpose everything...
    for t in range(1,12):
        # major
        vec = np.roll(templates[:,0],t)
        templates[:,t] = vec

        # minor
        vec = np.roll(templates[:,12],t)
        templates[:,12+t] = vec

        vec = np.roll(templates[:,24],t)
        templates[:,24+t] = vec


    if out_fname is not None:
        U.write_pickle_file(data=templates,fname=out_fname)

    return templates

def is_number(t):
    """ check to see if text is a number, from pythoncentral.org """

    try:
        float(t)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(t)
        return True
    except (TypeError, ValueError):
        pass

    return False

def fill_empty_lists(seqs,fill_symbol):
    """ fill empty list in list of lists of symbols
    with the fill symbol. """

    filled_seqs = []
    for seq in seqs:
        if seq == []:
            seq.append(fill_symbol)
        filled_seqs.append(seq)

    return filled_seqs

def remove_repeats_from_seq(seq):
    """ Replace consecutive repeating sybmols with
    empty strings. E.g. if input is ['a','b,','r','r'],
    output is ['a','b','r',''].

    Parameters
    ----------
    seq: list of strings
        a list of chord symbols

    Returns
    -------
    new_seq: list of strings
        a list of chord symbols with consecutive repeating
        symbols removed

    """

    #TODO: rewrite this using find_repeats_in_seq function
    N = len(seq)

    new_seq = N * ['']

    new_seq[0] = seq[0]
    prev = seq[0]
    for i in range(N):
        if seq[i] == prev:
            prev = seq[i]
            continue
        new_seq[i] = seq[i]
        prev = seq[i]

    return new_seq

def find_repeats_in_seq(seq):
    """ Find the indices corresponding to
    consecutive blocks of repeating symbols
    in the sequence.

    Parameters
    ----------
    seq: list of strings
        a list of chord symbols

    Returns
    -------
    idx: list of tuples of ints
        a list of of (start_idx,end_idx) tuples indicating
        the indices of each block of repeating symbols in
        the sequence. Note that these indices are *inclusive*,
        i.e. that the tuple (10,14) indicates that indices 10
        through 14 inclusive have the same symbol.
    """

    N = len(seq)
    idx = []
    prev = seq[0]
    start_idx = 0

    for i in range(N):
        if seq[i] == prev:
            prev = seq[i]
            continue
        prev = seq[i]
        idx.append((start_idx, i-1))
        start_idx = i
    idx.append((start_idx,N-1))
    return idx

def safelog2(x):
    """ safe log, so log(0) -> 0 """
    if type(x) == np.ndarray:
        x = np.where(x<=0.0,1.0,x)
    elif x<=0:
        x = 1.0
    return np.log2(x)

def compute_euclidean_dist(X,Y):
    """
    Compute the Euclidean disance between two matrices X and Y using the specified distance metric.

    args:
        - X (2-D numpy matrix): NxT matrix, where N = number of features and T = number of samples
        - Y (2-D numpy matrix): NxT matrix, where N = number of features and T = number of samples

    returns:
        - dists (numpy vector): distance between corresponding chromagram frames in X and Y
    """

    dists = np.sqrt( np.sum( (X-Y)**2,axis=0 ) )

    return dists


def compute_cosine_dist(X,Y):
    """
    Compute the cosine disance between two matrices X and Y using the specified distance metric.

    args:
        - X (2-D numpy matrix): NxT matrix, where N = number of features and T = number of samples
        - Y (2-D numpy matrix): NxT matrix, where N = number of features and T = number of samples

    returns:
        - dists (numpy vector): distance between corresponding chromagram frames in X and Y
    """
    n = (X*Y).sum(axis=0)
    d = np.sqrt( np.sum( (X)**2,axis=0 ) ) * np.sqrt( np.sum( (Y)**2,axis=0 ) )
    dists = 1.0 - n/d

    return dists


def compute_correlation_dist(X,Y):
    """
    Compute the correlation disance between two matrices X and Y using the specified distance metric.

    args:
        - X (2-D numpy matrix): NxT matrix, where N = number of features and T = number of samples
        - Y (2-D numpy matrix): NxT matrix, where N = number of features and T = number of samples

    returns:
        - dists (numpy vector): distance between corresponding chromagram frames in X and Y
    """

    Xd = X - np.mean(X,axis=0)
    Yd = Y - np.mean(Y,axis=0)
    n = (Xd*Yd).sum(axis=0)
    d = np.sqrt( np.sum( (Xd)**2,axis=0 ) ) * np.sqrt( np.sum( (Yd)**2,axis=0 ) )
    dists = 1.0 - n/d

    return dists

def compute_kl_div(P,Q):
    """ compute the KL divergence between distributions P and Q as:
    KL  = sum( P(i) * log2( P(i) / Q(i) ) )
    """
    Pnz = P + EPS
    Qnz = Q + EPS

    kl = np.sum( Pnz * np.log2( Pnz/Qnz ) )

    return kl

def compute_interval_vector(x):
    """ compute the interval vector for chroma vector x """

    int_vec = np.zeros((6,))
    pcs = x.nonzero()[0]
    n_pcs = pcs.shape[0]

    for i in range(n_pcs):
        for j in range(i,n_pcs):
            if i == j:
                continue
            curr_int = pcs[j] - pcs[i]
            if curr_int > 6:
                curr_int = 12 - curr_int
            int_vec[curr_int-1] += 1

    return int_vec
    
def compute_harmonic_diss(x):
    """ compute the harmonic dissonance of chroma vector x (can be 1x12 or 12x1 or 12,)
    """

    x = x.reshape((12,))
    ivec = compute_interval_vector(x)
    return np.dot(ivec,HARM_DISS_WEIGHTS) / HARM_DISS_NORM


def compute_harmonic_dissonance_vals(X):
    """ compute the harmonic dissonance of each from of chromagram X

    args:
        - X (2-D numpy matrix): NxT matrix, where N = number of features and T = number of samples

    returns:
        - diss_vals (numpy vector): harmonic dissonance of chromagram X (compute for each frame)
    """

    T = X.shape[1]
    diss_vals = np.zeros((T,))

    Xthr = X
    Xthr[Xthr<1e-10] = 0.0

    for t in range(T):
        diss_vals[t] = compute_harmonic_diss(Xthr[:,t])

    return diss_vals

def log_plus(x,y):
    """ Perform addition in the log semiring, i.e.

    Parameters
    ----------
    x: float
        weight
    y: float
        weight        

    Returns
    -------
    xplusy: float
        result of log-semiring addition
    """

    xplusy = -np.log(np.exp(-x) + np.exp(-y))
    return xplusy

def harmonic_mean(values):
    """ Compute the harmonic mean of the data in x.

    Parameters
    ----------
    values: list of floats
        list of values

    Returns
    -------
    hmean: float
        harmonic mean of list of values
    """

    inv_vals = np.array([(x+1e-20)**(-1.0) for x in values],dtype=np.float)
    n = float(len(values))
    hmean = n / inv_vals.sum()

    return hmean




def build_rb2rb_map():
    """ Build dictionary to map real book chord types
    to 36 maj/min/dim triads."""

    rb2rb_36_map = {}

    

    return rb2rb_36_map
