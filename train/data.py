""" Functions/classes etc. for data i/o and manipulation """

import numpy as np

import hamilton.core.data as coredata
import hamilton.core.utils as coreutils
from hamilton.core import fileutils
import hamilton.core.fsm as F
import hamilton.core.utils as CU
import hamilton.train.quantize as Q
import hamilton.train.utils as TU


REST_SYMBOL = 'R'
DISAMBIG_SYMBOL = '#'
MEL_SYM_PREFIX = 'pc'

MEL_QUANT_METHOD_VQ = 0
MEL_QUANT_METHOD_PC = 1

TIME_FORMAT_SEC = 0
TIME_FORMAT_MEAS = 1

CHORD_LABEL_FORMAT_ROMAN = 0
CHORD_LABEL_FORMAT_HARTE = 1

class Dataset(object):
    """ Class to encapusulate melodic and harmonic
    sequence data for a dataset, used for training FSTs.
    """

    def __init__(self,\
        jams_dir=None,\
        codebook=None,\
        melodic_context_len=16,\
        key_normalize=True,
        beats_per_meas=None,
        ignore_filelist=[],\
        time_format=TIME_FORMAT_MEAS,\
        chord_label_format=CHORD_LABEL_FORMAT_ROMAN,\
        chord_label_map_file=None,\
        transpose_to_12_keys=False):

        """ Constructor.

        Parameters
        ----------
        jams_dir: str
            Path to directory containing jams files
        codebook: np.array of floats
            matrix of codebook vectors, dimension k x 12,
            where k is number of vectors. if none, use
            pitch classes as symbols
        melodic_context_len: int
            length of each melodic/harmonic sequence
        key_normalize: boolean
            flag to indicate whether or not to transpose
            melody note values to key of C major
        chord_label_map_file: string
            filepath to a json file mapping chords in the 
            actual dataset to a smaller set of chords 
            (e.g., the 36 major, minor, and diminished triads)
        transpose_to_12_keys: boolean
            transpose the melodic and harmonic sequences to all 12 keys
            NOTE: this assumes that all sequences have been transposed
            to "C" first
        """

        self.jams_dir = jams_dir
        self.codebook = codebook
        self.melodic_context_len = melodic_context_len
        self.key_normalize = key_normalize
        self.beats_per_meas = beats_per_meas
        self.ignore_filelist = ignore_filelist
        self.time_format = time_format
        self.chord_label_format = chord_label_format
        self.chord_label_map_file = chord_label_map_file
        self.chord_label_map = None
        self.transpose_to_12_keys = transpose_to_12_keys

        if not self.chord_label_map_file is None:
            self.chord_label_map =\
                fileutils.read_json_file(self.chord_label_map_file)

        self.songs = []
        if self.beats_per_meas is None:
            self.load_from_jams()
        else:
            self.load_from_jams(beats_per_meas=beats_per_meas)

        if self.key_normalize:
            print 'key normalizing...'
            self.do_key_normalize()

        if not self.chord_label_map is None:
            print 'mapping chords using map file:',self.chord_label_map_file
            self.do_map_chord_labels()

        print 'quantizing...'
        self.do_quantize()

    def load_from_jams(self,\
        beats_per_meas=0.25):

        """ Get melodic and harmonic sequence data for all jams files
        in specified directory.

        Parameters
        ----------
        beats_per_meas: float
            beat subdivision of measure, e.g. 0.25 corresponds to
            4 beats/measure (4/4 time)

        Returns
        -------
        self.songs: list of Song objects
            list of Song objects, one for each jams file in jams
            directory
        """

        print 'reading JAMS from:',self.jams_dir

        jams = TU.import_jams(jams_dir=self.jams_dir,\
            ignore_filelist=self.ignore_filelist)

        print 'reading JAMS for songs...'
        for jam in jams:

            print jam.file_metadata.title
            if jam.note == []:
                continue
            curr_songs = jam_to_song_objects(jam=jam,\
                time_format=self.time_format)
            self.songs.extend(curr_songs)

    def do_key_normalize(self):
        """ Key normalize the melody and harmony """
        for song in self.songs:
            song.key_normalize_melody()
            if self.chord_label_format == CHORD_LABEL_FORMAT_HARTE:
                song.key_normalize_harmony()

    def do_map_chord_labels(self):
        """ Map chord labels to a smaller set using chord label map """

        for song in self.songs:
            chords = song.harmonic_sequence.get_labels()
            mapped_chords = [self.chord_label_map[c] for c in chords]
            song.harmonic_sequence.set_labels(mapped_chords)

    def do_quantize(self):
        """ Quantize the melody and harmony sequences of each song """

        for song in self.songs:
            song.quantize(mel_quantize_func=melody_note_quantize)

    def get_disambiguated_seqs(self):
        """ Build the pairs of disambiguated melody and harmony
        symbol seqeuences.
        """

        io_pairs_dict = self.seqs_to_io_pairs()
        input_seqs,output_seqs = self.make_disambiguated_seqs(io_pairs_dict=io_pairs_dict)

        return input_seqs,output_seqs

    def make_disambiguated_seqs(self,io_pairs_dict):
        """ Add disambiguation symbol to input/output pairs dictionary,
        and produce input and output sequences
        """

        input_seqs = []
        output_seqs = []

        disambig_cnt = 0

        for melody_str,chords in io_pairs_dict.items():
            unique_chords = set(chords)

            if len(unique_chords)>1:
                for chord in unique_chords:
                    melody_seq = melody_str.split()
                    melody_seq.append(DISAMBIG_SYMBOL+str(disambig_cnt))
                    disambig_cnt += 1
                    chord_seq = len(melody_seq) * [F.EPSILON_LABEL]
                    chord_seq[0] = chord
                    input_seqs.append(melody_seq)
                    output_seqs.append(chord_seq)
            else:
                melody_seq = melody_str.split()
                chord_seq = len(melody_seq) * [F.EPSILON_LABEL]
                chord_seq[0] = chords[0]
                input_seqs.append(melody_seq)
                output_seqs.append(chord_seq)

        return input_seqs,output_seqs

    def get_melody_harmony_sequence_pairs(self):
        """ Return all melody/harmony sequence pairs.

        """
        
        melodic_seqs = []
        harmonic_seqs = []

        for song in self.songs:
            mel_seq,harm_seq,_ = song.get_segments()
            melodic_seqs.extend(mel_seq)
            harmonic_seqs.extend(harm_seq)

        return melodic_seqs, harmonic_seqs    

    def seqs_to_io_pairs(self,transpose_to_key=None):
        """ Convert lists of melodic and harmonic sequences to
        a dictionary of input/output pairs.

        Returns
        -------
        io_pairs_dict: dictionary of lists of lists of strings
            dictionary in which each key is a sequences of melodic
            symbols of length melodic_context_len and each value
            is a lists of lists of chord symbols. Each of the
            lists of chord symbols is also of length melodic_context_len.
        transpose_to_key: int
            pitch class of key (0,11) to transpose all melody/harmony pairs to

        """

        melodic_seqs, harmonic_seqs = self.get_melody_harmony_sequence_pairs()

        if self.transpose_to_12_keys:
            print '*** transposing to all 12 keys....'
            melodic_seqs = transpose_melodies_to_all_keys(melodic_seqs)
            harmonic_seqs = transpose_harmonies_to_all_keys(harmonic_seqs)
        elif not transpose_to_key is None:
            tmp_melodic_seqs = transpose_all_melodies_to_key(melodic_seqs=melodic_seqs,to_key=transpose_to_key)
            melodic_seqs = tmp_melodic_seqs

            if not self.chord_label_map is None:
                tmp_harmonic_seqs =\
                    transpose_all_harmonies_to_key(harmonic_seqs=harmonic_seqs,to_key=transpose_to_key)
                harmonic_seqs = tmp_harmonic_seqs

        io_pairs_dict = {}

        for melodic_seq,harmonic_seq in zip(melodic_seqs,harmonic_seqs):

            for i in range(len(melodic_seq)):
                seg = melodic_seq[i:i+self.melodic_context_len]
                mel_slice = ' '.join(seg)

                if mel_slice in io_pairs_dict:
                    io_pairs_dict[mel_slice].append(harmonic_seq)
                else:
                    io_pairs_dict[mel_slice] = [harmonic_seq]

        return io_pairs_dict

    def get_harmonic_sequences(self,repeat_type=None):
        """ Get the harmonic sequences in the dataset

        Parameters
        ----------
        repeat_type: const
            Method to use to handle chord repetitions

        Returns
        -------
        harmonic_seqs: list of lists of strings
            list of harmonic sequences for each song
        """

        harmonic_seqs = []

        for song in self.songs:
            if repeat_type is None:
                harmonic_seqs.append(song.get_harmony_symbols())
            else:
                mel_segs,harm_segs,_ = song.get_segments()
                curr_seq = []
                times = song.harmonic_sequence.find_repeating_symbols()
                for t,chord in zip(times,harm_segs):
                    seglen = int(np.round(t[1] - t[0]))
                    curr_seq.extend(seglen * [chord])
                harmonic_seqs.append(curr_seq)

        if self.transpose_to_12_keys:
            print '*** transposing chord sequences to all 12 keys...'
            harmonic_seqs = transpose_harmonies_to_all_keys(harmonic_seqs)

        return harmonic_seqs

    def repeat_melodic_events(self,beat_quant_level=0.0625):
        """ Repeat melodic events based on beat quantization beat_quant_level
        """

        [s.melodic_sequence.repeat_events(beat_quant_level=beat_quant_level) \
            for s in self.songs]

def transpose_melodies_to_all_keys(melodic_seqs):
    """ transpose list of melody symbol sequences to all 12 keys 

    Parameters
    ----------
    melodic_seqs: list of list of strings
        list of melodic sequences, each of which is a list of strings, each
        with format 'pcn' where n is pitch class

    Returns
    -------
    melodic_seq_transp: list of list of strings
        list of melodic sequences (same format as above), original
        sequences transposed to all 12 keys
    """

    melodic_seq_transp = []

    for curr_melodic_seq in melodic_seqs:
        melodic_seq_transp.append(curr_melodic_seq)
        for key in range(1,12):
            mseq = transpose_melody_to_key(melody=curr_melodic_seq,to_key=key)
            # mseq = [coreutils.transpose_pc_from_c(pc,key) for pc in curr_melodic_seq]
            melodic_seq_transp.append(mseq)

    return melodic_seq_transp

def transpose_all_melodies_to_key(melodic_seqs,to_key):
    """ Transpose all the melodies to specified key.

    Parameters
    ----------
    melodic_seqs: list of list of strings
        list of melodic sequences, each of which is a list of strings, each
        with format 'pcn' where n is pitch class

    Returns
    -------
    melodic_seq_transp: list of list of strings
        list of melodic sequences (same format as above), original
        sequences transposed to specified key
    """

    melodic_seq_transp = []

    for curr_melodic_seq in melodic_seqs:
        mseq = transpose_melody_to_key(melody=curr_melodic_seq,to_key=to_key)
        melodic_seq_transp.append(mseq)

    return melodic_seq_transp

def transpose_melody_to_key(melody,to_key):
    """ Transpose the melody to the specified key_normalize

    Parameters
    ----------
    melody: list of strings
        list of pitch classes of format 'pcn', where n is (0,11)
    to_key: int
        pitch class of destination key (0,11)

    Returns
    -------
    melody_transp: list of strings
        list of transposed pitch classes of format 'pcn', where n is (0,11)

    """

    melody_transp = [coreutils.transpose_pc_from_c(pc,to_key) for pc in melody]
    return melody_transp

def transpose_harmonies_to_keys(harmonic_seqs):
    """ transpose list of melody symbol sequences to all 12 keys 

    Parameters
    ----------
    harmonic_seqs: list of strings or list of list of strings
        list of harmonic sequences, each of which is either a single 
        Harte format chord symbol or a list of Harte chord symbols

    Returns
    -------
    harmonic_seq_transp: list of strings or list of list of strings
        list of harmonic sequences (same format as above), original
        sequences transposed to all 12 keys
    """

    harmonic_seq_transp = []

    for curr_harmonic_seq in harmonic_seqs:
        harmonic_seq_transp.append(curr_harmonic_seq)
        for key in range(1,12):
            hseq = transpose_harmony_to_key(harmony=curr_harmonic_seq,to_key=key)
            # if type(curr_harmonic_seq) == str:
            #     hseq = coreutils.transpose_chord_from_c(curr_harmonic_seq,key)
            # elif type(curr_harmonic_seq) == list:
            #     hseq = [coreutils.transpose_chord_from_c(c,key) for c in curr_harmonic_seq]
            # else:
            #     print 'ERROR: problem transposing chords!'                
            #     return None
            harmonic_seq_transp.append(hseq)

    return harmonic_seq_transp

def transpose_all_harmonies_to_key(harmonic_seqs,to_key):
    """ Transpose all the harmonies to specified key.

    Parameters
    ----------
    harmony_seqs: list of list of strings
        list of harmonic sequences, each of which is a list of strings, each
        in Harte format

    Returns
    -------
    harmony_seqs_transp: list of list of strings
        list of harmonic sequences (same format as above), original
        sequences transposed to specified key
    """

    harmonic_seq_transp = []

    for curr_harmonic_seq in harmonic_seqs:
        hseq = transpose_harmony_to_key(harmony=curr_harmonic_seq,to_key=to_key)
        harmonic_seq_transp.append(hseq)

    return harmonic_seq_transp

def transpose_harmony_to_key(harmony,to_key):
    """ transpose the harmonic sequence to the specified key.

    Parameters
    ----------
    harmony: list of strings or list of list of strings
        list of harmonic sequences, each of which is either a single 
        Harte format chord symbol or a list of Harte chord symbols

    Returns
    -------
    harmony_transp: list of strings or list of list of strings
        list of harmonic sequences (same format as above), original
        sequences transposed to specified key
    """

    if type(harmony) == str:
        harmony_transp = coreutils.transpose_chord_from_c(harmony,to_key)
    elif type(harmony) == list:
        harmony_transp = [coreutils.transpose_chord_from_c(c,to_key) for c in harmony]
    else:
        print 'ERROR: problem transposing chords!'                
        return None

    return harmony_transp

def expand_harmonic_seqs(melodic_seqs,\
    harmonic_seqs,\
    expand_symbol=F.EPSILON_LABEL):
    """ expand each harmonic sequence so it's the
    same length as its corresponding melodic sequence """

    expanded_seqs = []
    for mel_seq,harm_seq in zip(melodic_seqs,harmonic_seqs):
        exp_seq = len(mel_seq) * [expand_symbol]
        exp_seq[0] = harm_seq[0]
        expanded_seqs.append(exp_seq)

    return expanded_seqs

def get_melody_chord_pairs_from_jam(jam,\
    codebook=None,\
    key_normalize=True,\
    beats_per_meas=0.25,\
    melody_sym_prefix='k'):
    """ Given a jam, return the beat-synchronous data
    (melody/harmony pairs).

    Parameters
    ----------
    jam: JAMS object
        the JAMS object to extract the data from
    key_normalize: boolean
        flag to indicate whether or not to transpose
        melody note values to key of C major
    codebook: np.array of floats
        matrix of codebook vectors, dimension k x 12,
        where k is number of vectors. if none, use
        pitch classes as symbols
    beats_per_meas: float
        number of beats per measure, e.g. 0.25 corresponds to
        4 beats/measure (4/4 time)
    melody_sym_prefix: string
        prefix for melody symbols (so each melody symbol will
        be melody_sym_prefix+string(codebook index))

    Returns
    -------
    pairs: triple of lists of melody/chord symbols
        lists of melody and harmony (for *both* harmonic transcriptions)
        symbols for each beat (as specified by beats_per_meas).
    """

    # some songs don't have melody transcriptions
    if jam.note == []:
        return None,None,None

    notes = jam.note[0].data

    duration = jam.file_metadata.duration
    chords = jam.chord[0].data
    note_chunks = TU.get_beat_sync_melodic_chunks(chords=chords,\
        notes=notes,\
        duration=duration,\
        key_normalize=key_normalize,\
        beats_per_meas=beats_per_meas)

    if codebook is None:
        note_syms = []

        for chunk in note_chunks:
            curr_syms = []
            for n in chunk:
                s = str(n.label.value % 12)
                if s == '':
                    s = REST_SYMBOL
                curr_syms.append(s)
            note_syms.append(' '.join(curr_syms))
    else:
        vq_idx = Q.quantize_melody_chunks(note_chunks=note_chunks,\
            codebook=codebook)

        # TODO: make this a function?
        note_syms = []

        for idx in vq_idx.tolist():
            if idx == -1:
                sym = REST_SYMBOL
            else:
                sym = melody_sym_prefix + str(idx)
            note_syms.append(sym)

    # first harmonic transcription...
    chunks = get_beat_sync_harmonic_chunks(chords=chords,\
        duration=duration,\
        beats_per_meas=beats_per_meas)

    # TODO: figure out a solution to when we have more than one
    # chord per beat
    chord_syms0 = [harmonic_chunk_to_symbol(c) for c in chunks]

    chords = jam.chord[1].data
    chunks = get_beat_sync_harmonic_chunks(chords=chords,\
        duration=duration,\
        beats_per_meas=beats_per_meas)

    chord_syms1 = [harmonic_chunk_to_symbol(c) for c in chunks]

    return note_syms,chord_syms0,chord_syms1

def harmonic_chunk_to_symbol(chunk):
    """ convert a harmonic chunk to a symbol

    Parameters
    ----------
    chunk: list of JAMS RangeAnnotations
        list of RangeAnnotations for chords corresponding to a beat

    Returns
    -------
    sym: string
        symbol representing this chunk

    """

    if chunk == []:
        return REST_SYMBOL

    chunk_syms = [c.label.value for c in chunk]
    sym = '_'.join(chunk_syms)
    return sym

def get_beat_sync_harmonic_chunks(chords,duration,beats_per_meas=0.25):
    """ Chunk chords by beat.

    Parameters
    ----------
    chords: JAMS RangeAnnotation (jams "chord[i].data" field)
        chords
    duration: float
        duration of the song in measures
    beats_per_meas: float
        number of beats per measure, e.g. 0.25 corresponds to 4 beats/measure
        (4/4 time)

    Returns
    -------
    chunks: list of list of JAMS RangeAnnotations
        each list of RangeAnnotations contains the chords for
        the corresponding beat, which ranges from 0.0 to duration,
        at intervals specified by beats_per_meas.
    """

    # beats = np.arange(0.0, duration+beats_per_meas, beats_per_meas)
    beats = np.arange(0.0, duration, beats_per_meas)
    n_beats = len(beats)
    chunks = [ [] for _ in range(n_beats) ]

    # TODO: should do this more efficiently....
    for i in range(n_beats-1):
        for chord in chords:
            # if chord.start.value >= beats[i] and \
            # chord.start.value < beats[i+1]:
            #     chunks[i].append(chord)
            if chord.start.value < beats[i+1] and \
            chord.end.value > beats[i]:
                chunks[i].append(chord)
            elif chord.start.value >= beats[i+1]:
                break

    return chunks

def build_key_map(chords,duration,beats_per_meas=0.25):
    """ Build a map of key centers for each beat.

    Parameters
    ----------
    chords: JAMS RangeAnnotation (jams "chord[i].data" field)
        chords
    duration: float
        duration of the song in measures
    beats_per_meas: float
        number of beats per measure, e.g. 0.25 corresponds to 4 beats/measure
        (4/4 time)

    Returns
    -------
    key_map: list of list of ints
        list of lists of integers corresponding to the key center (pitch class)
        at each beat, from 0.0 to duration
        (we used to use a list of list here in the extremely unlikely event that
        there's a key change in the middle of a beat, but this does not occur,
        at least not in the RockCorpus dataset. Makes it much simpler if we
        don't have to deal with key changes during a beat)
    """

    # beats = np.arange(0.0, duration+beats_per_meas, beats_per_meas)
    beats = np.arange(0.0, duration, beats_per_meas)
    n_beats = len(beats)

    # key_map = [ [] for _ in range(n_beats) ]
    key_map = n_beats * [0]

    for i in range(n_beats-1):
        for chord in chords:
            if chord.start.value < beats[i+1] and \
            chord.end.value > beats[i]:
                key_map[i] = chord.label.secondary_value
                # if chord.label.secondary_value not in key_map[i]:
                #     key_map[i].append(dict(key=chord.label.secondary_value,label=chord.label.value))
            elif chord.start.value >= beats[i+1]:
                break

    return key_map

def find_repeats_in_chords(jam_chord):
    """ Find the indices corresponding to
    consecutive blocks of repeating symbols
    in the sequence.

    Parameters
    ----------
    jam_chord: RangeAnnotation
        jam chords object, i.e. jam.chord[i]

    Returns
    -------
    times: list of tuples of floats
        a list of of (start_time,end_time) tuples indicating
        the times of each block of repeating symbols in
        the sequence. Note that these indices are *inclusive*,
        i.e. that the tuple (10,14) indicates that indices 10
        through 14 inclusive have the same symbol.
    """

    # start_times = jam_chord.starts.value
    # end_times = jam_chord.ends.value
    start_times = [s.start.value for s in jam_chord.data]
    end_times = [s.end.value for s in jam_chord.data]
    seq = [s.label.value for s in jam_chord.data]

    # idx = find_repeats_in_seq(seq=jam_chord.labels.value)
    idx = CU.find_repeats_in_seq(seq=seq)
    times = []

    for start_i,end_i in idx:
        start_t = start_times[start_i]
        end_t = end_times[end_i]
        times.append((start_t,end_t))

    return times


def jam_range_to_sequence(range_annot,\
    label_field='value'):
    """ Convert a JAMS RangeAnnotation object to a Sequence object

    Parameters
    ----------
    range_annot: JAMS RangeAnnotation
        JAMS RangeAnnotation object, e.g. jam.note[0]
    label_field: string
        optional label field to get data from
    Returns
    -------
    sequence: core.data.Sequence object
        Sequence object with information from the range annotation           
    """

    start_times = [s.start.value for s in range_annot.data]
    # if not measure_times is None:
    #     start_times = TU.beats_to_times(beat_values=start_times,\
    #         measure_times=measure_times)

    end_times = [s.end.value for s in range_annot.data]
    # if not measure_times is None:
    #     end_times = TU.beats_to_times(beat_values=end_times,\
    #         measure_times=measure_times)

    labels = [getattr(s.label,label_field) for s in range_annot.data]
    sequence = coredata.Sequence(start_times=start_times,\
        end_times=end_times,\
        labels=labels)

    return sequence

def jam_to_song_objects(jam,\
    time_format=TIME_FORMAT_SEC,\
    beat_quant_level=None):
    """ initialize song object(s) from a JAMS object
    Note that there may be more than one harmonic transcription,
    in which case more than one Song will be returned.

    Parameters
    ----------
    jam: JAMS object
        the JAMS object for the song(s)
    time_format: one of {TIME_FORMAT_SEC, TIME_FORMAT_MEAS}
        time format (seconds or measures)

    Returns
    -------
    songs: list of Song objects
        a list of song objects, corresponding to a melody and
        a single harmonic transcription
    """

    songs = []

    # mel_seq = jam_range_to_sequence(range_annot=jam.note[0])

    title = str(jam.file_metadata.title)
    for chord_annot in jam.chord:
        try:
            annotator = str(chord_annot.annotation_metadata.annotator.name)
        except AttributeError:
            annotator = 'unknown'

        for note_annot in jam.note:
            mel_seq = jam_range_to_sequence(range_annot=note_annot)
            mel_seq.type = coredata.SEQUENCE_TYPE_MELODY
            harm_seq = jam_range_to_sequence(range_annot=chord_annot)
            harm_seq.type = coredata.SEQUENCE_TYPE_HARMONY
            key_seq = jam_range_to_sequence(range_annot=chord_annot,\
                label_field='secondary_value')

            song = coredata.Song(title=title,\
                annotator=annotator,\
                melodic_sequence=mel_seq,\
                harmonic_sequence=harm_seq,\
                key_sequence=key_seq)

            if time_format == TIME_FORMAT_SEC:
                # TODO: there may be more than 1 beat annotation
                measure_times = TU.get_measure_times_from_jam(jam=jam)
                # measure_times = [d.time.value for d in jam.beat[0].data]
                song.convert_beats_to_seconds(measure_times=measure_times)
                # n_measures = np.max([d.label.secondary_value for d in jam.beat[0].data])+1
                # measure_times = n_measures*[None]
                # for d in jam.beat[0].data:
                #     meas_num = d.label.secondary_value
                #     t = d.time.value
                #     measure_times[meas_num] = t
            elif not beat_quant_level is None:
                song.melodic_sequence.repeat_events(beat_quant_level=beat_quant_level)
                
            songs.append(song)

    return songs

def melody_note_quantize(note):
    """ function to quantize a melody note

    Parameters
    ----------
    note: int
        MIDI note number (0-127)
    key: int
        pitch class of the key (0-11)

    Returns
    -------
    pc: string
        quantized melody note (prefix + pitch class)
    """
    pc = MEL_SYM_PREFIX + str(Q.midi_note_to_pc(note=note,key=None))
    return pc


def get_iois_from_dataset(dset,precision=4):
    """ Return a set of melody inter-onset intervals (IOIs) from 
    the dataset

    Parameters
    ----------
    dset: Dataset object
        a trained dataset object
    precision: int
        number of places beyond decimal point to
        use for rounding

    Returns
    -------
    ioi: set 
        the set of IOIs
    """

    # ioi = set()
    ioi = {}
    for song in dset.songs:
        start_times = song.melodic_sequence.get_start_times()
        end_times = song.melodic_sequence.get_end_times()
        diffs = [e-s for s,e in zip(start_times,end_times)]
        # diffs = [d - np.trunc(d) for d in diffs]
        # diffs = [np.round(d,precision) - np.trunc(d) for d in diffs]
        # diff_str = ['{0}'.format(d, precision=precision) for d in diffs]      
        # [ioi.add(str(d)) for d in diffs]
        for d in diffs:
            # if d>1.0:
            #     d = 1.0
            d = str(d)
            if d in ioi:
                ioi[d] += 1
            else:
                ioi[d] = 1
    return ioi
