""" Classes and functions useful in representing and manipulating data """

from copy import deepcopy
import numpy as np

import hamilton.core.fsm as fsm
import hamilton.core.utils as coreutils

SEQUENCE_TYPE_MELODY = 0
SEQUENCE_TYPE_HARMONY = 1

MIN_NOTE_DUR = 1e-5
MAX_NOTE_DUR = 1.0
DEFAULT_MIN_NOTE_DUR = 0.25
DEFAULT_MAX_NOTE_DUR = 1.0

class Event(object):
    """ a single event (e.g. chord or melody note) """

    def __init__(self,start_time,end_time,label):
        """ Constructor

        Parameters
        ----------
        start_time: float
            the start time of the event
        end_time: float
            the end time of the event
        label: string or int
            label for the event
        """

        self.start_time = start_time
        self.end_time = end_time
        self.label = label


class Sequence(object):
    """ a sequence of events (e.g. a chord sequence or melody) """

    def __init__(self,labels=[],\
        start_times=None,\
        end_times=None,\
        type=None):

        """ Constructor

        Parameters
        ----------
        label: list of strings or list of ints
            labels for the events
        start_times: list of floats
            the start times of the events
        end_time: list of floats
            the end times of the events
        type: int
            sequence type, either SEQUENCE_TYPE_HARMONY or
            SEQUENCE_TYPE_MELODY
        """

        self.events = []

        if start_times is None:
            start_times = len(labels) * [None]
        if end_times is None:
            end_times = len(labels) * [None]

        for st,et,l in zip(start_times,end_times,labels):
            self.add_event(start_time=st, end_time=et, label=l)

        self.type = type

    def num_events(self):
        """ Get number of events.

        Returns
        -------
        num_events: int
            number of events
        """

        return len(self.events)

    def duration(self):
        """ Get duration of sequence.

        Returns
        -------
        dur: float
            duration of sequence (in whatever time
            units are being used)
        """

        if len(self.events) < 1:
            return 0.0

        return self.events[-1].end_time

    def add_event(self,start_time,end_time,label):
        """
        Add an event to the sequence.

        Parameters
        ----------
        start_time: float
            the start time of the event
        end_time: float
            the end time of the event
        label: string
            label for the event
        """

        e = Event(start_time=start_time,end_time=end_time,label=label)
        self.events.append(e)

    def get_start_times(self):
        """ get a list of all the start times

        Returns
        -------
        start_times: list of floats
            the start times of the events
        """

        return self._get_event_attr_list('start_time')

    def get_end_times(self):
        """ get a list of all the end times

        Returns
        -------
        end_times: list of floats
            the start times of the events
        """

        return self._get_event_attr_list('end_time')


    def get_labels(self):
        """ get a list of all the labels

        Returns
        -------
        labels: list of strings or list of ints
            the labels of the events
        """

        return self._get_event_attr_list('label')

    def get_label_pairs(self,delim=''):
        """ Return a list of pairs of symbols. """

        labels = self.get_labels()
        pairs = []
        for i in range(len(labels)-1):
            p = labels[i] + delim + labels[i+1]
            pairs.append(p)

        return pairs

    def _get_event_attr_list(self,field):
        """ Get a list of the specified Event object attributes 

        field: string
            Event object attribute to get
        """

        vals = [getattr(e,field) for e in self.events]
        return vals

    def set_labels(self,values):
        """ set labels

        Parameters
        ----------
        labels: list of strings or list of ints
            the labels of the events
        """

        self._set_event_attr_list('label',values)

    def set_start_times(self,values):
        """ set start times

        Parameters
        ----------
        start_times: list of floats
            the start times of the events
        """

        self._set_event_attr_list('start_time',values)

    def set_end_times(self,values):
        """ set start times

        Parameters
        ----------
        end_times: list of floats
            the end times of the events
        """

        self._set_event_attr_list('end_time',values)

    def _set_event_attr_list(self,field,values):
        """ Get a list of the specified Event object attributes 

        field: string
            Event object attribute to get
        """

        for evt,val in zip(self.events,values):
            setattr(evt,field,val)

    def find_repeating_symbols(self):
        """ Find the indices corresponding to
        consecutive blocks of repeating symbols
        in the sequence.

        Returns
        -------
        times: list of tuples of floats
            a list of of (start_time,end_time) tuples indicating
            the times of each block of repeating symbols in
            the sequence. Note that these indices are *inclusive*,
            i.e. that the tuple (10,14) indicates that indices 10
            through 14 inclusive have the same symbol.
        """

        start_times = self.get_start_times()
        end_times = self.get_end_times()
        labels = self.get_labels()

        # idx = find_repeats_in_seq(seq=jam_chord.labels.value)
        idx = coreutils.find_repeats_in_seq(seq=labels)
        times = []

        for start_i,end_i in idx:
            start_t = start_times[start_i]
            end_t = end_times[end_i]
            times.append((start_t,end_t))

        return times

    def remove_symbols(self,symbol_to_remove):
        new_events = [e for e in self.events if not e.label == symbol_to_remove]
        # del self.events
        self.events = new_events

    def get_compressed_labels(self):
        """ Return the labels with consecutive repeating symbols removed.
        e.g. if labels = ['A','B','B','C','C','A','A'], 
        return ['A','B','C','A']
        """

        labels = coreutils.remove_repeats_from_seq(self.get_labels())
        labels = [label for label in labels if not label == '']
        return labels

    def repeat_events(self,beat_quant_level=0.0625):
        """ Repeat each event given a quantization level, so that, for example
        an event lasting a quarter note (0.25) will be repeated 4 times, given 
        a quantization level of a 16th note (0.0625)

        Parameters
        ----------
        beat_quant_level: float
            quantization level, where 0.25 == quarter note, etc.
        """

        new_events = []

        for orig_evt in self.events:
            dur = float(orig_evt.end_time - orig_evt.start_time)
            if dur <= beat_quant_level:
                new_events.append(orig_evt)
                continue

            n_repeats = int(np.round(dur/beat_quant_level))

            for n in range(n_repeats):
                st = orig_evt.start_time + n*beat_quant_level
                et = min(orig_evt.end_time, orig_evt.start_time + (n+1)*beat_quant_level)
                new_evt = Event(start_time=st,end_time=et,label=orig_evt.label)
                new_events.append(new_evt)

        self.events = new_events

    def get_num_repeat_beats(self,beat_quant_level=0.0625):
        """ For each event, get number of repeats given a quantization level.
        This is used to rhythmically expand melodic sequences, so, for example,
        a melody note of duration quarter note will be repeated 4 times if quantization
        level is 0.0625 (i.e., a 16th note). Round the number of repeats, so 
        any duration less than 1/2 the beat_quant_level is ignored.

        Parameters
        ----------
        beat_quant_level: float
            quantization level, where 0.25 == quarter note, etc.

        """

        durs = np.array(self.get_end_times()) - np.array(self.get_start_times())
        n_repeats = np.round(durs/beat_quant_level)
        n_repeats = n_repeats.astype(dtype=np.int)

        return n_repeats

    #TODO: change this method so that it's non-destructive, probably be adding 
    #fields to Event object (e.g., start_times/end_times and start_beats/end_beats
    #or something like that)
    def convert_beats_to_seconds(self,measure_times):
        """ Convert the start and end times from measures to seconds 

        NOTE: this is a destructive operation (i.e., it converts all times from
        measures to seconds instead of adding additional information)

        Parameters
        ----------
        measure_times: list of floats
            list of the starting time of each measure. 
            NOTE that for some reason, in the Rock Corpus Dataset, measures are
            counted starting at measure 1 in the beats annotations, but starting
            at measure 0.0 in the chord and melody annotations. Here, we assume
            that 0-indexing is used, to make it more compatible with the other
            annotations

        """

        start_beats = self.get_start_times()
        start_times = coreutils.beats_to_times(beat_values=start_beats,\
            measure_times=measure_times)

        end_beats = self.get_end_times()
        end_times = coreutils.beats_to_times(beat_values=end_beats,\
            measure_times=measure_times)

        for start_t,end_t,event in zip(start_times,end_times,self.events):
            event.start_time = start_t
            event.end_time = end_t

    def to_chromagram(self,label_map=None,
        time_res=0.0232,
        time_format='sec',
        duration=None):
        """ Generate a chromagram from the data in this sequence 
        using the provided map to convert melody symbols or chord labels,
        for example, to pitch classes.

        """

        # TODO: implement for beat time format
        if not time_format == 'sec':
            print 'UNIMPLEMENTED!'
            return None

        if duration is None:
            duration = self.duration()
        N = np.ceil(duration/time_res)

        cg = np.zeros((12,N))

        # add chords to chromagram
        for event in self.events:
            start_i = np.round(event.start_time / time_res)
            end_i = np.round(event.end_time / time_res)

            if self.type == SEQUENCE_TYPE_HARMONY:
                c = event.label
                if not label_map is None:
                    c = label_map[c]
                vec = coreutils.chord_symbol_to_pitch_vector(chord=c)
            elif self.type == SEQUENCE_TYPE_MELODY:
                p = event.label
                vec = coreutils.melody_symbol_to_pitch_vector(note=p,prefix='pc')
            else:
                print 'ERROR: specify a valid sequence type'
                return None

            cg[:,start_i:end_i] = vec.reshape((12,1))

        return cg
        


class Song(object):
    """ Class to encapsulate the data for an individual song. """

    def __init__(self,title,\
        annotator,\
        melodic_sequence,\
        harmonic_sequence,\
        key_sequence=None):
        """ Constructor.

        Parameters
        ----------
        title: string
            song title
        annotator: string
            name of annotator
        melodic_sequence: Sequence object
            Sequence object representing melody of song
        harmonic_sequence: Sequence object
            Sequence object representing harmony of song
        key_sequence: Sequence object
            Sequence object representing key(s) of song
        """

        self.title = title
        self.annotator = annotator
        self.melodic_sequence = melodic_sequence
        self.harmonic_sequence = harmonic_sequence
        self.key_sequence = key_sequence

    def get_melody_times(self):
        """ Get the onset time of each note in the melody

        Returns
        -------
        times: list of floats
            list of note onset times
        """

        return self.melodic_sequence.get_start_times()

    def get_melody_symbols(self):
        """ Get the symobls corresponding to the melodic sequence.

        Returns
        -------
        mel_seq: list of strings
            list of melody labels (e.g. pitch classes)
        """

        return self.melodic_sequence.get_labels()

    def get_harmony_symbols(self):
        """ Get the symobls corresponding to the harmonic sequence.

        Returns
        -------
        harm_seq: list of strings
            list of harmony labels (e.g. chord symbols)
        """

        return self.harmonic_sequence.get_labels()

    def key_normalize_harmony(self):
        """ Key normalize the harmonic sequence """

        _,chords,_ = self.get_segments()
        kn_chords = []

        for key,chord in zip(self.key_sequence.get_labels(),chords):
            transp_c = coreutils.transpose_chord_to_c(chord=chord,from_key=key)
            kn_chords.append(transp_c)

        self.harmonic_sequence.set_labels(kn_chords)

    def key_normalize_melody(self):
        """ Key normalize the melody sequence """

        mel_segs,keys,_ = self.get_segments(target_sequence=self.key_sequence)

        mel_kn = []
        for key,mel_seg in zip(keys,mel_segs):
            mel_seg_kn = [coreutils.transpose_midi_note_to_c(note=int(m),key=int(key)) \
                for m in mel_seg]
            mel_kn.extend(mel_seg_kn)

        for m,m_kn in zip(self.melodic_sequence.events,mel_kn):
            m.label = m_kn

    def quantize(self,mel_quantize_func=None,\
        harm_quantize_func=None):
        """ Quantize the symbols in the melody
        and harmony sequences.

        Parameters
        ----------
        mel_quantize_func: function
            function to quantize the melody symbols
            function should take string as an argument,
            and return a string
        harm_quantize_func: function
            function to quantize the harmony symbols
            function should take string as an argument,
            and return a string
        """

        if mel_quantize_func is not None:
            for e in self.melodic_sequence.events:
                e.label = mel_quantize_func(e.label)

        if harm_quantize_func is not None:
            for e in self.harmonic_sequence.events:
                e.label = harm_quantize_func(str(e.label))

    def get_segments(self,target_sequence=None):
        """ segment melody and harmony into sub-sequences
        that occur over a particular harmony.
        """

        # segment_times = self.harmonic_sequence.find_repeating_symbols()

        # chord_starts = self.harmonic_sequence.get_start_times()
        # chords = self.harmonic_sequence.get_labels()

        if target_sequence is None:
            target_sequence = self.harmonic_sequence

        segment_times = target_sequence.find_repeating_symbols()

        chord_starts = target_sequence.get_start_times()
        chords = target_sequence.get_labels()

        notes = self.melodic_sequence.events

        melody_segs = []
        harmony_segs = []
        weight_segs = []
        for start_t,end_t in segment_times:
            i = chord_starts.index(start_t)
            # key_pc = int(keys[i])

            mel_seg = []
            wt_seg = []
            for note in notes:
                if note.end_time < start_t:
                    continue
                if note.start_time >= end_t:
                    break

                # add weights based on note duration
                delta_t = note.end_time - note.start_time
                if delta_t < MIN_NOTE_DUR:
                    delta_t = DEFAULT_MIN_NOTE_DUR
                elif delta_t > MAX_NOTE_DUR:
                    delta_t = DEFAULT_MAX_NOTE_DUR

                wt = -np.log(delta_t)
                wt_seg.append(wt)

                mel_seg.append(str(note.label))
            if mel_seg == []:
                continue

            melody_segs.append(mel_seg)
            harmony_segs.append(str(chords[i]))
            weight_segs.append(wt_seg)

        return melody_segs,harmony_segs,weight_segs

    def get_equal_length_sequences(self):
        """ Get the melody and harmony sequences as
        equal length lists of symbols """

        isegs,osegs,_ = self.get_segments()

        input_seq = []
        output_seq = []

        for iseg,oseg in zip(isegs,osegs):
            N = len(iseg)
            input_seq.extend(iseg)
            output_seq.extend(N*[oseg])

        return input_seq,output_seq

    def add_timing_info_to_harmonic_sequence(self,boundary_symbol=fsm.EPSILON_LABEL):
        """ Add timing information to harmonic sequence
        (i.e., start and end times)

        Parameters
        ----------
        melodic_sequence: core.data.Sequence object
            Sequence object representing melody
        harmonic_sequence: core.data.Sequence object
            Sequence object representing harmony

        Returns
        -------
        harmonic_sequence: core.data.Sequence object
            Sequence object representing harmony, modifed to include
            timing information

        """
        assert self.melodic_sequence.num_events() == self.harmonic_sequence.num_events()

        start_i = 0
        for i,mel_evt in enumerate(self.melodic_sequence.events):
            harm_evt = self.harmonic_sequence.events[i]
            if mel_evt.label == boundary_symbol:
                harm_evt.start_time = self.melodic_sequence.events[start_i].start_time
                harm_evt.end_time = self.melodic_sequence.events[i-1].end_time
                start_i = i+1
            elif not harm_evt.label == boundary_symbol:
                harm_evt.start_time = self.melodic_sequence.events[start_i].start_time
                harm_evt.end_time = self.melodic_sequence.events[i].end_time
                start_i = i


    def remove_epsilons(self):
        """ remove epsilons in melody and harmony sequences """

        self.melodic_sequence.remove_symbols(symbol_to_remove=fsm.EPSILON_LABEL)
        self.harmonic_sequence.remove_symbols(symbol_to_remove=fsm.EPSILON_LABEL)


    #TODO: change this method so that it's non-destructive, probably by adding 
    # fields to Event object (e.g., start_times/end_times and start_beats/end_beats
    # or something like that)
    def convert_beats_to_seconds(self,measure_times):
        """ Convert the start and end times from measures to seconds 

        NOTE: this is a destructive operation (i.e., it converts all times from
        measures to seconds instead of adding additional information)

        Parameters
        ----------
        measure_times: list of floats
            list of the starting time of each measure. 
            NOTE that for some reason, in the Rock Corpus Dataset, measures are
            counted starting at measure 1 in the beats annotations, but starting
            at measure 0.0 in the chord and melody annotations. Here, we assume
            that 0-indexing is used, to make it more compatible with the other
            annotations

        """

        self.melodic_sequence.convert_beats_to_seconds(measure_times)
        self.harmonic_sequence.convert_beats_to_seconds(measure_times)

    def get_combined_chromagram(self,label_map=None,\
        time_res=0.0232,\
        time_format='sec',\
        duration=None):
        """ Generate a chromagram from the data in this sequence 
        using the provided map to convert melody symbols or chord labels,
        for example, to pitch classes.

        """
        # TODO: implement for beat time format
        if not time_format == 'sec':
            print 'UNIMPLEMENTED!'
            return None

        dur = max(self.melodic_sequence.duration(),self.harmonic_sequence.duration())

        cg_mel = self.melodic_sequence.to_chromagram(label_map=label_map,\
            time_res=time_res,
            time_format=time_format,
            duration=dur)

        cg_harm = self.harmonic_sequence.to_chromagram(label_map=label_map,\
            time_res=time_res,
            time_format=time_format,
            duration=dur)

        cg = cg_mel + cg_harm

        return cg

    def get_duration(self):
        """ Return the duration of the song, in whatever time units
        are currently used (i.e. beats or seconds) 

        Returns
        -------
        dur: float
            duration in either seconds or beats
        """

        dur = max(self.harmonic_sequence.duration(),\
            self.melodic_sequence.duration())

        return dur

def transpose_melodic_sequence_to_key(melodic_seq,to_key):
    """ Transpose all pitch classes in melodic sequence to the
    specified key

    Parameters
    ----------
    melodic_seq: core.data.Sequence object
        Sequence object representing melody
    to_key: int
        pitch class of target key

    Returns
    -------
    transp_melodic_seq: core.data.Sequence object
        Sequence object representing transposed melody
    """

    transp_melodic_seq = deepcopy(melodic_seq)
    notes = [coreutils.transpose_pc_from_c(pc=pc,to_key=to_key) for\
        pc in transp_melodic_seq.get_labels()]
    transp_melodic_seq.set_labels(notes)

    return transp_melodic_seq

def build_song_with_timing(melodic_sequence,\
    harmonic_sequence):
    """ Build a song with timing information from the melody
    added to the harmony, and with all epsilons removed. 
    This is used to create a Song object from the output
    of an FST
    """

    song = Song(title='tmp',annotator='tmp',\
        melodic_sequence=melodic_sequence,\
        harmonic_sequence=harmonic_sequence)
    song.add_timing_info_to_harmonic_sequence()
    song.remove_epsilons()

    return song


def build_sequence(labels,\
    start_times,\
    end_times,\
    ignore_label,\
    default_time=-1.0):
    """
    Build a Sequence object using the labels, and start and end times.
    Only add start and end times to events whose label does not
    equal the ignore label (these events have start and end times set to
    the default time)

    Parameters
    ----------
    labels: list of strings
        list of labels
    start_times: list of floats
        list of start times
    end_times: list of floats
        list of end times
    ignore_label: string
        label to ignore
    default_time: float
        value to use as default start/end time
    """

    # assert len(labels) == len(start_times) == len(end_times)

    sequence = Sequence(labels=[])
    i = 0
    for label in labels:
        if label == ignore_label:
            st = default_time
            et = default_time
        else:
            st = start_times[i]
            et = end_times[i]
            i += 1
        sequence.add_event(start_time=st,\
            end_time=et, \
            label=label)

    return sequence

def song_from_fst(melodic_sequence,harmonic_fst):
    """ Construct a Song object given a melody_sequence and a path FST,
    which contains the harmonic sequence information 
    """

    harmonic_fst.load_from_compiled()
    in_seq,out_seq = harmonic_fst.get_input_output_syms()

    tmp_melseq = build_sequence(labels=in_seq,\
        start_times=melodic_sequence.get_start_times(),\
        end_times=melodic_sequence.get_end_times(),\
        ignore_label=fsm.EPSILON_LABEL)

    tmp_harmseq = build_sequence(labels=out_seq,\
        start_times=melodic_sequence.num_events()*[None],\
        end_times=melodic_sequence.num_events()*[None],\
        ignore_label=fsm.EPSILON_LABEL)

    tmp_melseq.type = SEQUENCE_TYPE_MELODY
    tmp_harmseq.type = SEQUENCE_TYPE_HARMONY

    song = build_song_with_timing(melodic_sequence=tmp_melseq,\
        harmonic_sequence=tmp_harmseq)

    return song

def sequences_equal(sequence1, sequence2):
    """ Compare the start times, end times, and labels of
    two sequences.

    Parameters
    ----------
    sequence1: Sequence object
    sequence2: Sequence object

    Returns
    -------
    are_equal: Boolean
        true/false, depending on whether sequences
        are equal or not
    """

    are_equal = (np.array_equal(sequence1.get_start_times(),\
        sequence2.get_start_times()))\
        and (np.array_equal(sequence1.get_end_times(),\
        sequence2.get_end_times()))\
        and (sequence1.get_labels() == sequence2.get_labels())

    return are_equal

