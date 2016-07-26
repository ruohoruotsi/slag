import hamilton.core.fileutils as U
from hamilton.core import fsm
from hamilton.core import utils as coreutils
from hamilton.core import data as coredata
from hamilton.generate import chordgen

reload(coredata)
reload(coreutils)
reload(chordgen)

import argparse
import csv
import multiprocessing
import numpy as np

try:
    import librosa
except ImportError:
    print 'ERROR: unable to import librosa'

try:
    import pretty_midi
except ImportError:
    print 'ERROR: unable to import pretty_midi'

try:
    import vamp
except ImportError:
    print 'ERROR: unable to import vamp (Python)'

import numpy as np
import os
from random import choice
import sys


# FST stuff
USE_BQ = False #True
BEAT_QUANT_LEVEL = 0.125

BUILD_2ND_ORDER_ERGODIC = False

FST_ROOT_DIR = os.path.join('/Users',os.getlogin(),'Dropbox','work','src','hamilton','generate','fst')

INPUT_SYMS_FILE = os.path.join(FST_ROOT_DIR,'input.syms')
OUTPUT_SYMS_FILE = os.path.join(FST_ROOT_DIR,'output.syms')
LOG_FST_FILE = os.path.join(FST_ROOT_DIR,'LoG.fst')

BQ_INPUT_SYMS_FILE = os.path.join(FST_ROOT_DIR,'isyms_bq0125_L14.syms')
BQ_OUTPUT_SYMS_FILE = os.path.join(FST_ROOT_DIR,'osyms_bq0125_L14.syms')
BQ_LOG_FST_FILE = os.path.join(FST_ROOT_DIR,'LoG_bq0125_L14.fst')


#Example chord sequence for input into generate_chord_midi
TEST_SEQUENCE = [(0, 0.382, 'C:maj'),
                 (0.382, 12.391, 'G:maj'),
                 (13.0, 16.365, 'A:min7'),
                 (16.365, 20.382, 'C:maj')]

VOICING_FILE = os.path.join(os.path.dirname(__file__),\
    'chord_voicings.json')
ROMAN_2_CHORD_FILE = os.path.join(os.path.dirname(__file__),\
    '..','roman2labelmap.json')

PYIN_CONFIG_FILE = os.path.join(os.path.dirname(__file__),\
    'pyin_notes.n3')

TRANSCRIPT_PLUGIN_SLIVET = 'silvet:silvet'
TRANSCRIPT_PLUGIN_QM = 'qm-vamp-plugins:qm-transcription'



def make_examples(audio_fname='output.wav'):
    """ Make some audio examples """

    # start_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 11.0,12.0, 13.0, 14.0]
    # durs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    # notes = ['pc7', 'pc7', 'pc2', 'pc2', 'pc4', 'pc4', 'pc2', 'pc0', 'pc0', 'pc11', 'pc11', 'pc9', 'pc9', 'pc7']

    # start_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,12.0, 13.0, 14.0]
    # durs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    # notes = ['pc2', 'pc5', 'pc9', 'pc2', 'pc11', 'pc11', 'pc2', 'pc2', 'pc5', 'pc5', 'pc7', 'pc4', 'pc2', 'pc4','pc0']



    
    beat_quant_level = None
    is_1st_order_erg = False
    BUILD_2ND_ORDER_ERGODIC = False
    chordgen.BUILD_2ND_ORDER_ERGODIC = False

    # fst_dir = '/Users/jon/Dropbox/work/src/hamilton/train/models'
    fst_dir = os.path.join('/Users',os.getlogin(),'Dropbox/work/src/hamilton/train/models')

    # fst_dir = os.path.join(fst_dir,'ergodic1')
    # fst_fn = os.path.join(fst_dir,'LoG_erg1.fst')
    # isyms_fn = os.path.join(fst_dir,'isyms_erg1.syms')
    # osyms_fn = os.path.join(fst_dir,'osyms_erg1.syms')
    # is_1st_order_erg = True

    # fst_dir = os.path.join(fst_dir,'ergodic2')
    # fst_fn = os.path.join(fst_dir,'LoG_erg2.fst')
    # isyms_fn = os.path.join(fst_dir,'isyms_erg2.syms')
    # osyms_fn = os.path.join(fst_dir,'osyms_erg2.syms')
    # BUILD_2ND_ORDER_ERGODIC = True
    # chordgen.BUILD_2ND_ORDER_ERGODIC = True

    # fst_dir = os.path.join(fst_dir,'sequence')
    # fst_fn = os.path.join(fst_dir,'LoG_L14_N3.fst')
    # isyms_fn = os.path.join(fst_dir,'isyms_L14.syms')
    # osyms_fn = os.path.join(fst_dir,'osyms_L14.syms')

    fst_dir = os.path.join(fst_dir,'sequence_bq125')
    fst_fn = os.path.join(fst_dir,'LoG_bq0125_L14.fst')
    isyms_fn = os.path.join(fst_dir,'isyms_bq0125_L14.syms')
    osyms_fn = os.path.join(fst_dir,'osyms_bq0125_L14.syms')
    beat_quant_level = 0.125

    # fst_dir = os.path.join(fst_dir,'sequence_bq05')
    # fst_fn = os.path.join(fst_dir,'LoG_bq05_L14.fst')
    # isyms_fn = os.path.join(fst_dir,'isyms_bq05_L14.syms')
    # osyms_fn = os.path.join(fst_dir,'osyms_bq05_L14.syms')
    # beat_quant_level = 0.5

    isyms = fsm.SymbolTable(filename=isyms_fn)
    osyms = fsm.SymbolTable(filename=osyms_fn)
    fst = fsm.FST(filename=fst_fn,isyms_table=isyms,osyms_table=osyms)

    method = chordgen.GEN_METHOD_SHORTEST
    print 'Using shortest'

    # method = chordgen.GEN_METHOD_LOG_PROB
    # print 'Using log prob'

    # method = chordgen.GEN_METHOD_UNIFORM
    # print 'Using uniform'

    build_dir = './tmp'
    # generator = ChordGenerator(fst=fst,method=chordgen.GEN_METHOD_SHORTEST,roman_to_chords=True,\
    #     build_dir=build_dir)

    # twinkle in C
    # start_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 11.0,12.0, 13.0, 14.0]
    # durs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    # notes = ['pc0', 'pc0', 'pc7', 'pc7', 'pc9', 'pc9', 'pc7', 'pc5', 'pc5', 'pc4', 'pc4', 'pc2', 'pc2', 'pc0']

    # my romance
    # start_times = [0.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 11.0, 11.5, 12.0]
    # durs = [3.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 3.0, 0.5, 0.5, 4.0] 
    # notes = ['pc7', 'pc4', 'pc5', 'pc7', 'pc9', 'pc11', 'pc0', 'pc0', 'pc11', 'pc9', 'pc7']

    # start_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0] #, 16.0, 17.0]
    # durs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #, 1.0, 1.0]
    # notes = ['pc2', 'pc4', 'pc5', 'pc7', 'pc2', 'pc7', 'pc5', 'pc9', 'pc11', 'pc7', 'pc7', 'pc4', 'pc5', 'pc7', 'pc11', 'pc4'] #, 'pc0','pc0']


    # "altdeu" example (same as stimulus example in Ch. 4)
    # start_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,\
    #     4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 7.75,\
    #     8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0,\
    #     12.0, 12.5, 13.0, 14.0, 14.5, 15.0, 15.5,\
    #     16.0, 16.5, 17.0, 17.50, 17.75, 18.0, 18.5, 19.0, 19.5,\
    #     20.0]
    
    # durs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,\
    #     0.5, 0.5, 0.5, 0.5, 1.0, 0.75, 0.25,\
    #     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0,\
    #     0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5,\
    #     0.5, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5,\
    #     4.0]

    # notes = ['pc0', 'pc4', 'pc2', 'pc11', 'pc0', 'pc9', 'pc7', 'pc7',\
    #     'pc9', 'pc0', 'pc11', 'pc2', 'pc0', 'pc7', 'pc5',\
    #     'pc4', 'pc4', 'pc4', 'pc4', 'pc4', 'pc7', 'pc7',\
    #     'pc4', 'pc7', 'pc7', 'pc7', 'pc5', 'pc5', 'pc9',\
    #     'pc9', 'pc7', 'pc4', 'pc2', 'pc2', 'pc4', 'pc7', 'pc5', 'pc2',\
    #     'pc0']


    # hopefully good example???
    start_times = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 7.0, 7.5,\
        8.0, 8.5, 9.0, 10.0, 11.0, 12.0, 13.0, 13.5, 14.0, 15.0, 15.5, 16.0]
    durs = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, \
        0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 2.0]
    notes = ['pc11', 'pc9', 'pc7', 'pc5', 'pc4', 'pc5', 'pc2', 'pc11', 'pc0', 'pc4', 'pc5',\
        'pc9', 'pc7', 'pc5', 'pc4', 'pc2', 'pc11', 'pc2', 'pc11', 'pc0', 'pc4', 'pc0', 'pc0']

    assert( len(start_times) == len(durs) == len(notes) )

    end_times = np.array(start_times) + np.array(durs)
    end_times = end_times.tolist()

    melody_sequence = coredata.Sequence(labels=notes,start_times=start_times,end_times=end_times)
    if BUILD_2ND_ORDER_ERGODIC:
        print 'Using 2nd order ergodic'
        notes2 = []
        start_times2 = []
        end_times2 = []

        # start_times2 = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        # end_times2 =   [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 15.0]
        # notes2 = ['pc2pc4', 'pc5pc7', 'pc2pc7', 'pc5pc9', 'pc11pc7', 'pc7pc4', 'pc5pc7', 'pc11pc4']


        for i in range(0,len(notes)-1,2):
            tmp = notes[i] + notes[i+1]
            notes2.append(tmp)
            start_times2.append(start_times[i])
            end_times2.append(start_times[i] + durs[i] + durs[i+1])


        # print notes2
        # print start_times2
        # print end_times2

        # start_times2 = [0.0, 2.0, 4.0, 6.0, 9.0, 11.0, 13.0]
        # end_times2 =   [2.0, 4.0, 6.0, 9.0, 11.0, 13.0, 16.0]
        # notes2 = ['pc0pc0', 'pc7pc7', 'pc9pc9', 'pc7pc5', 'pc5pc4', 'pc4pc2', 'pc2pc0']

        # mel_labels = melody_sequence.get_label_pairs()
        # print mel_labels
        # n_start_times = start_times[0:-1]
        # n_end_times = end_times[1:]
        # del melody_sequence
        melody_sequence = coredata.Sequence(labels=notes,start_times=start_times2,end_times=end_times2)
    elif is_1st_order_erg:
        print 'Using 1st order ergodic'
        # start_times2 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 78.0, 9.0, 10.0, 11.0,12.0, 13.0, 14.0]
        start_times2 = start_times
        end_times2 =   end_times
        melody_sequence = coredata.Sequence(labels=notes,start_times=start_times2,end_times=end_times2)
    else:
        mel_labels = melody_sequence.get_labels()

    mel_data = zip(start_times,end_times,notes)    

    # if not beat_quant_level is None:
    #     print 'Using beat quant level:',beat_quant_level
    #     melody_sequence.repeat_events(beat_quant_level=beat_quant_level)

    harm_seq = chordgen.get_harmonic_seq(melody_sequence=melody_sequence,\
        full_fst=fst,\
        method=method,\
            build_dir=build_dir,\
            return_path_weight=False)

    # print harm_seq.get_labels()
    # print harm_seq.get_start_times()
    # print harm_seq.get_end_times()

    if BUILD_2ND_ORDER_ERGODIC:
        print 'before: st',start_times2
        print 'before: et',end_times2
        # harm_seq.set_start_times(start_times2)
        # harm_seq.set_end_times(end_times2)

    # if BUILD_2ND_ORDER_ERGODIC:
    #     harm_seq.set_start_times([0.0, 2.0, 4.0, 6.0, 9.0, 11.0, 13.0])
    #     harm_seq.set_end_times([2.0, 4.0, 6.0, 9.0, 11.0, 13.0, 16.0])
    # elif is_1st_order_erg:
    #     harm_seq.set_start_times(start_times)
    #     harm_seq.set_end_times(end_times)

    # if BUILD_2ND_ORDER_ERGODIC:
    #     harm_seq.set_start_times([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    #     harm_seq.set_end_times([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    # elif is_1st_order_erg:
    #     harm_seq.set_start_times(start_times)
    #     harm_seq.set_end_times(end_times)

    print harm_seq.get_start_times()
    print harm_seq.get_end_times()
    print harm_seq.get_labels()


    mel_data = zip(start_times,end_times,notes)    

    # write file
    print 'writing audio file:',audio_fname

    fs = 22050
    if BUILD_2ND_ORDER_ERGODIC or is_1st_order_erg:
        chord_data = zip(start_times2,end_times2,harm_seq.get_labels()+['C:maj'])
    else:
        chord_data = zip(harm_seq.get_start_times(),harm_seq.get_end_times(),harm_seq.get_labels())
    # mel_data = zip(melody_sequence.get_start_times(),melody_sequence.get_end_times(),melody_sequence.get_labels())    
    midi_chords = generate_chord_midi(chord_sequence=chord_data,melody_sequence=mel_data)
    y = midi_chords.fluidsynth(fs=fs)
    librosa.output.write_wav(audio_fname,y,fs)

class AudioAnalyzer(object):
    """ Class to analyze audio """

    def __init__(self,nfft=1024,\
        hop=512):

        self.nfft = nfft
        self.hop = hop

    # def detect_melody(self,y,fs,med_filt_len=31):
    #     chroma = librosa.feature.chroma_cqt(y=y,\
    #         sr=fs,\
    #         hop_length=self.hop)
    #     pcs = np.argmax(chroma,axis=0)
    #     pcs_filt = scipy.signal.medfilt(pcs,med_filt_len)
    #     return pcs_filt

    def get_estimated_melody_sequence_pyin(self,audio_fname):
        melody_sequence = get_pc_data_pyin_notes(input_audio_fname=audio_fname)
        return melody_sequence

    def get_estimated_melody_sequence(self,y,fs):
        # print 'retain only harmonic portion of signal...'
        # y = librosa.effects.harmonic(y)

        # print 'estimate tuning...'
        # tuning = librosa.estimate_tuning(y=y, sr=fs)
        # print 'adjust tuning...'
        # y_tuned = librosa.effects.pitch_shift(y, fs, -tuning)

        print 'estimate melody...'
        # melody_sequence = get_midi_data_slivet(y,fs)
        melody_sequence = get_pc_data_aubio(y,fs)
        return melody_sequence


class NoteGenerator(object):
    """ Class to generate notes """

    def __init__(self):
        pass

    def synthesize_to_file(self,output_audio_fname,\
        melody_sequence,\
        fs=22050,\
        octave_offset=0,\
        tuning_offset=None):
        """ Synthesize an audio file containing
        accompaniment and original melody audio, if 
        available 

        Parameters
        ----------
        output_audio_fname: string
            Full path/filename of output audio file
        melody_sequence: core.data.Sequence object
            Sequence object representing melody        
        fs: int
            sample rate
        """

        mel_seq = [(e.start_time,e.end_time,e.label) for e in melody_sequence.events]
        midi = generate_note_midi(note_sequence=mel_seq,octave_offset=octave_offset)
        y = midi.fluidsynth(fs=fs)
        librosa.output.write_wav(output_audio_fname,y,fs)


class ChordGenerator(object):
    """ Class to generate chords given melody """

    def __init__(self,fst,\
        method,\
        roman_to_chords=False,\
        build_dir='.',\
        voicing_file=None):
        """ Constructor

        Parameters
        ----------
        fst: FST object
            The fst used to generate output
        method: string (GEN_METHOD_SHORTEST, GEN_METHOD_LOG_PROB, 
            GEN_METHOD_UNIFORM)
            generation method to use
        """

        self.fst = fst
        self.method = method
        self.roman_to_chords = roman_to_chords
        self.roman_to_chord_map = None
        self.build_dir = build_dir

        if self.roman_to_chords:
            self.roman_to_chord_map = get_roman2chord_map()

        if voicing_file is None:
            self.voicings = None
        else:
            self.voicings = get_all_voicings(VOICING_FILE)

    def estimate_key(self,melodic_sequence):
        """ Estimate the key from the melodic sequence
        by finding the transposition that yields the 
        lowest weight accompaniment.

        Parameters
        ----------
        melodic_seq: core.data.Sequence object
            Sequence object representing melody

        Returns
        -------
        key: int
            pitch class of estimated key
        """

        key = 0

        return key

    def find_min_weight_transposition(self,melody_sequence):
        """ Find the minimum weight transposition by transposing melody 
        to all 12 keys, and determining which transposition produces
        the lowest total path weight (using shortest path decoding)
        Uses multiple processes.

        Parameters
        ----------
        melody_sequence: core.data.Sequence object
            Sequence object representing melody 
        """

        queues = [multiprocessing.Queue() for _ in range(12)]
        jobs = []
        for key in range(12):

            p = multiprocessing.Process(target=worker,\
                args=(melody_sequence,key,self.fst,self.build_dir,queues[key]))
            jobs.append(p)
            p.start()

        for job in jobs:
            job.join()

        weights = [q.get() for q in queues]

        return np.argmin(weights)


    def generate_harmonic_sequence(self,melody_sequence,min_key=None,return_min_key=False):
        """ Generate the harmonic sequence from the melodic sequence.

        Parameters
        ----------
        melody_sequence: core.data.Sequence object
            Sequence object representing melody 

        Returns
        -------
        harm_seq: core.data.Sequence object
            Sequence object representing generated harmony
        """

        if min_key is None:    
            min_key = 12 - self.find_min_weight_transposition(melody_sequence=melody_sequence)
            print 'min key=',min_key            
            mel_seq = coredata.transpose_melodic_sequence_to_key(melodic_seq=melody_sequence,\
                to_key=min_key)
        else:
            mel_seq = melody_sequence
        
        harm_seq = chordgen.get_harmonic_seq(melody_sequence=melody_sequence,\
            full_fst=self.fst,\
            method=self.method,\
            build_dir=self.build_dir,\
            return_path_weight=False)

        labels = harm_seq.get_labels()
        if self.roman_to_chords:
            labels = [self.roman_to_chord_map[c] for c in labels]
        transp_chords = [coreutils.transpose_chord_from_c(chord=c,to_key=min_key) \
            for c in labels]
        harm_seq.set_labels(transp_chords)

        if return_min_key:
            return harm_seq,min_key
        else:
            return harm_seq

    def generate_chord_voicings(self,harmonic_sequence,\
        delimiter=None):
        """ Generate chord voicings from the harmonic sequence
        """
        # labels = [coreutils.transpose_chord_from_c(chord=c,to_key=min_key) for c in labels]

        chords_voiced = []
        prev_voicing = None

        for label in harmonic_sequence.get_labels():
            notes = choose_voicing(label, self.voicings, prev_voicing)
            prev_voicing = notes
            chords_voiced.append(notes)
            if not delimiter is None:
                chords_voiced.append([delimiter])

        return chords_voiced 

    def synthesize_to_file(self,output_audio_fname,\
        melody_sequence,\
        melody_audio=None,\
        beat_times=None,\
        fs=22050,\
        tuning_offset=0,\
        min_chord_dur_beats=0,\
        transpose_melody=False):
        """ Synthesize an audio file containing
        accompaniment and original melody audio, if 
        available 

        Parameters
        ----------
        output_audio_fname: string
            Full path/filename of output audio file
        melody_sequence: core.data.Sequence object
            Sequence object representing melody 
        beat_times: np.array
            array of beat times (in seconds)       
        melody_audio: np.array 
            an array containing the melody audio (to write 
            to the output audio file)
        fs: int
            sample rate
        transpose_melody: booelan
            transpose melody to all 12 keys, generate accompaniment (using shortest path)
            and pick accompaniment that has lowest total path weight
        """

        # min_key = 12 - self.find_min_weight_transposition(melody_sequence=melody_sequence)

        # mel_seq = coredata.transpose_melodic_sequence_to_key(melodic_seq=melody_sequence,\
        #     to_key=min_key)

        # harm_seq = chordgen.get_harmonic_seq(melody_sequence=melody_sequence,\
        #     full_fst=self.fst,\
        #     method=self.method,\
        #     build_dir=self.build_dir,\
        #     return_path_weight=False)

        harm_seq = self.generate_harmonic_sequence(melody_sequence=melody_sequence,\
            min_key=None)

        start_times = harm_seq.get_start_times()
        end_times = harm_seq.get_end_times()            
        labels = harm_seq.get_labels()

        # adjust start/end times to beat times
        if not beat_times is None:
            start_times = adjust_times(event_times=start_times,beat_times=beat_times)
            end_times = adjust_times(event_times=end_times,beat_times=beat_times)


        print start_times,'\n',end_times
        print 'transpose to:',min_key

        if self.roman_to_chords:
            labels = [self.roman_to_chord_map[c] for c in labels]

        labels = [coreutils.transpose_chord_from_c(chord=c,to_key=min_key) for c in labels]

        # chord_seq = [(e.start_time,e.end_time,e.label) for e in harm_seq.events]
        chord_seq = zip(start_times,end_times,labels)
        for s,e,c in chord_seq:
            print s,e,c
        midi_chords = generate_chord_midi(chord_sequence=chord_seq)

        # # Write out the MIDI data
        # midi_chords.write('testmidi.mid')

        y = midi_chords.fluidsynth(fs=fs)

        if not melody_audio is None:
            melody_audio /= melody_audio.max()
            pad_sz = np.abs(y.shape[0] - melody_audio.shape[0])
            if y.shape[0] > melody_audio.shape[0]:
                melody_audio = np.hstack([melody_audio,np.zeros((pad_sz,))])
            elif melody_audio.shape[0] > y.shape[0]:
                y = np.hstack([y,np.zeros((pad_sz,))])
            y += melody_audio
            y /= y.max()

        librosa.output.write_wav(output_audio_fname,y,fs)


def worker(melody_sequence,key,fst,build_dir,que):

    tmp_mel_seq = coredata.transpose_melodic_sequence_to_key(melodic_seq=melody_sequence,\
        to_key=key)

    print "key:",key,"mel:",tmp_mel_seq.get_labels()

    path_fst_filename = 'path_' + str(key) + '.fst'
    melody_fst_filename = 'mel_' + str(key) + '.fst'
    out_harmony_fst_filename = 'harm_' + str(key) + '.fst'

    tmpseq,w = chordgen.get_harmonic_seq(melody_sequence=tmp_mel_seq,\
        full_fst=fst,\
        method=chordgen.GEN_METHOD_SHORTEST,\
        build_dir=build_dir,\
        return_path_weight=True,\
        path_fst_filename=path_fst_filename,\
        melody_fst_filename=melody_fst_filename,\
        out_harmony_fst_filename=out_harmony_fst_filename)

    print "key:",key,"harm:",tmpseq.get_labels(),w

    que.put(w)

# def aggregate_sequence_data(sequence,boundaries):
#     """ Aggregate features that fall within boundaries """

#     start_times = []
#     end_times = []
#     features = []

#     for b in boundaries:
#         for

def fill_beat_times(beat_times,tempo):
    """ fill in missing beat times at beginning of 
    beat_times array
    """

    beat_dt = 1./(tempo/60.0)
    filled_beat_times = []

    beat_list = beat_times.tolist()
    curr_beat = beat_list[0]
    tmp_beats = []
    while curr_beat>=0.0:
        tmp_beats.append(curr_beat)
        curr_beat -= beat_dt

    filled_beat_times = tmp_beats[::-1]
    filled_beat_times.extend(beat_list[1:])

    return np.array(filled_beat_times)

def adjust_times(event_times,beat_times):
    """ Adjust event times to beat times """

    adjusted_times = []
    for t in event_times:
        idx = np.argmin(np.abs(beat_times-t))
        adjusted_times.append(beat_times[idx])

    return adjusted_times

def parse_pyin_notes_csv_file(csv_fname):
    """ Parse a csv file produced by running pYIN:Notes
    from Sonic Annotator. 

    Rows seem to be:
        filename*, note_start_time, note_duration, note_freq
    (filename only in first row)
    """

    start_times = []
    durations = []
    frequencies = []

    with open(csv_fname,'rt') as fh:
        reader = csv.reader(fh,delimiter=',')
        for row in reader:
            _, start_t, dur, freq = row
            start_times.append(float(start_t))
            durations.append(float(dur))
            frequencies.append(float(freq))

    return start_times, durations, frequencies

def run_sonic_annotator(algorithm_config_fname,\
    input_audio_fname,\
    output_fname,\
    sonic_annotator_bin='sonic-annotator'):
    """ Launch Sonic Annotator to analyze an audio file.

    Parameters
    ----------
    algorithm_config_fname: string
        full path to configuration file for algorithm used to
        estimate melody notes. Generated using Sonic Annotator, e.g.:
            sonic-annotator -s vamp:pyin:pyin:notes > pyin_notes.n3
        Can be modified by hand.
    input_audio_fname: string
        audio file to be analyzed
    output_fname: string
        output csv file containing output data 
    sonic_annotator_bin: string
        full path to sonic-annotator binary
    """

    cmd = [sonic_annotator_bin, '-t', algorithm_config_fname,\
        input_audio_fname, '-w', 'csv', '--csv-stdout']

    std_out, std_err = coreutils.launch_process(cmd, outfile_name=output_fname, outfile_type='text')

    # seems as though stout going to stderr?
    print 'std_out:',std_out
    print 'std_err:',std_err

def get_pc_data_pyin_notes(input_audio_fname):

    output_csv_fname = 'tmp_melody_notes.csv'
    #'pyin_notes.n3',\
    run_sonic_annotator(algorithm_config_fname=PYIN_CONFIG_FILE,\
        input_audio_fname=input_audio_fname,\
        output_fname=output_csv_fname,\
        sonic_annotator_bin='sonic-annotator')

    start_t, durs, freqs =\
        parse_pyin_notes_csv_file(csv_fname=output_csv_fname)

    midi = librosa.core.hz_to_midi(freqs)
    midi = [int(m) for m in np.round(midi)]
 
    end_t = np.add(start_t,durs)

    # to pitch class representation
    labels = ['pc' + str(coreutils.midi_note_to_pc(n)) for n in midi]
    print 'labels:',labels
    melody_sequence = coredata.Sequence(labels=labels,\
        start_times=start_t,end_times=end_t)

    return melody_sequence


def get_pc_data_aubio(y,fs):
    """ Use SLIVET note transcription method to get
    melodic sequence.
    """

    data = vamp.collect(y,fs,'vamp-aubio:aubionotes')
    freqs = [d['values'][0] for d in data['list']]
    midi = librosa.core.hz_to_midi(freqs)
    midi = [int(m) for m in np.round(midi)]

    start_t = [d['timestamp'].to_float() for d in data['list']]
    end_t = start_t[1:]
    end_t.append(y.shape[0]/float(fs))

    # to pitch class representation
    labels = ['pc' + str(coreutils.midi_note_to_pc(n)) for n in midi]
    melody_sequence = coredata.Sequence(labels=labels,\
        start_times=start_t,end_times=end_t)

    return melody_sequence

def get_midi_data_slivet(y,fs):
    """ Use SLIVET note transcription method to get
    melodic sequence.
    """

    data = vamp.collect(y,fs,'silvet:silvet')
    labels = []
    start_t = []
    end_t = []

    for d in data['list']:
        n = librosa.note_to_midi(d['label'])
        # pc = coreutils.midi_note_to_pc(n)
        # label = 'pc' + str(pc)
        st = d['timestamp'].to_float()
        # et = st + d['duration'].to_float()

        # labels.append(label)
        labels.append(n)
        start_t.append(st)
        # end_t.append(et)

    end_t = start_t[1:]
    # for some reason the last duration gets
    # screwed up
    # st = start_t[-1]
    # et = st + (float(y.shape[0])/float(fs))   
    delta_t = start_t[-1] - start_t[-2]
    et = end_t[-1] + delta_t
    end_t.append(et)

    # to pitch class representation
    labels = ['pc' + str(coreutils.midi_note_to_pc(n)) for n in labels]
    melody_sequence = coredata.Sequence(labels=labels,\
        start_times=start_t,end_times=end_t)

    return melody_sequence

def get_roman2chord_map(map_file=ROMAN_2_CHORD_FILE):
    """ Load the Roman numeral to chord label map """

    r2l_map = U.read_json_file(fname=map_file)

    return r2l_map


def get_all_voicings(voicing_file=None):
    """ Load chord voicings
    Args:
        voicing_file (str): path to json file of voicings
    Returns:
        voicing (dict): keys are chord names, vals are lists of voicings.
            Each voicing is a list of up to length 6 of midi note numbers.
    """

    if voicing_file is None:
        voicing_file = VOICING_FILE

    voicings = U.read_json_file(fname=voicing_file)

    return voicings


def voicing_dist(previous_voicing, voicing_candidate):
    """ Find the 'distance' between the previous voicing and the candidate.
    Args:
        previous_voicing (list): previous voicing
        voicing_candidate (list): current voicing candidate
    Returns:
        dist (float): average of min distance between notes
    """
    previous_voicing = np.array(previous_voicing)
    voicing_candidate = np.array(voicing_candidate)
    note_dists = np.zeros(len(previous_voicing))
    for i, note in enumerate(previous_voicing):
        can_dist = np.abs(note - voicing_candidate)
        value = voicing_candidate[np.where(can_dist == can_dist.min())]
        if len(value) > 1:
            value = value[0]
        note_dists[i] = np.abs(note - value)
    return np.mean(note_dists)


def choose_voicing(chord_name, voicings, prev_voicing=None):
    """ Given a chord name, a set of possible voicings, and the previous
    voicing, choose the best voicing.
    Args:
        chord_name (str): chord name of the form C:maj6, G:dim7, etc.
        voicings (dict): dictionary of possible voicings
        previous_voicing (list): Optional - previous voicing.
    Returns:
        voicing (list): best voicing for the given chord name
    """
    voicing_candidates = voicings[chord_name]
    if prev_voicing is not None:
        cand_dist = np.zeros(len(voicing_candidates))
        for i, cand in enumerate(voicing_candidates):
            cand_dist[i] = voicing_dist(prev_voicing, cand)
        voicing = voicing_candidates[np.argmin(cand_dist)]
    else:
        voicing = choice(voicing_candidates)
    return voicing

def generate_chord_midi(chord_sequence,\
    melody_sequence=None,\
    instrument="Acoustic Grand Piano",\
    octave_offset=2,\
    tuning_offset=0,\
    voicings=None):
    """ Given list of triples of the form (start_time, end_time, chord_name),
        generate midi file.
    Args:
        output_path (str): path to output midi file
        chord_sequence (list): list of triples of the form
            (start_time, end_time, chord_name), with start_time, end_time in
            seconds, and chord names of the form 'A:min6', 'Bb:maj', etc
        instrument (str): General Midi instrument name. Defaults to piano.
    Returns:
        Nothing
    """

    if voicings is None:
        voicings = get_all_voicings(VOICING_FILE)

    midi_chords = pretty_midi.PrettyMIDI(initial_tempo=180.)

    # Create an Instrument instance for chords
    guitar_program = pretty_midi.instrument_name_to_program(instrument)
    chords = pretty_midi.Instrument(program=guitar_program)

    # Iterate over note names, which will be converted to note number later
    prev_voicing = None
    for triple in chord_sequence:
        start_t = triple[0]
        end_t = triple[1]
        notes = choose_voicing(triple[2], voicings, prev_voicing)
        prev_voicing = notes
        for n in notes:
            note = pretty_midi.Note(velocity=100, pitch=n+12*octave_offset+tuning_offset,
                                    start=start_t, end=end_t)
            chords.notes.append(note)

    if not melody_sequence is None:
        prev_n = None
        for start_t,end_t,n in melody_sequence:
            pc = int(n.replace('pc',''))
            p = coreutils.pc_oct_to_midi_note(pc=pc,octave=6,prev_note=prev_n)
            prev_n = p
            note = pretty_midi.Note(velocity=100, pitch=p,
                                    start=start_t, end=end_t)
            chords.notes.append(note)

    # Add the chords instrument to the PrettyMIDI object
    midi_chords.instruments.append(chords)

    return midi_chords

def generate_note_midi(note_sequence,\
    instrument="Voice Oohs",\
    octave_offset=2):
    """ Given list of triples of the form (start_time, end_time, chord_name),
        generate midi file.
    Args:
        output_path (str): path to output midi file
        note_sequence (list): list of triples of the form
            (start_time, end_time, chord_name), with start_time, end_time in
            seconds, and MIDI note numbers
        instrument (str): General Midi instrument name. Defaults to piano.
    Returns:
        Nothing
    """

    midi = pretty_midi.PrettyMIDI()

    # Create an Instrument instance for chords
    guitar_program = pretty_midi.instrument_name_to_program(instrument)
    chords = pretty_midi.Instrument(program=guitar_program)

    # Iterate over note names, which will be converted to note number later
    for start_t,end_t,note in note_sequence:
        pm_note = pretty_midi.Note(velocity=100, pitch=note+12*octave_offset,
                                    start=start_t, end=end_t)
        chords.notes.append(pm_note)

    # Add the chords instrument to the PrettyMIDI object
    midi.instruments.append(chords)

    return midi

def generate_midi_file(output_path,\
    chord_sequence,\
    instrument="Acoustic Grand Piano"):

    midi_chords = generate_chord_midi(chord_sequence=chord_sequence,\
        instrument="Acoustic Grand Piano")

    # Write out the MIDI data
    midi_chords.write(output_path)

def melody_seq_to_harmony_seq(melody_seq,\
    fst,\
    output_path_midi,\
    instrument="Acoustic Grand Piano"):

    harm_seq = chordgen.get_harmonic_seq(melody_seq,fst)
    chord_seq = [ (s.start_time,s.end_time,s.label) for s in harm_seq.events]
    generate_chord_midi_file(output_path=output_path_midi,\
        chord_sequence=chord_seq,\
        instrument=instrument)


def generate_accomp(melody_audio_fname,\
        out_fname,\
        use_pyin_notes=True,\
        alyzer=None,\
        generator=None,\
        gen_method=None,\
        roman_to_chords=True,\
        do_estimate_tuning=False,\
        do_remove_perc=False,\
        min_chord_dur_beats=None,\
        build_dir='.'):
    """ Generate accompaniment from the melody audio file. 


    Parameters
    ----------
    melody_audio_fname: string
        full path of audio file containing melody
    out_fname: string
        full path of audio file containing melody plus generated accompaniment
    use_pyin_notes: Boolean
        use pYIN:Notes to estimate melody notes
    """


    # create analyzer and chord generator objects
    if alyzer is None:
        print 'make an AudioAnalyzer'
        alyzer = AudioAnalyzer()
    if gen_method is None:
        gen_method = chordgen.GEN_METHOD_SHORTEST
    if generator is None:
        # load FST
        print 'make ChordGenerator'

        if USE_BQ:
            isyms = fsm.SymbolTable(filename=BQ_INPUT_SYMS_FILE)
            osyms = fsm.SymbolTable(filename=BQ_OUTPUT_SYMS_FILE)
            fst = fsm.FST(filename=BQ_LOG_FST_FILE,isyms_table=isyms,osyms_table=osyms)
            rom2label = False
        else:
            isyms = fsm.SymbolTable(filename=INPUT_SYMS_FILE)
            osyms = fsm.SymbolTable(filename=OUTPUT_SYMS_FILE)
            fst = fsm.FST(filename=LOG_FST_FILE,isyms_table=isyms,osyms_table=osyms)
            rom2label = True
        generator = ChordGenerator(fst=fst,method=gen_method,roman_to_chords=rom2label,\
            build_dir=build_dir)

    # load audio file and re-tune
    print 80*'.'
    print 'analyzing audio...\n\n'
    print 'loading audio file:',melody_audio_fname,'\n'
    y_orig, sr = librosa.load(melody_audio_fname)

    print 'computing beat times...'
    tempo, beats = librosa.beat.beat_track(y=y_orig, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    # print 'beat times:',beat_times
    beat_times = fill_beat_times(beat_times,tempo)
    print 'beat times:',beat_times


    onset_frames = librosa.onset.onset_detect(y=y_orig, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print 'onset times:',onset_times

    # get rid of percussive parts of signal
    if do_remove_perc:
        print 'retain only harmonic portion of signal...'
        y_harm = librosa.effects.harmonic(y_orig)
    else:
        y_harm = y_orig

    if do_estimate_tuning:
        print 'estimate tuning...'
        tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
        print 'correct tuning...'
        y_tuned = librosa.effects.pitch_shift(y_harm, sr, -tuning)
    else:
        tuning = 0
        y_tuned = y_harm

    print 80*'.'

    if use_pyin_notes:
        print 'estimating melody...'
        tmp_audio_fname = 'tmp_melody_audio.wav'
        librosa.output.write_wav(path=tmp_audio_fname, y=y_tuned, sr=sr)
        mseq = alyzer.get_estimated_melody_sequence_pyin(audio_fname=tmp_audio_fname)
    else:
        # generate the accompaniment and write to a file
        # print 'generate accompaniment...'
        mseq = alyzer.get_estimated_melody_sequence(y_tuned,sr)

    if USE_BQ:
        mseq.repeat_events(beat_quant_level=BEAT_QUANT_LEVEL)
    print 'estimating key...'
    key = generator.estimate_key(mseq)

    print 'synthesizing audio to file...'
    generator.synthesize_to_file(output_audio_fname=out_fname,\
        melody_sequence=mseq,\
        fs=sr,\
        beat_times=beat_times,\
        # beat_times=onset_times,\
        melody_audio=y_orig,\
        tuning_offset=tuning,\
        min_chord_dur_beats=min_chord_dur_beats)

    print 'done'
    print 80*'.'


def main():
    parser = argparse.ArgumentParser(
        description="Generate accompaniment for input melody audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("in_dir",
    #                     action="store",
    #                     help="RockCorpus main folder")
    parser.add_argument("-i",\
                        action="store",\
                        dest="in_melody_file",\
                        help="Input audio file (monophonic melody, .wav format)")
    parser.add_argument("-o",\
                        action="store",\
                        dest="out_accomp_file",\
                        help="Output audio file (original melody plus MIDI accompaniment, .wav format)")    

    parser.add_argument('--est_tuning',\
        dest='do_estimate_tuning',\
        action='store_true',\
        help="Peform tuning estimate on the input audio file before pitch detection.",\
        default=False)

    parser.add_argument('--remove_perc',\
        dest='do_remove_perc',\
        action='store_true',\
        help="Peform percussion removal algorithm on the input audio file before pitch detection.",\
        default=False)

    parser.add_argument('--use_aubio',\
        dest='use_aubio',\
        action='store_true',\
        help="Use aubio instead of PYIN algorithm to do pitch detection on the input audio file.",\
        default=False)

    parser.add_argument('-g',\
        dest='gen_method',\
        action='store',\
        help='Method used to generate harmony ("shortest", "log_prob","uniform").',\
        default=chordgen.GEN_METHOD_SHORTEST)

    args = parser.parse_args()

    print 'Generating accompaniment file:',args.out_accomp_file,'for melody file:',args.in_melody_file
    print 'estimate tuning:',args.do_estimate_tuning
    print 'percussion removal:',args.do_remove_perc
    print 'generation method:',args.gen_method

    if not args.gen_method in\
        [chordgen.GEN_METHOD_SHORTEST, chordgen.GEN_METHOD_LOG_PROB, chordgen.GEN_METHOD_UNIFORM]:
        print '[ERROR] unknown generation method:',args.gen_method
        print 'Using method "shortest"'
        args.gen_method = chordgen.GEN_METHOD_SHORTEST

    use_pyin = not args.use_aubio

    generate_accomp(melody_audio_fname=args.in_melody_file, \
        out_fname=args.out_accomp_file,\
        do_estimate_tuning=args.do_estimate_tuning,\
        do_remove_perc=args.do_remove_perc,\
        use_pyin_notes=use_pyin,\
        gen_method=args.gen_method)

if __name__ == '__main__':
    main()