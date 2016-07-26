import gc
import glob
import numpy as np
import os

import scipy.linalg as LA
from scipy.interpolate import interp1d

try:
    import pandas as pd
    import seaborn as sns
except ImportError:
    print 'unable to import pandas or seaborn'

# from pyjams import pyjams
import pyjams

import hamilton.core.utils as coreutils
from hamilton.core import data as coredata
from hamilton.core import fileutils
from hamilton.core import fsm
from hamilton.generate import chordgen
from hamilton.generate import chordsynth
from hamilton.train import data as traindata
from hamilton.train import quantize
from hamilton.train import train


BEAT_QUANT_LEVEL = None #0.0625

ROCK_TRANSMAT_FNAME = 'transmat_rock.pkl'
JAZZ_TRANSMAT_FNAME = 'transmat_jazz.pkl'

ROMAN_2_MAJMINDIM_FILE = os.path.join(os.path.dirname(__file__),\
    '..','roman2majmindim.json')

METHOD_NAMES = { chordgen.GEN_METHOD_SHORTEST: 'shortest',\
    chordgen.GEN_METHOD_LOG_PROB: 'log prob',\
    chordgen.GEN_METHOD_UNIFORM: 'uniform'}

GEN_METHOD = chordgen.GEN_METHOD_SHORTEST
# GEN_METHOD = chordgen.GEN_METHOD_LOG_PROB
# GEN_METHOD = chordgen.GEN_METHOD_UNIFORM

#                           (Intercept)   diss/ch_diss_mean  ch_diss_std   kl_folk
QUANT_COEFFS_FOLK = np.array([1.2403695618, -0.1956439984, -1.3921110602, -0.0005554353])

#                           (Intercept)   diss/ch_diss_mean  ch_diss_std   kl_jazz
QUANT_COEFFS_JAZZ = np.array([1.249516941, -0.145372093, -0.694247878, -0.001427873])


QUANT_COEFFS = QUANT_COEFFS_FOLK


# results file lists
FLIST_EXPT_1_ERG_1ST_ORDER = glob.glob('/Users/jon/Dropbox/results/EXPT_1_fst/ergodic_1stOrder/*.json')
FLIST_EXPT_1_ERG_2ND_ORDER = glob.glob('/Users/jon/Dropbox/results/EXPT_1_fst/ergodic_2ndOrder/*.json')
FLIST_EXPT_1_SEQ_CONST_WT = glob.glob('/Users/jon/Dropbox/results/EXPT_1_fst/tree-constwt/*.json')
FLIST_EXPT_1_SEQ_EXP_N05 = glob.glob('/Users/jon/Dropbox/results/EXPT_1_fst/tree-expN0.5/*.json')
FLIST_EXPT_1_SEQ_EXP_N1 = glob.glob('/Users/jon/Dropbox/results/EXPT_1_fst/tree-expN1/*.json')
FLIST_EXPT_1_SEQ_EXP_N2 = glob.glob('/Users/jon/Dropbox/results/EXPT_1_fst/tree-expN2/*.json')
FLIST_EXPT_1_SEQ_EXP_N3 = glob.glob('/Users/jon/Dropbox/results/EXPT_1_fst/tree-expN3/*.json')

FLIST_EXPT_2_KN = glob.glob('/Users/jon/Dropbox/results/EXPT_2_ngram/old/best_kn/*.json')
FLIST_EXPT_2_WB = glob.glob('/Users/jon/Dropbox/results/EXPT_2_ngram/old/best_wb/*.json')
FLIST_EXPT_2_UNSMOOTH = glob.glob('/Users/jon/Dropbox/results/EXPT_2_ngram/old/unsmoothed_results/*.json')

class Fold(object):
    """ class to hold various information for a data fold """

    def __init__(self,**kwargs):
        """ Constructor 

        Attributes:
        -----------
        fold_num: int
            fold number
        trainset: Dataset object
            dataset for training FST model
        testset: Dataset object
            dataset for testing FST model
        build_dir: string
            full path to FST build directory
        fst_basename: string
            base filename for all FST models
        ngram_order: int
            n-gram order
        input_syms_filename: string
            full path to input symbols filename
        output_syms_filename: string
            full path to output symbols filename
        chord_seqs_filename: string
            full path to chord sequences filename
        label_map: dictionary
            dictionary to map chord labels in dataset to 
            restricted set of chordlabels
        codebook: np.array of floats
            codebook matrix, 12xN where N is number of chord types
        transmat: np.array of floats
            transition matrix for "real world" chord transitions
        time_resolution: float
            time resolution for constructing chromagrams
        num_output_paths: int
            number of output paths in generated harmony FST to consider
        unseen_transition_weight: float
            weight for unseen transitions in ergodic melody-to-chord FST
            if None, build regular FST instead of ergodic FST
        compose_fsts: boolean
            compose L and G into LoG, or use separately (i.e., compose melody
            with min(det(L)), compose that result with G)
        """

        self.fold_num = None
        self.trainset = None
        self.testset = None
        self.build_dir = None
        self.fst_basename = None
        self.ngram_order = None
        self.input_syms_filename = None
        self.output_syms_filename = None
        self.chord_seqs_filename = None
        self.label_map = None
        self.codebook = None
        self.transmat = None
        self.time_resolution = None
        self.num_output_paths = None
        self.unseen_transition_weight = None
        self.omit_ngram = None
        self.time_format = None
        self.witten_bell_k = None
        self.kn_num_bins = None
        self.kn_discount_D = None
        self.compose_fsts = None

        for key, value in kwargs.iteritems():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.time_resolution is None:
            self.time_resolution = 0.0232

        if self.omit_ngram == 1:
            self.omit_ngram = True
        else:
            self.omit_ngram = False

        if self.compose_fsts == 1 or self.compose_fsts is None:
            self.compose_fsts = True
        else:
            self.compose_fsts = False

        # self.chords = list(set(self.label_map.values()))

        print 'Compose FSTs = ',self.compose_fsts

        self.fst = None
        self.L_fst = None
        self.G_fst = None
        self.results = {}

        print 'TIME FORMAT:',self.time_format

    def build_models(self):
        """ build the phoneme FST and ngram models """
        L_fst_fn = os.path.join(self.build_dir,self.fst_basename+'_L.fst')
        ngram_fn = os.path.join(self.build_dir,self.fst_basename+'_ngram.fst')
        LoG_fst_fn = os.path.join(self.build_dir,self.fst_basename+'_LoG.fst')
        try:
            fsts = train.build_LoG(lang_mod_filename=L_fst_fn,\
                ngram_filename=ngram_fn,\
                LoG_filename=LoG_fst_fn,
                ngram_order=self.ngram_order,\
                isyms_filename=self.input_syms_filename,\
                osyms_filename=self.output_syms_filename,\
                dataset=self.trainset,\
                chord_seqs_filename=self.chord_seqs_filename,\
                unseen_transition_weight=self.unseen_transition_weight,\
                omit_ngram=self.omit_ngram,\
                witten_bell_k=self.witten_bell_k,\
                kn_num_bins=self.kn_num_bins,\
                kn_discount_D=self.kn_discount_D,\
                compose_fsts=self.compose_fsts)

            if self.compose_fsts:
                self.fst = fsts[0]
            else:
                self.L_fst,self.G_fst = fsts

        except AttributeError:
            print 'ERROR: unable to build L o G model'
            self.fst = None

    def evaluate(self):
        """ evaluate the FST/ngram model """

        if self.fst is None and self.L_fst is None and self.G_fst is None:
            print 'fst is None, skipping fold'
            return

        for filename in self.testset:
            jam = pyjams.load(filename)
            if len(jam.note) == 0:
                continue

            if not BEAT_QUANT_LEVEL is None:
                print 'using BEAT_QUANT_LEVEL=',BEAT_QUANT_LEVEL

            songs = traindata.jam_to_song_objects(jam=jam,\
                time_format=self.time_format,\
                beat_quant_level=BEAT_QUANT_LEVEL)
            print jam.file_metadata.title

            song = songs[0]

            # diss_mean,diss_std,kl,avg_seq_diff,chord_cov,ent_rate,pir,weights,ent_rate_mel,pir_mel =\
            #     self.evaluate_song(song)
            metrics = self.evaluate_song(song)

            # curr_key = 'gen'
            # song_results = {}
            # song_results[curr_key] = {}
            # song_results[curr_key]['kl'] = kl
            # song_results[curr_key]['diss_mean'] = diss_mean
            # song_results[curr_key]['diss_std'] = diss_std
            # song_results[curr_key]['seq_diff'] = avg_seq_diff
            # song_results[curr_key]['chord_cov'] = chord_cov
            # song_results[curr_key]['ent_rate'] = ent_rate
            # song_results[curr_key]['pir'] = pir
            # song_results[curr_key]['weights'] = weights
            # song_results[curr_key]['ent_rate_mel'] = ent_rate_mel
            # song_results[curr_key]['pir_mel'] = pir_mel

            key = song.title
 
            if key in self.results.keys():
                print '********************************* CONFLICTING KEY:',key
                key = key + '__conflict'
            self.results[key] = metrics


    def evaluate_song(self,song):
        """ Evaluate a single test song

        Parameters
        ----------
        song: Song object
            song to evaluate

        Returns
        -------
        diss_mean: float
            mean of frame-wise dissonance
         diss_std: float
            standard deviation of frame-wise dissonance
        kl: float
            KL divergence between chord sequence and "real world" 
            transition matrix
        chord_cov: float
            chord "coverage" -- i.e., how many chords were used in the
            harmonic sequence out of all possible chords in the vocabulary
        chord_diffs: float
            measure of how much chords change, computed as 
            # chord changes / length of sequence - 1 
        """

        n_chord_types = self.codebook.shape[1]

        song.key_normalize_melody()
        song.quantize(mel_quantize_func=traindata.melody_note_quantize)

        # generate harmony
        print 'generating harmonic sequence using method:',METHOD_NAMES[GEN_METHOD]
        song.harmonic_sequence = None

        # compute the metrics
        # if self.num_output_paths == 1:
        if True:
            print 80*'.'
            print 'generating harmonic sequence...'
            print 'npaths=',self.num_output_paths
            print 80*'.'
            metrics = {}
            for n in range(self.num_output_paths):
                if self.compose_fsts:
                    curr_harm_seq = chordgen.get_harmonic_seq(\
                        melody_sequence=song.melodic_sequence,\
                        full_fst=self.fst,\
                        method=GEN_METHOD,\
                        npaths=1)
                else:
                    curr_harm_seq = chordgen.get_harmonic_seq_separate_fsts(\
                        melody_sequence=song.melodic_sequence,\
                        L_fst=self.L_fst,\
                        G_fst=self.G_fst,\
                        method=GEN_METHOD,\
                        npaths=1)

                if curr_harm_seq is None:
                    print 'ERROR: unable to generate output for:',song.title
                    return None

                song.harmonic_sequence = curr_harm_seq
                print 'harmonic sequence:',song.harmonic_sequence.get_labels()
                m = self.compute_metrics(song)
                weights = 0.0  #<--- just ignore this for now....
                for k,v in m.items():
                    if k not in metrics.keys():
                        metrics[k] = [v]
                    else:
                        metrics[k].append(v)
        else:
            print '# output paths=',self.num_output_paths
            metrics = self.compute_metrics_multipath(song)

        return metrics

    def compute_metrics(self,song):
        """ compute metrics for the specified song 
        """

        return compute_metrics(song=song,\
            label_map=self.label_map,\
            time_resolution=self.time_resolution,\
            codebook=self.codebook,\
            transmat=self.transmat)

    def compute_metrics_multipath(self,song):
        """ compute metrics for the specified song 
        in the multiple output path case
        """

        filebase = os.path.join(self.build_dir,'_path_fst_')

        npaths_fst = chordgen.build_path_fst(\
            melody_sequence=song.melodic_sequence,\
            full_fst=self.fst,\
            method=GEN_METHOD,
            npaths=self.num_output_paths)

        path_fsts = fsm.get_all_path_fsts(fst=npaths_fst,filebase=filebase)

        metrics = {}

        for path_fst in path_fsts:
            tmp_song = coredata.song_from_fst(\
                melodic_sequence=song.melodic_sequence,\
                harmonic_fst=path_fst)

            print 'harmonic sequence:',tmp_song.harmonic_sequence.get_labels()
            m = compute_metrics(song=tmp_song,\
                label_map=self.label_map,\
                time_resolution=self.time_resolution,\
                codebook=self.codebook,\
                transmat=self.transmat)

            m['weight'] = path_fst.total_weight()

            for k,v in m.items():
                if k not in metrics.keys():
                    metrics[k] = [v]
                else:
                    metrics[k].append(v)

        return metrics

class Evaluator(object):
    """ class to train and evaluate models using
    cross-fold validation
    """

    def __init__(self,config_file):

        """ Constructor

        Parameters
        ----------
        config_file: str
            full path to json configuration file that
            specifies all parameters for this evaluation
        """

        self.config_file = config_file

        # TODO: fix this...
        self.unseen_transition_weight = None
        self.omit_ngram = 0
        self.compose_fsts = True
        self.time_resolution = None
        self.time_format = None
        self.label_map_file = None
        self.witten_bell_k = None
        self.kn_discount_D = None
        self.kn_num_bins = None

        # set configuration parameters as properties
        config_params = fileutils.read_json_file(fname=self.config_file)
        for k,v in config_params.items():
            setattr(self,k,v)

        print 'writing to results file:',self.results_file

        # error/sanity checking
        # an actual value here means that we're building an ergodic FST,
        # which requires a melodic context length of 1
        # if self.unseen_transition_weight is not None:
        #     assert self.fst_melodic_context_len == 1

        if self.time_format is None or self.time_format == 'sec':
            self.time_format = traindata.TIME_FORMAT_SEC
        elif self.time_format == 'meas':
            self.time_format = traindata.TIME_FORMAT_MEAS

        if self.time_resolution is None:
            self.time_resolution = 0.0232

        self.data_folds = []
 
        if not self.label_map_file is None:
            self.label_map = fileutils.read_json_file(fname=self.label_map_file)
        else:
            self.label_map = None

        if self.codebook_file is None:
            self.codebook_file = 'default codebook: Major/minor/dim templates'
            self.codebook = coreutils.build_maj_min_dim_templates()
        else:
            self.codebook = fileutils.read_pickle_file(fname=self.codebook_file)

        self.transmat = fileutils.read_pickle_file(fname=self.transmat_file)

        print
        print 40*'-'
        print 'time format:',self.time_format
        print 'chord label map file:',self.chord_label_map_file
        print 40*'-'
        print

        if self.compute_dataset_metrics:
            print '\nevaluating dataset'
            print 'time res:',self.time_resolution
            self.eval_dataset()
        else:
            print '\nbuilding data folds...'
            self.build_data_folds()
            print '\ndone building data folds\n'

        print 40*'-'
        print 'melodic context length:',self.fst_melodic_context_len
        print 'unseen transition weight:',self.unseen_transition_weight
        print 'omit ngram:',self.omit_ngram
        print 'num. output paths:',self.num_output_paths
        print 40*'-'

    def build_data_folds(self):
        """ build the Dataset folds """

        randlist = glob.glob(self.jams_dir+'/*.jams')
        nfiles = len(randlist)
        foldsize = int(np.ceil(float(nfiles)/float(self.num_folds)))
        foldidx = range(0,nfiles,foldsize)
        np.random.shuffle(randlist)
        for foldnum in range(self.num_folds):
            print 20*'-'
            print 'building data fold #',foldnum+1,'/',self.num_folds

            testset = randlist[foldidx[foldnum]:foldidx[foldnum]+foldsize]

            trainset = traindata.Dataset(jams_dir=self.jams_dir,\
                ignore_filelist=testset,
                melodic_context_len = self.fst_melodic_context_len,\
                time_format=self.time_format,\
                chord_label_map_file=self.chord_label_map_file)

            if not BEAT_QUANT_LEVEL is None:
                print '** (train set) using BEAT_QUANT_LEVEL=',BEAT_QUANT_LEVEL
                trainset.repeat_melodic_events(beat_quant_level=BEAT_QUANT_LEVEL)

            fold = Fold(fold_num = foldnum,\
                trainset = trainset,\
                testset = testset,\
                build_dir = self.build_dir,\
                fst_basename = self.fst_basename,\
                ngram_order = self.ngram_order,\
                input_syms_filename = self.input_syms_filename,\
                output_syms_filename = self.output_syms_filename,\
                chord_seqs_filename = self.chord_seqs_filename,\
                label_map = self.label_map,\
                codebook = self.codebook,\
                transmat = self.transmat,\
                time_resolution = self.time_resolution,
                # compute_dataset_metrics = self.compute_dataset_metrics,\
                num_output_paths = self.num_output_paths,\
                unseen_transition_weight=self.unseen_transition_weight,\
                omit_ngram=self.omit_ngram,\
                compose_fsts=self.compose_fsts,\
                time_format=self.time_format,\
                chord_label_map_file=self.chord_label_map_file,\
                witten_bell_k=self.witten_bell_k,\
                kn_num_bins=self.kn_num_bins,\
                kn_discount_D=self.kn_discount_D)

            self.data_folds.append(fold)

    def build_and_eval(self):
        """ build and evaluate models for all folds """

        if self.compute_dataset_metrics:
            print 'already computed dataset metrics....'
            return

        results = {}
        for i,fold in enumerate(self.data_folds):
            print 20*'-'
            print 'evaluating data fold #',i+1,'/',self.num_folds
            print 'building models...'
            fold.build_models()
            print '\nevaluating...'
            fold.evaluate()
            print '\n'
            # if not self.compose_fsts:
            # # if True:
            #     results[i] = fold.results
            #     fname = self.results_file
            #     fname = fname.replace('.json','')
            #     fname = fname + '_' + str(i) + '.json'
            #     self.write_results(results=results,fname=fname)
            results[i] = fold.results
            if not self.compose_fsts:
                fname = self.results_file
                fname = fname.replace('.json','')
                fname = fname + '_' + str(i) + '.json'
                self.write_results(results=results,fname=fname)

        self.write_results(results=results)

    def write_results(self,results,fname=None):
        """ Write results to json file """

        if fname is None:
            fname = self.results_file
        filepath = os.path.join(self.results_dir,fname)
        print '\n\nwriting results to',filepath
        fileutils.write_json_file(fname=filepath, data=results)

    def eval_dataset(self):

        filenames = glob.glob(self.jams_dir+'/*.jams')

        results = {}
        results['0'] = {}

        for filename in filenames:
            jam = pyjams.load(filename)
            if len(jam.note) == 0:
                continue

            songs = traindata.jam_to_song_objects(jam=jam,time_format=self.time_format)
            print jam.file_metadata.title

            song_results = {}

            for i,song in enumerate(songs):
                song.key_normalize_melody()
                song.quantize(mel_quantize_func=traindata.melody_note_quantize)

                # diss_mean, diss_std, kl = compute_metrics(song=song,\
                #     label_map=self.label_map,\
                #     time_resolution=self.time_resolution,\
                #     codebook=self.codebook,\
                #     transmat=self.transmat)

                # # hcg = song.harmonic_sequence.to_chromagram(label_map=self.label_map,time_res=self.time_resolution)
                # # idx = quantize.quantize_chromagram_to_indices(hcg,self.codebook)
                # # a = build_transmat_from_seq(idx,self.codebook.shape[1])
                # # a = a.T
                # symbol_sequence = song.harmonic_sequence.get_labels()
                # # print symbol_sequence
                # a = build_minimal_transmat_from_seq(symbol_sequence=symbol_sequence)
                # pi_a = compute_stationary_dist(a)
                # ent_rate = compute_entropy_rate(P=a,pi_a=pi_a)
                # pir = compute_predictive_info_rate(P=a, pi_a=pi_a)

                # curr_key = 'data'
                # song_results[curr_key] = {}
                # song_results[curr_key]['kl'] = kl
                # song_results[curr_key]['diss_mean'] = diss_mean
                # song_results[curr_key]['diss_std'] = diss_std        
                # song_results[curr_key]['ent_rate'] = ent_rate
                # song_results[curr_key]['pir'] = pir

                metrics = self.compute_metrics(song)

                key = song.title + '_' + str(i)
                if key in results.keys():
                    print 'CONFLICTING KEY:',key
                results['0'][key] = metrics

        print 'writing results...'
        self.write_results(results=results)

    def compute_metrics(self,song):
        return compute_metrics(song=song,\
            label_map=self.label_map,\
            time_resolution=self.time_resolution,\
            codebook=self.codebook,\
            transmat=self.transmat)

def chord_list_to_idx(chord_list,chord_idx_list,label_map):
    """ Convert a list of chords to indices """

    idx = []
    for chord in chord_list:
        chord_sym = label_map[chord]
        chord_idx = chord_idx_list.index(chord_sym)
        idx.append(chord_idx)

    return idx

def compute_stationary_dist(P,overwrite=False):
    """ compute the stationary distribution from transition probability matrix P
    from: https://github.com/jstac/quant-econ/blob/master/quantecon/mc_tools.py 
    also see gth_solve.py 
    """

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = LA.eig(P, left=True, right=False)

    # Find the index for unit eigenvalues
    index = np.where(np.abs(eigvals - 1.) < 1e-12)[0]

    # Pull out the eigenvectors that correspond to unit eig-vals
    uniteigvecs = eigvecs[:, index]

    stationary_dists = uniteigvecs/(np.sum(uniteigvecs, axis=0) + 1e-20)

    return np.abs(stationary_dists)

def compute_entropy_rate(P,pi_a):
    """ compute the entropy rate (equation 4 in Abdallah & Plumbley Information Dynamics)
    """

    N = P.shape[0]
    pi_a = pi_a.reshape((N,))

    ent_rate = 0.0

    for i in range(N):
        for j in range(N):
            ent_rate += -pi_a[i] * P[j,i] * coreutils.safelog2(P[j,i])

    return ent_rate


def compute_entropy(P):
    """ Compute the entropy of a distribution """

    ent = -np.sum(P*coreutils.safelog2(P))
    return ent

def compute_multi_information_rate(P,pi_a):
    """ Compute the redundancy, or multi-information rate 
    (see Abdallah & Plumbley Info. Dynamics "Related Work" section)
    """

    mir = compute_entropy(pi_a) - compute_entropy_rate(P,pi_a)
    return mir

def compute_predictive_info_rate(P, pi_a):
    """ compute the predictive information rate (equation 7 in Abdallah & Plumbley Information Dynamics)
    """

    P2 = np.dot(P,P)
    pi_a2 = compute_stationary_dist(P=P2)
    # TODO: figure out fix for this
    if not pi_a.shape == pi_a2.shape:
        print '*********** ERROR WITH compute_stationary_dist'
        pi_a2 = pi_a
    H2 = compute_entropy_rate(P=P2,pi_a=pi_a2)

    H = compute_entropy_rate(P=P,pi_a=pi_a)

    return H2 - H

def compute_nll(X,codebook,model):
    """ compute the log-likelihood of chromagram """

    idx = quantize.quantize_chromagram_to_indices(X=X,codebook=codebook)

    nfr = X.shape[1]
    nll = np.zeros((nfr,))

    for t in range(nfr-1):
        i = idx[t]
        j = idx[t+1]
        if i<0 or j<0:
            continue
        nll[t] = -coreutils.safelog2(model[i,j])

    return nll

def compute_kl_div(X,codebook,model):
    """ compute the KL divergence between model and transition matrix from chromagram 

    Parameters
    ----------
    X: np.matrix
        chromagram matrix, 12xT where T is number of time frames
    codebook: np.matrix
        codebook matrix, 12xN where N is number of chord types
    model: np.matrix
        transition matrix for "real world" chord transitions
    """

    idx = quantize.quantize_chromagram_to_indices(X=X,codebook=codebook)
    transmat = build_transmat_from_seq(idx=idx,size=codebook.shape[1])

    kl_div = coreutils.compute_kl_div(P=model,Q=transmat)

    return kl_div

# def compute_quant_metric(diss_mean, diss_std, kl, coeffs=QUANT_COEFFS):
def compute_quant_metric(diss_mean, chord_diss_mean, chord_diss_std, kl, coeffs=QUANT_COEFFS):
    """ Compute the quantitative metric.
    """

    chord_diss_mean += 1e-20
    vals = np.array([1.0, diss_mean/chord_diss_mean, chord_diss_std, kl],dtype=np.float)        
    qm = np.dot(vals, coeffs)

    return qm

def get_harmonic_seq_from_jam(jam,\
    full_fst,\
    melody_fst_filename,\
    out_harmony_fst_filename,\
    chord_seq_num=0,\
    method=GEN_METHOD,
    roman_to_label_fst=None,\
    time_format=traindata.TIME_FORMAT_MEAS):
    """ Get the harmonic sequence from the melody in
    the given JAM file.

    Parameters
    ----------
    jam: JAM object
        JAM object
    full_fst: FST object
        full FST model (composition of L and G,
        determinized/minimized, disambiguation symbols removed)
    chord_seq_num: int
        number of chord sequence to use for key-normalization
    melody_fst_filename: string
        name of compiled melody FST
    out_harmony_fst_filename: string
        name of compiled FST file consisting of output
        harmony sequence
    method: string
        method to generate accompaniment sequences

    Returns
    -------
    harmony_seq: list of strings
        list of melodic symbols
    """

    songs = traindata.jam_to_song_objects(jam=jam,time_format=time_format)
    melseq = song.get_melodic_sequence(chord_seq_num=chord_seq_num)

    inseq,outseq = chordgen.get_harmonic_seq(melody_seq=melseq,\
        full_fst=full_fst,\
        melody_fst_filename=melody_fst_filename,\
        out_harmony_fst_filename=out_harmony_fst_filename,\
        method=method,\
        roman_to_label_fst=roman_to_label_fst)

    return inseq,outseq

def build_minimal_transmat_from_seq(symbol_sequence):
    """ Build a minimal transition matrix from the symbol
    sequence, which means that the only the symbols
    in the sequence are considered.
    """

    symbols = list(set(symbol_sequence))
    N = len(symbols)

    # build the list of indices
    idx = []

    for s in symbol_sequence:
        i = symbols.index(s)
        idx.append(i)

    a = build_transmat_from_seq(idx=idx,size=N)

    return a

def build_transmat_from_seq(idx,size):
    """ build a transition matrix from a series of indices into a codebook (i.e., quantized chord sequence).
    transition matrix T[i,j] -> i = from idx, j = to idx

    Parameters
    ----------
    idx: list of ints
        a sequence of integers corresponding to chord classes
    size: int
        total number of possible classes

    Returns
    -------
    T: numpy matrix of shape (size,size)
        transition matrix T[i,j] -> i = from idx, j = to idx
    """

    T = np.zeros((size,size)) + coreutils.EPS
    N = len(idx)
    for n in range(N-1):
        T[idx[n],idx[n+1]] += 1.0

    # ******* there's a better way to do this, too tired right now to figure it out....
    for i in range(size):
        T[i,:] = T[i,:] / T[i,:].sum()

    return T


def compute_metrics(song,\
    label_map,\
    time_resolution,\
    codebook,\
    transmat):
    """ Compute metrics for the specified song 

    Parameters
    ----------
    song: Song object
        the song to analyze
    label_map: dictionary
        dictionary to map quantized labels to a restricted set (e.g.
        36 major/minor/diminished triads)
    time_resolution: float
        time resolution used when generating chromagrams
    codebook: np.array of floats
        matrix of codebook vectors, dimension k x 12,
        where k is number of vectors. if none, use
        pitch classes as symbols
   transmat: np.array of floats
        transition matrix for "real world" chord transitions

    Returns
    -------
    metrics: dict
        dictionary with computed metrics, which include:
        mean_diss_val: float
            mean of frame-wise dissonance values computed on melody+chord sequence 
            chromagram
        std_diss_val: float
            standard deviation of frame-wise dissonance values computed on 
            melody+chord sequence chromagram
        kl: float
            KL divergence between transition matrix computed from song chord sequence
            and "real world" transition matrix (transmat)
        harm_diss_mean: float
            mean of frame-wise dissonance values computed on chord sequence chromagram 
        harm_diss_std: float
            standard deviation of frame-wise dissonance values computed on chord sequence
            chromagram

    """

    metrics = {}

    # compute dissonance metrics and KL
    harm_cg =\
        song.harmonic_sequence.to_chromagram(label_map=label_map,\
            time_res=time_resolution)
    comb_cg = song.get_combined_chromagram(label_map=label_map,\
        time_res=time_resolution)

    kl = compute_kl_div(X=harm_cg, codebook=codebook,\
        model=transmat)

    diss_vals = coreutils.compute_harmonic_dissonance_vals(X=comb_cg)
    harm_diss_vals = coreutils.compute_harmonic_dissonance_vals(X=harm_cg)

    # compute information-theoretic metrics
    # symbol_sequence = song.harmonic_sequence.get_labels()
    print 'using compressed label seqs...'
    symbol_sequence = song.harmonic_sequence.get_compressed_labels()
    a = build_minimal_transmat_from_seq(symbol_sequence=symbol_sequence)
    pi_a = compute_stationary_dist(a)
    ent_rate = compute_entropy_rate(P=a,pi_a=pi_a)
    pir = compute_predictive_info_rate(P=a, pi_a=pi_a)
    mir = compute_multi_information_rate(P=a, pi_a=pi_a)

    # symbol_sequence = song.melodic_sequence.get_labels()
    symbol_sequence = song.melodic_sequence.get_compressed_labels()
    a = build_minimal_transmat_from_seq(symbol_sequence=symbol_sequence)            
    pi_a = compute_stationary_dist(a)
    mel_ent_rate = compute_entropy_rate(P=a,pi_a=pi_a)
    mel_pir = compute_predictive_info_rate(P=a, pi_a=pi_a)
    mel_mir = compute_multi_information_rate(P=a, pi_a=pi_a)

    metrics['diss_mean'] = diss_vals.mean()
    metrics['diss_std'] = diss_vals.std()
    metrics['kl'] = kl
    metrics['harm_diss_mean'] = harm_diss_vals.mean()
    metrics['harm_diss_std'] = harm_diss_vals.std()
    metrics['ent_rate'] = ent_rate
    metrics['pir'] = pir
    metrics['mir'] = mir
    metrics['mel_ent_rate'] = mel_ent_rate
    metrics['mel_pir'] = mel_pir
    metrics['mel_mir'] = mel_mir
    metrics['weight'] = None

    return metrics

def compute_avg_seq_diff(sequence):
    """ Compute the average sequence diff value """

    seq_diff = sequence_diff(sequence=sequence)

    return seq_diff.sum() / float(len(seq_diff))

def compute_chord_coverage(sequence,n_chord_types):
    """ Compute the chord 'coverage' """

    chordset = set(sequence)

    return float(len(chordset)) / float(n_chord_types)

def sequence_diff(sequence):
    """ Peform a diff() (in the sense of derivative)
    on a sequence of strings. The diff sequence will
    consist of 1 indicating a difference and 0 indicating
    no difference. 
    """

    N = len(sequence)-1
    seq_diffs = np.zeros((N,))
    for i in range(N):
        if not sequence[i] == sequence[i+1]:
            seq_diffs[i] = 1

    return seq_diffs


def get_chord_hist_from_jams(r2c_map_file=None):
    jamsdir = '/Users/jon/Dropbox/testjams'
    if r2c_map_file is None:
        r2c_map_file = 'roman2majmindim.json'

    dset = traindata.Dataset(jams_dir=jamsdir,chord_label_map_file=r2c_map_file)
    iopairs_dict = dset.seqs_to_io_pairs()
    chord_dist = {}

    keys = ['C:maj', 'Db:maj', 'D:maj', 'Eb:maj', 'E:maj', 'F:maj', 'Gb:maj',\
        'G:maj', 'Ab:maj', 'A:maj', 'Bb:maj', 'B:maj', 'C:min', 'Db:min', 'D:min',\
        'Eb:min', 'E:min', 'F:min', 'Gb:min', 'G:min', 'Ab:min', 'A:min', 'Bb:min',\
        'B:min', 'C:dim', 'Db:dim', 'D:dim', 'Eb:dim', 'E:dim', 'F:dim', 'Gb:dim',\
        'G:dim', 'Ab:dim', 'A:dim', 'Bb:dim', 'B:dim']

    for k in keys:
        chord_dist[k] = 0

    for seq in iopairs_dict.values():
        for c in seq:
            chord_dist[c] += 1    

    # for seq in iopairs_dict.values():
    #     for c in seq:
    #         if not c in chord_dist:
    #             chord_dist[c] = 1
    #         else:
    #             chord_dist[c] += 1

    return chord_dist

def build_chord_hist(iopairs_dict,chord_types):
    chord_hist = {}
    for k in iopairs_dict.keys():
        chord_hist[k] = {}
        for c in chord_types:
            chord_hist[k][c] = 0

    for k,v in iopairs_dict.items():
        for c in v:
            chord_hist[k][c] += 1

    return chord_hist

def get_chord_hist_from_file(fn,r2c_map=None):
    if r2c_map is None:
        r2c_map = fileutils.read_json_file('roman2majmindim.json')

    seqs = get_chords_seqs_from_file(fn)
    chord_hist = build_chord_hist_from_seq_list(seqs,r2c_map)
    return chord_hist

def build_chord_hist_from_seq_list(seqs,r2c_map):
    chord_hist = {}

    triads = ['maj','min','dim']
    roots = coreutils.PITCH_CLASSES
    # keys = []
    for triad in triads:
        for root in roots:
            k = root + ':' + triad
            chord_hist[k] = 0

    # keys = r2c_map.values()
    # for k in keys:
    #     chord_hist[k] = 0

    for seq in seqs:
        for c in seq:
            c = r2c_map[c]
            chord_hist[c] += 1

    return chord_hist

def get_chords_seqs_from_file(fname):
    lines = fileutils.read_text_file(fname)
    seqs = []
    for line in lines:
        line = line.strip()
        if 'harmonic sequence:' in line:
            seq = line.split(':')[1]
            seq = seq.strip()
            seq = seq.replace('[','')
            seq = seq.replace(']','')
            seq = seq.replace("'",'')
            seq = seq.split(',')
            seq = [s.strip() for s in seq]
            seqs.append(seq)

    return seqs

def compute_unique_chord_seqs_per_song(chord_seqs,n_paths):
    """ Find the proportion of unique chord sequences computed 
    on a per song basis for k paths """

    n_songs = 192
    n_seqs_per_song = 100

    song_maps = []
    for n in range(n_songs):
        curr_map = {}
        start_i = n*n_seqs_per_song
        end_i = start_i + n_paths
        curr_seqs = chord_seqs[start_i:end_i]
        for i in range(n_paths):
            seq = curr_seqs[i]
            key = ' '.join(seq)
            if key not in curr_map:
                curr_map[key] = 1
            else:
                curr_map[key] += 1
        song_maps.append(curr_map)

    return song_maps

def compute_average_unique_seqs_per_song(fname,n_paths):
    chord_seqs = get_chords_seqs_from_file(fname)

    seqs_per_song = []

    if type(n_paths) == int:
        n_paths = [n_paths]

    for n in n_paths:
        smaps = compute_unique_chord_seqs_per_song(chord_seqs=chord_seqs,\
            n_paths=n)

        lens = [len(s) for s in smaps]
        avg = float(np.sum(lens)) / 192.0
        seqs_per_song.append(avg)

    return seqs_per_song

def main():
    """ Main function """

    filenames = [\
        'ergodic/build_config_ergN11.0.json',\
        # 'tree/build_config_context16_path1_wb_k1.0_N2.json',\
        # 'tree/build_config_context18_path1_wb_k1.0_N2.json',\
        # 'tree/build_config_context20_path1_wb_k1.0_N2.json',\
        # 'tree/build_config_context22_path1_wb_k1.0_N2.json',\
    ]

    msg = '\n'.join(filenames)
    print '\n'
    print 'Current batch of config files:\n',msg
    print '\n'
    print 40*'-'

    for fn in filenames:
        # if fn == 'tree/build_config_context16_pathLOG.json':
        #     GEN_METHOD = chordgen.GEN_METHOD_LOG_PROB
        # elif fn == 'tree/build_config_context16_pathUNIFORM.json':
        #     GEN_METHOD = chordgen.GEN_METHOD_UNIFORM
        # else:
        #     print '????'

        fn = os.path.join('train','config_files',fn)
        print 40*'='
        print 'gen method:',METHOD_NAMES[GEN_METHOD]

        print '\n\n'
        print 'evaluating config file:',fn
        try:
            evaluator = Evaluator(config_file=fn)
            evaluator.build_and_eval()
            for f in evaluator.data_folds:
                del f.trainset
                del f.testset
                del f.fst
                del f
            del evaluator
            gc.collect()
        except IOError:
            print 'unable to process:',fn
            continue
        print '\n\n'


if __name__ == '__main__':
    main()