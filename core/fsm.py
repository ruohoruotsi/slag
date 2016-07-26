""" Module for doing various finite state machine stuff
(training FSTs and generating output data) and various
classes for wrapping OpenFST etc. """

import numpy as np
import shutil
import os

import hamilton.core.utils as CU
import hamilton.core.fileutils as U

# from marl.audio import utils as marlutils
import marl

FST_BIN_DIR = '/usr/local/bin'
EPSILON_LABEL = '<epsilon>'


class FST(object):
    """ class to wrap all OpenFST operations """

    def __init__(self,filename,\
        isyms_table,\
        osyms_table,\
        keep_isyms=True,\
        keep_osyms=True,\
        arc_type='log',\
        is_acceptor=False,\
        overwrite_isyms_file=True):

        """ Constructor

        Parameters
        ----------
        filename: string
            FST file name
        isyms_table: SymbolTable object
            input symbols
        osyms_table: SymbolTable object
            output symbols
        keep_isymbols: bool
            keep input symbols table
        keep_osymbols: bool
            keep output symbols table
        arc_type: string
            one of 'standard' or 'log'
        is_acceptor: bool
            is this FST an acceptor or not
        initial_state: int
            the number corresponding to the initial state

        """

        f = os.path.split(filename)[1].split('.')
        if not f[-1] == 'fst':
            self.filename = filename + '.fst'
        else:
            self.filename = filename
        self.isyms_table = isyms_table
        self.osyms_table = osyms_table
        self.keep_isyms = keep_isyms
        self.keep_osyms = keep_osyms
        self.arc_type = arc_type
        self.is_acceptor = is_acceptor
        self.overwrite_isyms_file = overwrite_isyms_file


        # FST text definition format
        # self.final_state = []
        # self.initial_state = initial_state
        self.states = {}
        self.arcs = []

        # is this .fst encoded?
        self.is_encoded = False


    def num_states(self):
        """ get the total number of states in the FST """

        if len(self.states) == 0:
            return 0

        n = np.max(self.states.keys()) + 1
        assert len(self.states) == n
        return n

    def total_weight(self):
        """ Return the total weight of all paths.
        Weights with no weight are assigned weight 0.

        Returns
        -------
        tot_weight: float
            total weight of all paths
        """

        weights = [a.weight for a in self.arcs if not a.weight is None]
        tot_weight = np.sum(weights)

        return tot_weight

    def final_states(self):
        """ return a list of final states """

        return [ s for s in self.states.values() if s.is_final ]

    def initial_states(self):
        """ return a list of initial states """

        return [ s for s in self.states.values() if s.is_initial ]

    def insert_prefinal_disambig_symbols(self,disambig_sym='#'):
        """ add disambiguation symbol before each penultimate state.
        """

        curr_state_num = len(self.states)
        disambig_num = 0

        for curr_arc in self.arcs:
            from_s = self.states[curr_arc.from_state]
            if not from_s.is_initial:
                continue

            print 'from=',curr_arc.from_state,'\tto=',curr_arc.to_state

            # make a new intermediary state, make current
            # arc point to that instead of final state
            self.add_state(number=curr_state_num)
            final_state_num = curr_arc.to_state
            curr_arc.to_state = self.states[curr_state_num]

            # add an arc from the newly created state to 
            # the final state with input
            self.add_arc(from_state=curr_state_num,\
                to_state=final_state_num,\
                input_label=disambig_sym+str(disambig_num),\
                output_label=EPSILON_LABEL)

            # # make a new intermediary state, make current
            # # arc point to that instead of final state
            # self.add_state(number=curr_state_num)
            # final_state_num = curr_arc.to_state
            # curr_arc.to_state = self.states[curr_state_num]

            # # add an arc from the newly created state to 
            # # the final state with input
            # self.add_arc(from_state=curr_state_num,\
            #     to_state=final_state_num,\
            #     input_label=disambig_sym+str(disambig_num),\
            #     output_label=EPSILON_LABEL)

            # increment counters
            curr_state_num += 1
            disambig_num += 1

    def get_arcs_with_from_state(self,from_state_num):
        """ return the arc with the specified from state number """

        from_arcs = []
        for arc in self.arcs:
            if arc.from_state == from_state_num:
                from_arcs.append(arc)
        return from_arcs

    def substitute_disambig_symbol(self,\
        disambig_sym,\
        subs_sym=EPSILON_LABEL):
        """ Add disambiguation symbol to melodic sequences that map to different
        chords """

        self.load_from_compiled()
        # self.sub_disambig_symbol(\
        #     disambig_sym=disambig_sym,\
        #     subs_sym=subs_sym)
        for i,line in enumerate(self.arcs):
            if disambig_sym in line.input_label:
                self.arcs[i].input_label = subs_sym
            if disambig_sym in line.output_label:
                self.arcs[i].output_label = subs_sym
        self.compile()
        self.clear_text_fmt()

    def load_from_compiled(self,fst_filename=None):
        """ load the fst information from the compiled .fst """

        self.clear_text_fmt()
        text = self.to_text(fst_filename=fst_filename)
        # self.data_openfst_fmt = FSTData()
        # self.data_openfst_fmt.text_to_fstdata(text=text)
        self.text_to_fstdata(text=text)

    def clear_text_fmt(self):
        """ clear the OpenFST text format when not needed to save memory """

        del self.arcs
        self.arcs = []
        del self.states
        self.states = {}

    def replace_infinite_weights(self,weight=99.0,infinity_txt='Infinity'):
        """ Replace any 'Infinity' in weights with default weight value """

        self.load_from_compiled()

        for i,line in enumerate(self.arcs):
            if line.weight == infinity_txt:
                self.arcs[i].weight = weight

        self.compile()
        self.clear_text_fmt()

    def build_m2c(self,do_closure=True,do_determinize_twice=True):
        """ run the sequence of commands for building
        the melody-to-chord FST """

        print 'building melody->chord FST'
        self.compile()
        self.map('to_standard')
        self.determinize()
        if do_determinize_twice:
            print 'running determinize() twice!'
            self.determinize()  # TODO: is this necessary?
        self.minimize()
        self.map('to_log')
        self.arcsort()
        if do_closure:
            self.closure()

    def read_from_text_file(self, text_file):
        """
        Read in fst text file as FST class
        """

        text = U.read_text_file(text_file)
        self.text_to_fstdata(text=text)

    def text_to_fstdata(self,text):
        """ read in lines of text (text FST format),
        convert to FSTData representation """

        for line in text:
            tokens = line.split()
            n = len(tokens)

            from_state = int(tokens[0])

            if n == 1:
                # final_state = [ (int(from_state),None) ]
                # self.add_final_state(final_state)
                self.states[int(from_state)].is_final = True
                # from_state_n = int(from_state)
                # if not from_state_n in self.states:
                #     self.add_state(number=from_state_n)
                # self.states[from_state_n].is_final = True

            elif n ==2:
                # final_weight = float(tokens[1])
                # self.add_final_state( [(int(from_state), final_weight)] )
                self.states[int(from_state)].is_final = True
                self.states[int(from_state)].weight = float(tokens[1])
            else:
                to_state = int(tokens[1])
                input_label = tokens[2]
                output_label = ''
                weight = None

                if n == 5:
                    output_label = tokens[3]
                    weight = tokens[4]
                elif n == 4:
                    output_label = tokens[3]

                self.add_arc(from_state=from_state,to_state=to_state,\
                    input_label=input_label,output_label=output_label,\
                    weight=weight)
                # print 'adding:',from_state
                self.add_state(number=from_state)
                # print 'adding:',to_state
                self.add_state(number=to_state)

    # methods for definition file construction.
    def get_text_format(self):
        """
        Return the FST definition in text format
        (i.e. OpenFST definition file format)
        """
        fmt = [ line.to_text() for line in self.arcs ]
        for state in self.states.values():
            if not state.is_final:
                continue
            if state.weight is None:
                final_weight = ''
            else:
                final_weight = state.weight
            fmt.append(str(state.number) + '\t' + str(final_weight))
        return fmt

    def print_fst(self):
        """ Print the FST """

        for line in self.get_text_format():
            print line

    def add_state(self,number,\
        weight=None,\
        is_final=False):

        """ Add a state to the FST """

        #TODO:rewrite this!!!!!!!!!!!!!!!!! ************

        number = int(number)

        # state has already been added
        if number in self.states:
            # print '*** state number:',number,'has already been added. ***'
            return

        self.states[number] = State(number=number,\
            weight=weight,is_final=is_final)

    def add_arc(self,\
        from_state,\
        to_state,\
        input_label,\
        output_label,\
        weight=None):

        """
        Add an edge to the FST.
        """

        self.arcs.append(Arc(from_state=from_state, \
            to_state=to_state, \
            input_label=input_label, \
            output_label=output_label, \
            weight=weight))

        self.add_state(number=from_state)
        self.add_state(number=to_state)

    def get_input_symbols(self):
        """
        Return a list of the input symbols.

        returns:
            input_syms (list of strings): list of input symbols
        """

        input_syms = set([ l.input_label for l in self.arcs ])
        return list(input_syms)

    def get_output_symbols(self):
        """
        Return a list of the input symbols.

        returns:
            output_syms (list of strings): list of input symbols
        """

        output_syms = set([ l.output_label for l in self.arcs ])
        return list(output_syms)

    def get_input_output_syms(self):
        """ Get the sequences of input and output symbols
        defined by this FST in from state order.

        Returns
        -------
        input_syms: list of strings
            list of the input symbols
        output_syms: list of strings
            list of the output symbols
        """

        self.topsort()
        self.compile()
        self.load_from_compiled()

        input_syms = []
        output_syms = []

        # TODO: find a more efficient way to do this
        for n in range(self.num_states()):
            for arc in self.arcs:
                if arc.from_state == n:
                    input_syms.append(arc.input_label)
                    output_syms.append(arc.output_label)

        return input_syms,output_syms

    #  -------------------- OpenFST functions --------------------------

    def draw(self,image_filename=None,\
        format='ps'):
        """ Draw the FST and save to a file """

        if image_filename is None:
            image_filename = U.filebase(filepath=self.filename)
            image_filename = image_filename + '.' + format

        cmd = [os.path.join(FST_BIN_DIR,'fstdraw')]
        # cmd.append('-portrait')
        cmd.append(self.filename)
        cmd.append('|')
        cmd.append('dot')
        cmd.append('-T'+format)
 
        print cmd
        stdout,stderr = CU.launch_process(command=cmd,\
            outfile_name=image_filename,\
            outfile_type='binary')
        if not stderr == '':
            print 'ERROR: unable to draw',image_filename
            print 'command:',' '.join(cmd)
            print 'error:',stderr
            return None


    def print_it(self):
        """ print the fst definition """
        txt = self.to_text()
        for l in txt:
            print l.strip()

    def to_text(self,fst_filename=None):
        """ get the contents of the .fst in text format """

        if fst_filename is None:
            fst_filename = self.filename

        # check files
        errormsg = U.file_check(filename=fst_filename)
        if errormsg is not None:
            print 'ERROR reading fst file:',fst_filename
            print errormsg
            return None

        if self.isyms_table is not None:
            errormsg = U.file_check(\
                filename=self.isyms_table.filename)
            if errormsg is not None:
                print 'ERROR reading isyms file:',self.isyms_table.filename
                print errormsg
                # return None

        if self.osyms_table.filename is not None:
            errormsg = U.file_check(filename=\
                self.osyms_table.filename)
            if errormsg is not None:
                print 'ERROR reading osyms file:',self.osyms_table.filename
                print errormsg
                # return None

        temp_textfile = marl.fileutils.temp_file('txt')

        cmd = [os.path.join(FST_BIN_DIR,'fstprint')]
        if not self.isyms_table.filename is None:
            cmd.append('--isymbols=' + self.isyms_table.filename)
        if not self.is_acceptor and not self.osyms_table.filename is None:
            cmd.append('--osymbols=' + self.osyms_table.filename)
        else:
            cmd.append('--acceptor')
        cmd.append(fst_filename)
        cmd.append(temp_textfile)

        stdout,stderr = CU.launch_process(command=cmd)

        if not stderr == '':
            print 'ERROR: unable to print',fst_filename
            print 'command:',' '.join(cmd)
            print 'error:',stderr
            return None

        data = U.read_text_file(fname=temp_textfile)
        return data

    def compile(self):
        """ compile into a .fst file. Note that the data should be formatted
        properly for fst format """

        # write fst def to temp file
        temp_textfile = marl.fileutils.temp_file('txt')
        # print '\n',temp_textfile,'\n'
        U.write_text_file(data=self.get_text_format(), \
            fname=temp_textfile)

        cmd = [os.path.join(FST_BIN_DIR,'fstcompile')]

        if self.is_acceptor:
            cmd.append('--acceptor')

        cmd.append('--arc_type='+self.arc_type)
        if not self.isyms_table.filename is None:
            cmd.append('--isymbols='+self.isyms_table.filename)
        if not self.osyms_table.filename is None:
            cmd.append('--osymbols='+self.osyms_table.filename)

        if self.keep_isyms:
            cmd.append('--keep_isymbols=1')
        else:
            cmd.append('--keep_isymbols=0')

        if self.keep_osyms:
            cmd.append('--keep_osymbols=1')
        else:
            cmd.append('--keep_osymbols=0')

        cmd.append(temp_textfile)
        cmd.append(self.filename)

        # print 'COMMAND:',cmd

        stdout, stderr = CU.launch_process(command=cmd)

        if not stderr == '':
            print 'ERROR: unable to compile'
            print 'command:',' '.join(cmd)
            print 'error:',stderr
            return

    def encode(self,encode_labels=True,\
        encode_weights=False,\
        encoder_filename = 'encoder',\
        in_fst=None,\
        out_fst=None):
        """ encode an fst """

        args = []

        if encode_labels:
            args.append('--encode_labels=1')
        if encode_weights:
            args.append('--encode_weights=1')

        if in_fst is None:
            in_fst = self.filename

        if out_fst is None:
            out_fst = marl.fileutils.temp_file('fst')
            copy_fst = True
        else:
            copy_fst = False

        # cmd = ['fstencode']
        cmd = [os.path.join(FST_BIN_DIR,'fstencode')]

        cmd.extend(args)
        cmd.append(in_fst)
        cmd.append(encoder_filename)

        stdout, stderr = CU.launch_process(command=cmd,\
            outfile_name=out_fst,\
            outfile_type='binary')

        if not stderr == '':
            print 'ERROR: unable to execute',cmd
            print 'command:',' '.join(cmd)
            print 'error:',stderr
            return

        if copy_fst:
            shutil.copy2(out_fst,self.filename)

        self.is_encoded = True
        # self.load_from_compiled(fst_filename=self.filename)

    def decode(self,encoder_filename='encoder',in_fst=None,out_fst=None):
        """ encode an fst """

        args = '--decode=1'

        if in_fst is None:
            in_fst = self.filename

        if out_fst is None:
            out_fst = marl.fileutils.temp_file('fst')
            copy_fst = True
        else:
            copy_fst = False

        # cmd = ['fstencode']
        cmd = [os.path.join(FST_BIN_DIR,'fstencode')]
        cmd.append(args)
        cmd.append(in_fst)
        cmd.append(encoder_filename)

        stdout, stderr = CU.launch_process(command=cmd,\
            outfile_name=out_fst,\
            outfile_type='binary')

        if not stderr == '':
            print 'ERROR: unable to execute',cmd
            print 'command:',' '.join(cmd)
            print 'error:',stderr
            return

        if copy_fst:
            shutil.copy2(out_fst,self.filename)

        self.is_encoded = False

        self.load_from_compiled(fst_filename=self.filename)

    def rmepsilon(self,in_fst=None,out_fst=None):
        """ remove epsilons from an fst
        if in_fst is None, use .fst pointed to by self.filename.
        this .fst will be replaced by the determinized .fst.
        """

        self.run_openfst_command(command='fstrmepsilon',\
            in_fst=in_fst,out_fst=out_fst)

    def determinize(self,in_fst=None,out_fst=None):
        """ determinize an fst
        if in_fst is None, use .fst pointed to by self.filename.
        this .fst will be replaced by the determinized .fst.
        """

        self.run_openfst_command(command='fstdeterminize',\
            in_fst=in_fst,out_fst=out_fst)

    def minimize(self,in_fst=None,out_fst=None):
        """ minimize an fst
        if in_fst is None, use .fst pointed to by self.filename.
        this .fst will be replaced by the minimized .fst.
        """

        self.run_openfst_command(command='fstminimize',\
            in_fst=in_fst,out_fst=out_fst)

    def closure(self,in_fst=None,out_fst=None,allow_empty_path=False):
        """ create Kleene closure on an fst
        if in_fst is None, use .fst pointed to by self.filename.
        this .fst will be replaced by the minimized .fst.
        """

        args = None

        if not allow_empty_path:
            args = '--closure_plus=1'

        self.run_openfst_command(command='fstclosure',\
            in_fst=in_fst,out_fst=out_fst,opt_args=args)

    def topsort(self,in_fst=None,out_fst=None):
        """ topologically sort fst
        if in_fst is None, use .fst pointed to by self.filename.
        this .fst will be replaced by the minimized .fst.
        """

        self.run_openfst_command(command='fsttopsort',\
            in_fst=in_fst,out_fst=out_fst)

    def map_to_log(self,in_fst=None,out_fst=None):
        """ convenience method to map to log arc type """

        self.map(map_type='to_log',in_fst=in_fst,out_fst=out_fst)

    def map_to_standard(self,in_fst=None,out_fst=None):
        """ convenience method to map to standard arc type """

        self.map(map_type='to_standard',in_fst=in_fst,out_fst=out_fst)

    def map(self,map_type='to_standard',in_fst=None,out_fst=None):
        """ Apply an operation to each arc of an FST
        if in_fst is None, use .fst pointed to by self.filename.
        this .fst will be replaced by the minimized .fst.
        """

        if map_type == 'to_log':
            self.arc_type = 'log'
        elif map_type == 'to_standard':
            self.arc_type = 'standard'
        else:
            print 'UNKNOWN MAP TYPE:',map_type

        args = '--map_type=' + map_type
        self.run_openfst_command(command='fstmap',\
            in_fst=in_fst,out_fst=out_fst,opt_args=args)

    def arcsort(self,sort_type='ilabel',in_fst=None,out_fst=None):
        """ Sort arcs of an FST
        if in_fst is None, use .fst pointed to by self.filename.
        this .fst will be replaced by the minimized .fst.
        """

        args = '--sort_type=' + sort_type
        self.run_openfst_command(command='fstarcsort',\
            in_fst=in_fst,out_fst=out_fst,opt_args=args)

    def run_openfst_command(self,command,in_fst=None,\
        out_fst=None,opt_args=None):
        """ run a simple OpenFST operation (e.g. minimize, determinize, etc.)
        if in_fst is None, use .fst pointed to by self.filename.
        this .fst will be replaced by the processed .fst.
        """

        command = os.path.join(FST_BIN_DIR,command)

        if in_fst is None:
            in_fst = self.filename

        if out_fst is None:
            out_fst = marl.fileutils.temp_file('fst')
            copy_fst = True
        else:
            copy_fst = False

        if opt_args is not None:
            if type(opt_args) == list:
                cmd = [command]
                cmd.extend(opt_args)
                cmd.append(in_fst)
                cmd.append(out_fst)
            else:
                cmd = [command,opt_args,in_fst,out_fst]
        else:
            cmd = [command,in_fst,out_fst]

        stdout, stderr = CU.launch_process(command=cmd)

        if not stderr == '':
            print 'ERROR: unable to execute',command
            print 'command:',' '.join(cmd)
            print 'error:',stderr
            return

        if copy_fst:
            shutil.copy2(out_fst,self.filename)


class BranchFST(FST):
    """ Class that represents a useful type of FST, a "branch",
    which has a separate branch for each input/output symbol
    sequence. This is used, for example, for building the
    phoneme-to-word transducer (L).
    """

    def __init__(self,\
        filename,\
        isyms_table,\
        osyms_table,\
        keep_isyms=True,\
        keep_osyms=True,\
        arc_type='log',\
        is_acceptor=False,\
        overwrite_isyms_file=True):
        """ Constructor

        Parameters
        ----------
        filename: string
            FST file name
        isyms_table: SymbolTable object
            input symbols
        osyms_table: SymbolTable object
            output symbols
        keep_isymbols: bool
            keep input symbols table
        keep_osymbols: bool
            keep output symbols table
        arc_type: string
            one of 'standard' or 'log'
        is_acceptor: bool
            is this FST an acceptor or not
        initial_state: int
            the number corresponding to the initial state
        """

        FST.__init__(self,\
            filename=filename,\
            isyms_table=isyms_table,\
            osyms_table=osyms_table,\
            keep_isyms=keep_isyms,\
            keep_osyms=keep_osyms,\
            arc_type=arc_type,\
            is_acceptor=is_acceptor,\
            overwrite_isyms_file=overwrite_isyms_file)

    def build(self,\
        input_seqs,\
        output_seqs,\
        weights,\
        make_factor_transducer=True,\
        add_epsilons=True):
        """ Build the FST using the provided lists of input and
        output symbols, and weights.

        Parameters
        ----------
        input_symbols: list of lists of strings
            list of lists of input symbols
        output_symbols: list of lists of strings
            list of lists of output symbols
        weights: list of floats
            list of weights for each arc. these are
            initial weights for each input/output
            sequence
        """
        
        assert len(input_seqs) == len(output_seqs) == len(weights)

        # find final state number
        final_state_num = 1
        for seq in input_seqs:
            final_state_num = final_state_num + len(seq) + 1

        initial_state_num = 0

        self.add_state(number=initial_state_num)
        self.add_state(number=final_state_num)
        self.states[final_state_num].is_final = True

        curr_n = initial_state_num

        for input_seq,output_seq,weight in zip(input_seqs,output_seqs,weights):

            if add_epsilons:
                input_seq.append(EPSILON_LABEL)
                output_seq.append(EPSILON_LABEL)

            # print input_seq,output_seq
            assert len(input_seq) == len(output_seq)
            seq_len = len(input_seq)

            # initial_weight = 1.0
            # if seq_len<4:
            #     initial_weight *= (2.0*seq_len)

            for i in range(seq_len):

                if i == 0:
                    from_state = initial_state_num
                    to_state = curr_n+1
                    weight = weight #initial_weight
                    # curr_n += 1
                elif i == seq_len - 1:
                    from_state = curr_n
                    to_state = final_state_num
                    # if i>0 and data.DISAMBIG_SYMBOL in str(input_seq[i-1]):
                    #     weight = 10.0
                    # else:
                    weight = None
                else:
                    from_state = curr_n
                    to_state = curr_n + 1
                    weight = None
                    # curr_n += 1

                in_lab = str(input_seq[i])      # could be unicode
                out_lab = str(output_seq[i])    # could be unicode

                self.add_arc(from_state=from_state,to_state=to_state,\
                    input_label=in_lab,output_label=out_lab,weight=weight)

                curr_n += 1


class FlowerFST(FST):
    """ Class that represents a useful type of FST, a "flower",
    which has only one state (state number 0), with each arc
    going to and from that state.
    """

    def __init__(self,\
        filename,\
        isyms_table,\
        osyms_table,\
        keep_isyms=True,\
        keep_osyms=True,\
        arc_type='log',\
        is_acceptor=False,\
        overwrite_isyms_file=True):
        """ Constructor

        Parameters
        ----------
        filename: string
            FST file name
        isyms_table: SymbolTable object
            input symbols
        osyms_table: SymbolTable object
            output symbols
        keep_isymbols: bool
            keep input symbols table
        keep_osymbols: bool
            keep output symbols table
        arc_type: string
            one of 'standard' or 'log'
        is_acceptor: bool
            is this FST an acceptor or not
        initial_state: int
            the number corresponding to the initial state
        """

        FST.__init__(self,\
            filename=filename,\
            isyms_table=isyms_table,\
            osyms_table=osyms_table,\
            keep_isyms=keep_isyms,\
            keep_osyms=keep_osyms,\
            arc_type=arc_type,\
            is_acceptor=is_acceptor,\
            overwrite_isyms_file=overwrite_isyms_file)

    def build(self,\
        input_symbols,\
        output_symbols,\
        weights=None,\
        initial_state_num=0):
        """ Build the FST using the provided lists of input and
        output symbols, and, optionally, weights.

        Parameters
        ----------
        input_symbols: list of strings
            list of input symbols
        output_symbols: list of strings
            list of output symbols
        weights: list of floats
            list of weights for each arc
        initial_state_num: int
            number of initial state
        """

        assert len(input_symbols) == len(output_symbols)

        if weights is None:
            weights = len(input_symbols) * [None]

        self.add_state(number=0)
        for ilabel,olabel,weight in zip(input_symbols,output_symbols,weights):
            self.add_arc(from_state=0,to_state=0,\
                input_label=ilabel,output_label=olabel,weight=weight)
        self.states[0].is_final = True


class LinearChainFST(FST):
    """ Class that represents a useful type of FST, a "linear chain",
    which looks something like this: 0 -> 1 -> 2 -> 3 (where numbers
    represent states, "->" represent arcs)
    """

    def __init__(self,\
        filename,\
        isyms_table,\
        osyms_table=None,\
        keep_isyms=True,\
        keep_osyms=True,\
        arc_type='log',\
        is_acceptor=False,\
        overwrite_isyms_file=True):
        """ Constructor

        Parameters
        ----------
        filename: string
            FST file name
        isyms_table: SymbolTable object
            input symbols
        osyms_table: SymbolTable object
            output symbols
        keep_isymbols: bool
            keep input symbols table
        keep_osymbols: bool
            keep output symbols table
        arc_type: string
            one of 'standard' or 'log'
        is_acceptor: bool
            is this FST an acceptor or not
        initial_state: int
            the number corresponding to the initial state
        """

        if osyms_table is None:
            osyms_table = isyms_table

        FST.__init__(self,\
            filename=filename,\
            isyms_table=isyms_table,\
            osyms_table=osyms_table,\
            keep_isyms=keep_isyms,\
            keep_osyms=keep_osyms,\
            arc_type=arc_type,\
            is_acceptor=is_acceptor,\
            overwrite_isyms_file=overwrite_isyms_file)

    def build(self,\
        input_symbols,\
        output_symbols=None,\
        weights=None,\
        initial_state_num=0):
        """ Build the FST using the provided lists of input (and possibly
        output) symbols.

        Parameters
        ----------
        input_symbols: list of strings
            list of input symbols
        output_symbols: list of strings
            list of output symbols
        weights: list of floats
            list of weights for each arc
        initial_state_num: int
            number of initial state
        """

        if output_symbols is None:
            output_symbols = input_symbols

        if weights is None:
            weights = [None] * len(input_symbols)

        assert len(input_symbols) == len(output_symbols) == len(weights)

        state_num = initial_state_num

        for isym,osym,w in zip(input_symbols,output_symbols,weights):
            self.add_arc(from_state=state_num,\
                to_state=state_num+1,\
                input_label=isym,\
                output_label=osym,\
                weight=w)
            state_num += 1

        self.states[state_num].is_final = True
        self.compile()

    def total_weight(self):
        """ Return the total weight of the FST.
        If weights are negative log probabilites, we 
        should be able to just add the weight associated
        with each arc.

        Returns
        -------
        total_wt: float
            Weight of all the paths
        """

        total_wt = 0.0
        for a in self.arcs:
            if not a.weight is None:
                total_wt += float(a.weight)

        return total_wt


class SymbolTable(object):
    """ A class to encapsulate OpenFST symbol table. """

    def __init__(self,filename,symbols=None):
        """ Constructor.

        Parameters
        ----------
        filename: string
            name of associated file
        symbols: list of strings
            list of symbols

        """

        self.filename = filename
        self._symbols = []

        # we always need an epsilon symbol...
        if not symbols is None:
            self.set_symbols(symbols=symbols)

    def set_symbols_from_seq(self,sym_seqs):
        """ convenience method to get the set of
        symbols from symbol sequence(s), and
        set the symbols property from this set.

        Parameters
        ----------
        sym_seqs: list of strings or list of lists of strings
            either a list of symbols, or a list of lists
            of symbols

        """

        if type(sym_seqs[0]) == list:
            sym_seqs = [ v for subseq in sym_seqs for v in subseq ]

        symbols = list(set(sym_seqs))
        self.set_symbols(symbols=symbols)

    def get_symbols(self):
        """ Get the list of symbols

        Returns
        -------
        symbols: list of strings
            list of symbols
        """

        return self._symbols

    def set_symbols(self,symbols,add_epsilon_label=True):
        """ Set the symbols property

        Parameters
        ----------
        symbols: list of strings
            list of symbols

        """

        symbols = list(set(symbols))
        symbols.sort()

        # if add_epsilon_label and not EPSILON_LABEL in symbols:
        if add_epsilon_label and not EPSILON_LABEL in symbols:
            # sym_set = set(symbols) - set(EPSILON_LABEL)
            # symbols = list(sym_set)
            symbols.sort()
            symbols.insert(0,EPSILON_LABEL)

        self._symbols = symbols

    def get_symbols_text(self):
        """ get symbols data (for writing to .syms file) """

        txt = []

        for n, s in enumerate(self._symbols):
            txt.append(s + ' ' + str(n))

        return txt

    def write_file(self):
        """ write symbols file """

        U.write_text_file(fname=self.filename,\
            data=self.get_symbols_text())

    def load_file(self):
        """ read existing symbols file and
        set symbols from data in file """

        text = U.read_text_file(fname=self.filename)
        n_syms = len(text)
        symbols = n_syms*[None]
        for t in text:
            sym,n = t.split()
            n = int(n)
            symbols[n] = sym

        self.set_symbols(symbols=symbols)


class State(object):
    """ A class that represents a state in an FST """

    def __init__(self,number,\
        weight=None,\
        is_final=False):
        """ Constructor.

        Parameters
        ----------
        number: int
            state number
        weight: float
            weight associated with this state
        is_final: boolean
            is this state final or not
        """

        self.number = number
        self.weight = weight
        # apparently, only state 0 can be the initial state in OpenFST
        self.is_initial = self.number == 0
        self.is_final = is_final


class Arc(object):
    """ A class that represents an arc in an FST, in OpenFST definition
    file format.

    Format is:
            from_state to_state input_label output_label weight
    """

    def __init__(self,\
        from_state,\
        to_state,\
        input_label,\
        output_label,\
        weight=None):
        """
        Constructor.

        Parameters
        ----------
        from_state: int
            from state number
        to_state: int
            to state number
        input_label: string
            input arc label
        output_label: string
            output arc label
        weight: float
            arc weight

        """

        self.from_state = from_state
        self.to_state = to_state
        self.input_label = input_label
        self.output_label = output_label
        self.weight = weight

    def to_text(self):
        """
        Return the line in text format for OpenFST definition file.

        Parameters
        ----------
        None

        Returns
        -------
        line: string
            line that should appear in OpenFST file.
        """

        if self.weight is None:
            w = ''
        else:
            w = str(self.weight)

        line = str(self.from_state) + ' ' + \
        str(self.to_state) + ' ' + str(self.input_label) + \
        ' ' + str(self.output_label) + ' ' + w
        return line


def compose(in1_fst,in2_fst,out_fst_filename):
    """ compose two (compiled) fst's """

    exe = os.path.join(FST_BIN_DIR,'fstcompose')
    cmd = [exe,in1_fst.filename,in2_fst.filename,out_fst_filename]
    stdout,stderr = CU.launch_process(command=cmd)

    if not stderr == '':
        print 'ERROR: unable to compose FSTs',\
            in1_fst.filename,'and',in2_fst.filename
        print 'command:',' '.join(cmd)
        print 'output:',stdout
        print 'error:',stderr
        return None

    out_fst = FST(filename=out_fst_filename,\
        isyms_table=in1_fst.isyms_table,\
        osyms_table=in2_fst.osyms_table)
    out_fst.load_from_compiled()

    return out_fst

def union(in1_fst,in2_fst,out_fst_filename):
    """ perform union of two (compiled) fst's """

    exe = os.path.join(FST_BIN_DIR,'fstunion')
    cmd = [exe,in1_fst.filename,in2_fst.filename,out_fst_filename]
    stdout,stderr = CU.launch_process(command=cmd)

    if not stderr == '':
        print 'ERROR: unable to union FSTs',\
            in1_fst.filename,'and',in2_fst.filename
        print 'command:',' '.join(cmd)
        print 'output:',stdout
        print 'error:',stderr
        return None

    out_fst = FST(filename=out_fst_filename,\
        isyms_table=in1_fst.isyms_table,\
        osyms_table=in2_fst.osyms_table)
    out_fst.load_from_compiled()

    return out_fst

def get_all_path_fsts(fst,filebase='path_fst_'):
    """ Return a list of LinearChainFSTs, each corresponding
    to a path through the specified FST.
    Note that this is designed to work only in the case that all
    paths begin at the initial state (i.e., none of the paths split,
    though they may merge)
    """

    arc_paths = get_all_paths(fst=fst)

    path_fsts = []

    for n,arc_path in enumerate(arc_paths):
        filename = filebase + str(n)
        path_fst = LinearChainFST(filename=filename,\
            isyms_table=fst.isyms_table,\
            osyms_table=fst.osyms_table)
        isyms = [a.input_label for a in arc_path]
        osyms = [a.output_label for a in arc_path]        
        weights = [a.weight for a in arc_path]
        path_fst.build(input_symbols=isyms,\
            output_symbols=osyms,\
            weights=weights)

        path_fsts.append(path_fst)

    return path_fsts

def get_all_paths(fst):
    """ return a list of all the paths through the specified FST.
    Note that this is designed to work only in the case that all
    paths begin at the initial state (i.e., none of the paths split,
    though they may merge)
    """

    paths = []
    for init_state in fst.initial_states():
        start_arcs = \
            fst.get_arcs_with_from_state(from_state_num=init_state.number)

        for start_arc in start_arcs:
            curr_path = [start_arc]
            curr_state_num = start_arc.to_state

            while not fst.states[curr_state_num].is_final:
                curr_arc = \
                    fst.get_arcs_with_from_state(from_state_num=curr_state_num)[0]
                curr_state_num = curr_arc.to_state
                curr_path.append(curr_arc)
            paths.append(curr_path)

    return paths


