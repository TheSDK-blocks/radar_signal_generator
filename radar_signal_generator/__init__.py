"""
================
Signal Generator
================

Multi-purpose signal generator for The SyDeKick. Can generate sine waves and
pulse-shape signals.

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

import numpy as np
import scipy.signal as scsig
import tempfile
import pdb

from dataclasses import dataclass

from thesdk import *
from vhdl import *

@dataclass
class rect_attrib:
    T       : float      # Pulse width [s]      
    t_start : float      # Pulse start time [s]
    N       : int        # Numbe of samples
    fs      : float      # Sample rate [smaples/s]

@dataclass
class chirp_attrib:
    T       : float      # Pulse width [s]
    B       : float      # Sweep bandwidth [s]
    N       : int        # Number of samples [samples/s]

class radar_signal_generator(thesdk):
    """
    Attributes:
    TODO

    """

    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = ['sig_freq','lo_freq','sig_amp','sig_cm','coherent','tau','sig_phase','sig_osr','nsamp','fs','extra_sampl', 'nonoverlap_period']
        self.sig_freq = 1e6
        self.lo_freq = 0
        self.sig_amp = 0.5
        self.tau = 0
        self.sig_cm = 0
        self.sig_phase = 0
        self.sig_osr = 1
        self.nsamp = 1024
        self.extra_sampl = 0
        self.fs = 2e9
        self.snr = 0
        self.jitter_sd = None
        self.coherent = False
        self.sigtype = 'sine'
        self.high = 1
        self.low = 0
        self.after = 0
        self.duty = 0.5
        self.trise = 5e-12
        self.tfall = 5e-12
        self.nonoverlap_period = 40e-12
        self.slopetype='rising'
        self.IOS=Bundle()
        self.IOS.Members['out']= IO()
        self.model='py'
        self.init()

        self.signal_type = None
        self.T           = 5e-1
        self.t_start     = 1e-1
        self.N           = 1024
        self.fs          = 1e3
        self.B           = 5e2

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

    def init(self):
        pass

    def run(self,*arg):
        if self.model=='py':
            return self.main()

        else: 
            pass

    def main(self):
        """
        Generates chosen signal type and returns it as "ndarray[np.complex128]" type.
        """
        # TEMPORARY PARAMETER DEFINITIONS (replace with self.config)
        rect_config     = rect_attrib(T=1e-4, t_start=1e-4, N=1024, fs=1e6)
        chirp_config    = chirp_attrib(T=1e-4, B=100, N=1024)
                
        match self.signal_type: 
            case 'rect':
                return self.rect()
            case 'chirp':
                return self.chirp()
            # TODO: case 'binary_coded_PSK': 1e-3
            case _:
                self.print_log(type='F',msg='Signal type \'%s\' not supported.' % self.signal_type)
                return None

    def rect(self):
        """
        Parameters:
        rect_config(T, t_start, N, fs)  # Rect signal attributes
        ----------
        Returns:[:64]
        x : ndar[:64]ray(dtype=complex128)   # Generated pulse waveform I/Q samples

        """
        T, t_start, N, fs = self.T, self.t_start, self.N, self.fs

        fs, N = self.fs, self.N

        n_high = int(T*fs)
        n_start = int(t_start*fs)
        n_end = min(N, n_start + n_high)
        x = np.zeros(N, dtype=np.complex128)
        x[n_start:n_end] = np.ones(n_end-n_start, dtype=np.complex128)
        
        return x

    def chirp(self):
        """
        Parameters:
        chirp_config(T, B, N)  # Rect signal attributes
        ----------
        Returns:
        x : ndarray(dtype=complex128)   # Generated pulse waveform I/Q samples
        """
        import scipy

        #N, T = 1024, 0.01
        #chirp_pulse = scipy.signal.chirp(t, f0=0, f1=10, t1=10, method='linear', complex=True)

        T, B, N = self.T, self.B, self.N
        fs, Nr = self.fs, self.N

        t = np.arange(N) * T
        n_end = int(T*fs)
        t = np.arange(n_end)/fs

        f1 = -B/2
        f2 = B/2

        chirp_pulse = np.zeros(Nr, dtype=np.complex128)
        chirp_pulse[:n_end] = scipy.signal.chirp(t, f1, T, f2, method='linear', complex=True)
        #chirp_pulse[:n_end].real = scipy.signal.chirp(t, f1, T, f2, method='linear', phi=90)
        #chirp_pulse[:n_end].imag = scipy.signal.chirp(t, f1, T, f2, method='linear', phi=0)

        return chirp_pulse


if __name__=="__main__":
    from  radar_signal_generator import *
    from  iq_plotting import *
    from  dsp_toolkit import *
    import pdb
    import matplotlib.pyplot as plt
    
    # Instantiate dut
    dut = radar_signal_generator()
    
    # Define test params
    dut.signal_type = "chirp"
    #dut.signal_type = "rect"

    # Set model and run
    dut.model = 'py'
    x = dut.run()
       
    n = dut.N
    fs = dut.fs

    # Genearte header file
    # TODO: Adjust scaling coefficient for header generation
    header_scaling_coeff = 65536
    x_scaled = scale_dbm(x, -30)
    if True: 
        dsp_toolkit.generate_header(
            x,
            I = x_scaled.real*header_scaling_coeff, 
            Q = x_scaled.imag*header_scaling_coeff, 
            header_file_path = "generated_header.h"
       )

    x_scaled = scale_dbm(x, -30)

    # time-domain
    plt.figure()
    plt.plot(x_scaled.real, 'blue')
    plt.plot(x_scaled.imag, 'red')
    plt.legend(['I','Q'])
    # spectrum
    plt.figure()
    plt.grid()
    plt.ylabel("dBm")
    plot_bb_spectrum(x_scaled, dut.fs, scale='v2dbm', window='rect', color='black', ylim=[-70,10])

    # Show plot
    plt.show()
    
    input()
