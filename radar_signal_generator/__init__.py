"""
================
Radar signal Generator
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
class RadarRectParameters:
    fs: float           = None          # Sample rate
    pulse_time: float  = None          # Pulse width (s)
    prf: float          = None          # Pulse repetition frequency (f)
    nsamp: int          = None          # Number of samples in the chirp
    phase: float        = None          # Phase offset
    window: str         = None          # Windowing algorithm for smooth transitions

@dataclass
class RadarChirpParameters:
    fs: float           = None          # Sample rate
    pulse_time: float  = None          # Pulse width (s)
    prf: float          = None          # Pulse repetition frequency (f)
    bw: float           = None          # Modulation bandwith (f)
    nsamp: int          = None          # Number of samples in the chirp
    phase: float        = None          # Phase offset
    window: str         = None          # Windowing algorithm for smooth transitions

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
        
        ## Constants
        self.__c            = 299792458 # m/s

        ## Calculated values
        #self.__duty         = self.pulse_time/self.prf

        # Radar siggen attributes
        self.params = None

        # IO
        self.IOS=Bundle()
        self.IOS.Members['IQ_OUT'] = IO()

    def run(self,*arg):
        self.main()

    def main(self):
        """
        Description:
            Main method for the radar signal generator. 
            1. Generates correct signal type based in given params
            2. Windows the signal to smoothen the pulse edge transients
            3. Routes the signal to output IO
        """
        outval_IQ = None

        #pulse_repetitions = self.params.nsamp // time_as_samples(pulse_time)
        #for _ in range(pulse_repetitions)

        # Chosen signal type is assigned to outputs via outval
        match self.signal_type(): 
            case 'rect': 
              outval_IQ = self.rect()
            case 'chirp': 
              #outval_IQ = self.chirp()
              outval_IQ = self.chirp_bb()
            # Other possible waveforms: 
            #'binary phase coded', 'non-linear FM', 'discrete frequency-shift', 'polyphase codes', 'compound Barker codes', 'code sequencing', 'complementary codes', 'pulse burst', 'stretch'
            case _:
                self.print_log(type='F',msg='Signal type \'%s\' not supported.' % self.params.sigtype)
        outval_IQ = self.apply_window(outval_IQ)
        self.IOS.Members['IQ_OUT'].Data = outval_IQ

    def apply_window(self, s):
        """
        Description:
            Applies selected window type to input signal. Main purpose is to smoothen the transient 
            at the end of the pulse.
        Parameters:
            s: np.complex128    = unwindowed radar signal
        Returns:  
            s: np.compex128     = windowed signal
        """
        
        from scipy.signal.windows import tukey

        window = self.params.window

        w = np.ones_like(s)
        if window == 'no':
            w = w
        elif window == 'hamming':
            w = np.hamming(len(s))
        elif window == 'hann':
            w = np.hanning(len(s))
        elif window == 'tukey':
            # Tukey alpha ~= 2*edge_frac so that each edge gets edge_frac of the pulse
            edge_frac = 0.08    # NOTE: Use this to control tukey effectiveness
            L = int(self.time_as_samples(self.params.pulse_time))
            w_short = tukey(L, alpha=2*edge_frac)
            w = s
            w[:L] *= w_short
        else:
            self.print_log(type='W',msg='\'%s\' is not a valid windowing method. Defaulting to no window.' % self.param.window)

            s = s * w

        return s

    def rect(self):
        """
        Parameters:
        rect_config(T, t_start, N, fs)  # Rect signal attributes
        ----------
        Returns:[:64]
        x : ndar[:64]ray(dtype=complex128)   # Generated pulse waveform I/Q samples

        """
        #T, t_start, N, fs = self.T, self.t_start, self.N, self.fs
        #fs, N = self.fs, self.N
        T, t_start, N, fs = self.tfall, self.tstart, self.nsamp, self.fs

        n_high = int(T*fs)
        n_start = int(t_start*fs)
        n_end = min(N, n_start + n_high)
        x = np.zeros(N, dtype=np.complex128)
        x[n_start:n_end] = np.ones(n_end-n_start, dtype=np.complex128)

        return x

    def chirp(self):
        """
        Description:
            Chirp generation with scipy (not in use but here just in case)
        Returns:
            chirp_pulse: np.complex128 = generated chirp signal
        """
        import scipy

        fs = self.params.fs
        f0 = 0
        T = self.params.pulse_time 
        B = self.params.bw
        N = self.params.nsamp
        phi0=-np.pi/2

        t = np.arange(N) * T
        n_end = int(T*fs)
        t = np.arange(n_end)/fs

        f1 = f0 -B/2
        f2 = f0 + B/2

        chirp_pulse = np.zeros(N, dtype=np.complex128)
        chirp_pulse[:n_end] = scipy.signal.chirp(t, f1, T, f2, method='linear', phi=phi0, complex=True)

        return chirp_pulse

    def chirp_bb(self):
        """
        Description:
            Generates a base-band chirp signal based in configured params.
        Returns:
            chirp_pulse: np.complex128 = generated chirp signal
        """
        fs = self.params.fs
        f0 = 0
        T = self.params.pulse_time 
        B = self.params.bw
        N = self.params.nsamp
        phi=-np.pi/2

        t = np.arange(0, T, 1/fs)
        mu = B / T                      # chirp rate
        #f_start =0 #max(f0, 1e-9)         # start just above 0 for one-sided baseband
        #phase = 2*np.pi*(f_start*t + 0.5*mu*t**2) + phi
        phase = 2*np.pi*(f0*t + 0.5*mu*t**2) + phi
        s = np.exp(1j * phase)          # analytic (I/Q) chirp
        out = np.zeros(N, dtype=np.complex128)
        out[:len(s)] = s[:N]

        return out

    def signal_type(self):
        sig_type = None
        if isinstance(self.params, RadarRectParameters): return 'rect'
        elif isinstance(self.params, RadarChirpParameters): return 'chirp'
        else: return None

    # ----- Equations and Calculations ----- #
    def time_as_samples(self, time_interval): 
        """
        Description:
            Calculates number of samples in any given time interval 
        Param: 
            time_interval: float = any time frame (s)
        Returns:
            samples: int = number of samples
        """
        samples = int(time_interval * self.params.fs)
        return samples

    def repetition_time(self):
        """
        Description:
            How much time does one pulse repetition cycle take
        Returns:
            repetition_time: float = number of samples
        """
        return 1/self.params.prf

    def max_signal_range(self):
        """
        Description:
            Calculates the maximum range 
        Returns:
            radar_range: int = number of samples
        """
        return self.__c * (self.repetition_time() - self.params.pulse_time)

    def osr(self): 
        """
        Description:
            Calculates the oversampling ratio from highest frequency and sample rate
        Returns:
            osr: float = number of samples
        """
        return self.params.fs / (2*self.params.bw)

    def range_resolution(self):
        """
        Description:
            Calculates the range resolution (the maximum detectable difference between 
            multiple objects/reflections)
        Returns:
            range_resolution: float = in meters
        """
        return self.__c / (2*self.params.bw)

    def min_signal_range(self):
        """
        Description:
          Calculates 
        Returns:
        """
        return self.__c * self.params.pulse_time / 2
        
    def signal_time(self):
        """
        Description:
          Calculates full length of the generated signal in seconds
        Returns:
          signal_time = signal time in seconds
        """
        return self.params.nsamp / self.params.fs

    # ----- Print Info ----- #
    def print_signal_specs(self):
        """
        Description:
            Prints a table of all information and specs that can be determined from from the 
            generated signal.
        """
        width_1 = 25
        width_2 = 20
        full_width = width_1 + width_2 + 7
        print('-' * full_width)
        print(f"| {'Parameter':<{width_1}} | {'Value':>{width_2}} |")
        print("=" * full_width)
        print(f"| {'Signal type':<{width_1}} | {self.signal_type():>{width_2}} |")
        print('-' * full_width)
        print(f"| {'Sample rate [Hz]':<{width_1}} | {self.params.fs:>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'Generated Signal Time [s]':<{width_1}} | {self.signal_time():>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'PRF [Hz]':<{width_1}} | {self.params.prf:>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'Pulse Repetition Time [s]':<{width_1}} | {self.repetition_time():>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'Pulse time [s]':<{width_1}} | {self.params.pulse_time:>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'Max signal range [m]':<{width_1}} | {self.max_signal_range():>{width_2}.2f} |")
        print('-' * full_width)
        print(f"| {'Min signal range [m]':<{width_1}} | {self.min_signal_range():>{width_2}.2f} |")
        print('-' * full_width)
        print(f"| {'Range resolution [m]':<{width_1}} | {self.range_resolution():>{width_2}.2f} |")
        print('-' * full_width)
        print(f"| {'Bandwidth [Hz]':<{width_1}} | {self.params.bw:>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'Oversampling ratio':<{width_1}} | {self.osr():>{width_2}.2f} |")
        print('-' * full_width)
        print(f"| {'Samples':<{width_1}} | {self.params.nsamp:>{width_2}d} |")
        print('-' * full_width)
        print(f"| {'Initial phase [rad]':<{width_1}} | {self.params.phase:>{width_2}.2f} |")
        print('-' * full_width)
        print(f"| {'Window':<{width_1}} | {self.params.window:>{width_2}} |")
        print('-' * full_width)

if __name__=="__main__":
    from  radar_signal_generator import *
    from  iq_plotting import *
    from  dsp_toolkit import *
    import pdb
    import matplotlib.pyplot as plt

    test_cases = {
        'chirp_test':
        RadarChirpParameters(
              fs            = 400e6,      # Sample rate
              pulse_time   = 1e-6,        # Length of the radar pulse in seconds
              prf           = 14e3,       # Pulse repetition frequency
              bw            = 100e6,       # Bandwidth of the chirp linear frequency modulation
              nsamp         = 2**10,      # Number of samples in the output (total length of the generated signal)
              phase         = np.pi/2,    # Phase offset
              window        = 'tukey',    # Windowing algorithm for smooth transients
        )
    }

    chosen_test_case = 'chirp_test'
    
    dut = radar_signal_generator()
    dut.params = test_cases[chosen_test_case]
    dut.run()

    r = np.complex128(dut.IOS.Members['IQ_OUT'].Data)
       
    # Genearte header file
    # TODO: Adjust scaling coefficient for header generation
    header_scaling_coeff = 65536
    r_scaled = scale_dbm(r, -30)
    if True: 
        dsp_toolkit.generate_header(
            r,
            I = r_scaled.real*header_scaling_coeff, 
            Q = r_scaled.imag*header_scaling_coeff, 
            header_file_path = "generated_header.h"
       )
    
    plot_simple(r, dut.params.fs, label='Chirp')
    #plot_spectrum(r, dut.params.fs, title='Chirp')
    plot_bb_spectrum(r, 400e6, scale='dbfs', window='rect', color='black', ylim=[-100,10])

    dut.print_signal_specs()
    

    ## time-domain
    #plt.figure()
    #plt.plot(r_scaled.real, 'blue')
    #plt.plot(r_scaled.imag, 'red')
    #plt.legend(['I','Q'])
    ## spectrum
    #plt.figure()
    #plt.grid()
    #plt.ylabel("dBm")
    #plot_bb_spectrum(r_scaled, dut.params.fs, scale='v2dbm', window='rect', color='black', ylim=[-70,10])

    # Show plot
    plt.show()
    
    input()
