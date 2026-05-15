"""
================
Radar signal Generator
================
"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk')) 
import pdb

# The SyDeKick imports
from thesdk import *
#from thesdk import IO
from plot_format                import set_style
from signal_analyser            import signal_analyser

# Python imports
import numpy as np
import scipy.constants as const
from dataclasses import dataclass
from math import floor, ceil
from scipy.signal import chirp, hilbert


# ---------- Generic signal dataclasses ---------- #
# NOTE: Use Generic* dataclasses for inheritance when creating specific signal types
@dataclass
class GenericSignalParameters:
    """ All signal parameters common to all signal types. This is inherited
    to create custom singal types.
    """
    fs: float           = None          # Sample rate
    nsamp: int          = None          # Number of samples in the signal
    snr: float          = None
    amp: float          = None          # Amplitude of the generated signal
    fc: float           = None          # Carrier frequency
    ch: list            = None          # List of channels default 0
    # NOTE: fc can be used to generate passband signal directly without transmitter 
    #       upconversion or used to model passband equivalent behaviour for baseband 
    #       signal (needs less samples -> faster simulation speed).

@dataclass
class GenericPulsedRadarSignalParameters(GenericSignalParameters):
    """ All parameteres common to pulsed radar signals
    """
    # Given parameters
    pulse_time: float   = None          # Pulse width (s)
    prf: list[float]    = None          # Pulse repetition frequency (f)
    phase: float        = None          # Phase offset
    window: str         = None          # Windowing algorithm for smooth transitions

# ---------- Signal type specific dataclasses ---------- #
# NOTE: Define signal specific dataclasses here
@dataclass
class RadarRectParameters(GenericPulsedRadarSignalParameters):
    """ Extends pulsed radar signals with sinusoidal rect signal.
    Simplest pulsed radar signal, hence same parameters than generic pulsed radar signals.
    """
    pass

@dataclass
class RadarChirpParameters(GenericPulsedRadarSignalParameters):
    """ Extends pulsed radar signals with LFM chirp parameters
    """
    bw: float           = None          # Modulation bandwidth (f)


@dataclass
class RadarTriangularChirpParameters(GenericPulsedRadarSignalParameters):
    """ Extends pulsed radar signals with LFM chirp parameters
    """
    bw: float           = None          # Again the Modulation bandwidth in (f)
    mode: str           = "linear"      # method for the scipy.chirp function further down

# TODO: Add dataclasses for new signal types here.
#       For example:
#           BinaryPhaseCodedParameters(GenericPulsedRadarSignalParameters):
#           FrequencyModulatedContinuousWaveform(GenericSignalParameters): # FMCW

# ------------------------------------------------------ #

class radar_signal_generator(thesdk):
    """
    Radar Signal Generator:
        This entity generates a radar signal based on given radar signal dataclass
        Supported waveforms:
            Pulsed waveforms:
                rect
                chirp
                TODO: binary_phase_coded, etc.
            Continuous waveforms:
                TODO: FMCW, etc.
    """
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self, **kwargs):
        self.print_log(type='I', msg='Initializing %s' %(__name__))

        # Radar siggen attributes
        # NOTE: Needs to be defined as specified dataclass
        self.params = kwargs.get('signal_params')
        #if self.params.ch == None: self.params.ch = [0]

        # Advanced radar signal generator options
        self.enable_periodic_pulse_generation = True # Disable for only one pulse
        self.full_pri_output = False # Enable for non-clipped PRIs (NOTE: by enabling signal length diverges from nsamp)

        # IO 
        self.IOS.Members['IQ_OUT'] = IO()
        self.IOS.Members['IQ_REF_OUT'] = IO()

        self.init()

    def init(self):
        #self.params.ch = [0]
        pass
        
    def run(self):
        self.main()

    # ----- Signal Generator ----- #
    def main(self):
        """
        Description:
            Main method for the radar signal generator.
            1. Generates correct signal type based in given params
            2. Windows the signal to smoothen the pulse edge transients
            3. Assigns the signal to output IO
        """

        full_output = None
        pulse_output = None
        outval_IQ = None
        periods = []

        if self.enable_periodic_pulse_generation:
            for _ in range(ceil(self.pulse_count())): # TODO: Optimize this by reusing single generated pulse instead of generating a new pulse every loop
                # Iterate generated signal over predetermined channels
                channel_periods = []
                for f_ch in self.params.ch:
                    # Chosen signal type is assigned to outputs via outval
                    match self.signal_type(): 
                        case 'rect': 
                            pulse = self.rect()
                        case 'chirp': 
                            pulse = self.chirp()
                        case 'tri_chirp':
                            pulse = self.tri_chirp()
                        case 'binary_phase_coded': 
                            pulse = self.binary_phase_coded()
                        # Other possible waveforms: 
                        #'binary phase coded', 'non-linear FM', 'discrete frequency-shift', 'polyphase codes', 'compound Barker codes', 'code sequencing', 'complementary codes', 'pulse burst', 'stretch'
                        case _:
                            self.print_log(type='F',msg='Signal type \'%s\' not supported.' % self.params.sigtype)
                    # Apply upconversion to the generated and windowed signal
                    #channel_period = self.apply_window(pulse)
                    channel_period = self.apply_upconversion(self.apply_window(pulse), f_ch)
                    
                    # Apply one channe period to list of all channels in a period
                    channel_periods.append(channel_period)
                # Sum channels together in one period
                period = np.sum(channel_periods, axis=0)
                # Append multi channel period to list of all periods
                periods.append(period)
            # Output variables
            full_output = np.concatenate(periods)
            pulse_output = pulse[0:self.time_as_samples(self.params.pulse_time)]
            #pulse_output = full_output[0:ceil(self.params.fs*self.params.pulse_time)]
            #pulse_output = periods[0][0:ceil(self.params.fs*self.params.pulse_time)]

        else:
            # Chosen signal type is assigned to outputs via outval
            match self.signal_type(): 
                case 'rect': 
                    period = self.rect()
                case 'chirp': 
                    period = self.chirp()
                case 'tri_chirp':
                    pulse = self.tri_chirp()
                case _:
                    self.print_log(type='F',msg='Signal type \'%s\' not supported.' % self.params.sigtype)
            # Output variables
            full_output = self.apply_window(period)
            pulse_output = full_output[0:ceil(self.params.fs*self.params.pulse_time)]
        # Output signal
        #if self.params.snr is not None:
        #    full_output = self.apply_noise(full_output)
        #outval_IQ = self.apply_rms(outval_IQ)

        if not self.full_pri_output: full_output = full_output[0:self.params.nsamp]
        self.IOS.Members['IQ_OUT'].Data     = full_output
        self.IOS.Members['IQ_REF_OUT'].Data = pulse_output

    # Select signal type based on signal parameters
    def signal_type(self):
        sig_type = None
        if isinstance(self.params, RadarRectParameters): return 'rect'
        elif isinstance(self.params, RadarChirpParameters): return 'chirp'
        elif isinstance(self.params, RadarTriangularChirpParameters): return 'tri_chirp'
        else: return None # Calls unknown signal type error

    # ----- Generate Pulse Period ----- #
    def rect(self):
        """
        Parameters:
        rect_config(T, t_start, N, fs)  # Rect signal attributes
        ----------
        Returns:[:64]
        x : ndar[:64]ray(dtype=complex128)   # Generated pulse waveform I/Q samples

        """
        T, t_start, N, fs = self.tfall, self.tstart, self.nsamp, self.fs

        n_high = int(T*fs)
        n_start = int(t_start*fs)
        n_end = min(N, n_start + n_high)
        x = np.zeros(N, dtype=np.complex128)
        x[n_start:n_end] = np.ones(n_end-n_start, dtype=np.complex128)

        return x

    def tri_chirp(self):

        f0 = 0.0
        T = self.params.pulse_time
        B = self.params.bw
        N = self.time_as_samples(self.pri())
        phi0_rad = -np.pi / 2.0
        phi0_deg = phi0_rad * 180.0 / np.pi

        fs = self.params.fs
        dt = 1.0 / fs
        n_end = int(T * fs)

        t = np.arange(n_end) * dt
        half_T = T / 2.0

        # Edge frequencies
        f1 = f0 - B / 2.0  # lowest at t = 0 and t = T
        f2 = f0 + B / 2.0  # highest at t = T/2

        # Slope for each half (Hz/s)
        k = (f2 - f1) / half_T

        # Masks and relative times
        mask_up = t < half_T
        t_up = t[mask_up]                 # 0 .. T/2
        t_down = t[~mask_up]              # T/2 .. T
        t_down_rel = t_down - half_T      # 0 .. T/2 for the down-sweep

        # Phase continuity at T/2:
        # phi_mid = phi0 + 2*pi * ∫_0^{T/2} f(τ) dτ = phi0 + 2*pi*(f1*half_T + 0.5*k*half_T^2)
        phase_mid_rad = phi0_rad + 2.0 * np.pi * (f1 * half_T + 0.5 * k * half_T * half_T)
        phi_mid_deg = (phase_mid_rad * 180.0 / np.pi) % 360.0

        try:
            # Newer SciPy versions: chirp supports complex=True (analytic signal)
            y_up = chirp(t_up, f0=f1, t1=half_T, f1=f2, method=self.params.mode, phi=phi0_deg, complex=True)
            y_down = chirp(t_down_rel, f0=f2, t1=half_T, f1=f1, method=self.params.mode, phi=phi_mid_deg, complex=True)
            y = np.empty(n_end, dtype=np.complex128)
            y[mask_up] = y_up
            y[~mask_up] = y_down
        except TypeError:
            # Fallback for SciPy versions without complex=True:
            # Generate real chirps for each half with proper phase continuity, then make analytic via Hilbert.
            y_up = chirp(t_up, f0=f1, t1=half_T, f1=f2, method=self.params.mode, phi=phi0_deg, complex=False)
            y_down = chirp(t_down_rel, f0=f2, t1=half_T, f1=f1, method=self.params.mode, phi=phi_mid_deg, complex=False)
            y_real = np.empty(n_end, dtype=float)
            y_real[mask_up] = y_up
            y_real[~mask_up] = y_down
            y = hilbert(y_real).astype(np.complex128)

        # Scale amplitude and place into an N-length buffer
        tri_chirp_complex = self.params.amp * y
        chirp_pulse = np.zeros(N, dtype=np.complex128)
        chirp_pulse[:n_end] = tri_chirp_complex

        return chirp_pulse


    def chirp(self):
        """
        Description:
            Chirp generation with scipy (not in use but here just in case)
        Returns:
            chirp_pulse: np.complex128 = generated chirp signal
        """

        f0 = 0
        T = self.params.pulse_time
        B = self.params.bw
        N = self.time_as_samples(self.pri())
        phi0=-np.pi/2

        t = np.arange(N) * self.params.pulse_time
        n_end = int(self.params.pulse_time*self.params.fs)
        t = np.arange(n_end)/self.params.fs

        f1 = f0 -B/2
        f2 = f0 + B/2

        chirp_pulse = np.zeros(N, dtype=np.complex128)
        chirp_pulse[:n_end] = self.params.amp * chirp(t, f1, T, f2, method='linear', phi=phi0, complex=True)

        return chirp_pulse


    # ----- Signal Processing ----- #
    # Apply properties to existing signals

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
        if window == 'none':
            w = w
        elif window == 'hamming':
            w = np.hamming(len(s))
        elif window == 'hann':
            w = np.hanning(len(s))
        elif window == 'tukey':
            # Tukey alpha ~= 2*edge_frac so that each edge gets edge_frac of the pulse
            edge_frac = 0.04    # NOTE: Use this to control tukey effectiveness (good value edge_frac = 0.04)
            L = int(self.time_as_samples(self.params.pulse_time))
            w_short = tukey(L, alpha=2*edge_frac)
            w = s
            w[:L] *= w_short
        else:
            self.print_log(type='W',msg='\'%s\' is not a valid windowing method. Defaulting to no window.' % self.param.window)

            s = s * w

        return s

    def apply_noise(self, s):
        n = len(s)
        sn = (np.random.randn(n) + 1j*np.random.randn(n)) * (1/(10**(self.params.snr/20.0)))
        s = s+sn
        return s

    def apply_rms(self, s):
        s = s / np.sqrt(np.mean(np.abs(s)**2))
        return s

    def apply_upconversion(self, s, f_ch):
        import scipy.constants as const
        f_nyquist = self.params.fs/2
        assert (f_ch < f_nyquist - self.params.bw/2), 'Channel frequency exceeds Nyquist frequency'
        t = np.arange(self.params.fs*1/self.params.prf[0])/self.params.fs
        t = t[0:len(s)] # NOTE: This should not be needed but fixes some edgecase
        s_ch = s * np.exp(2*const.pi*1j*f_ch * t)
        return s_ch

    # ----- Derivative Signal Properties ----- #
    # Get/calculate derivative signal properties
    def pulse_power(self):
        s = self.IOS.Members['IQ_REF_OUT'].Data
        s = np.asarray(s)
        P_signal = np.mean(np.abs(s)**2)
        return P_signal

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

    def pri(self):
        """
        Description:
            How much time does one pulse repetition cycle take
        Returns:
            pri: float = pulse_repetition_interval
        """
        return 1/self.params.prf[0]

    def pulse_count(self):
        """
        Description:
            How many pulse repetition intervals fit within simulated time period
        Returns:
            pri_count: float
        """
        pulse_time = self.pri()
        pulse_repetitions = self.params.nsamp / self.time_as_samples(pulse_time)
        return pulse_repetitions

    def max_signal_range(self):
        """
        Description:
            Calculates the maximum range
        Returns:
            radar_range: int = number of samples
        """
        return (const.c  * (self.pri() - self.params.pulse_time)) / 2

    def min_signal_range(self):
        """
        Description:
          Calculates minimum range (assuming the transmission and reception cannot overlap)
        Returns:
          min_signal_range: float =
        """
        return const.c * self.params.pulse_time / 2

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
        return const.c / (2*self.params.bw)


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
        print(f"| {'Power [W]':<{width_1}} | {self.pulse_power():>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'Sample rate [Hz]':<{width_1}} | {self.params.fs:>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'Generated Signal Time [s]':<{width_1}} | {self.signal_time():>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'PRF [Hz]':<{width_1}} | {self.params.prf[0]:>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'PRI [s]':<{width_1}} | {self.pri():>{width_2}.3e} |")
        print('-' * full_width)
        print(f"| {'Simulated Pulse Count':<{width_1}} | {self.pulse_count():>{width_2}.2f} |")
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
        RadarChirpParameters( # TODO: Find realistic specs from some source
              amp           = 1,          # Amp
              fs            = 100e6,      # Sample rate
              pulse_time    = 50e-6,      # Length of the radar pulse in seconds
              prf           = [14e3],       # Pulse repetition frequency
              bw            = 1e6,        # Bandwidth of the chirp linear frequency modulation
              ch            = [0],        # List of channels
              nsamp         = 2**10,      # Number of samples in the output (total length of the generated signal)
              phase         = np.pi/2,    # Phase offset
              window        = 'tukey',    # Windowing algorithm for smooth transients,
        ),
        'tri_chirp_test_ref':
        RadarTriangularChirpParameters(
              amp           = 1,          # Amp
              fs            = 100e6,      # Sample rate
              pulse_time    = 50e-6,      # Length of the radar pulse in seconds
              prf           = [14e3],     # Pulse repetition frequency
              bw            = 1e6,        # Bandwidth of the chirp linear frequency modulation
              ch            = [0],        # List of channels
              nsamp         = 2**11,      # Number of samples in the output (total length of the generated signal)
              phase         = np.pi/2,    # Phase offset
              window        = 'tukey',    # Windowing algorithm for smooth transients,
              mode          = "linear",   # {‘linear’, ‘quadratic’, ‘hyperbolic’}, optional
        ),
        'tri_chirp_test':
        RadarTriangularChirpParameters(
              amp           = 1,          # Amp
              fs            = 100e6,      # Sample rate
              pulse_time    = 50e-6,      # Length of the radar pulse in seconds
              prf           = [5e3],      # Pulse repetition frequency
              bw            = 2e6,        # Bandwidth of the chirp linear frequency modulation
              ch            = [0],        # List of channels
              nsamp         = 2**11,      # Number of samples in the output (total length of the generated signal)
              phase         = np.pi/2,    # Phase offset
              window        = 'tukey',    # Windowing algorithm for smooth transients,
              mode          = "linear",   # {‘linear’, ‘quadratic’, ‘hyperbolic’}, optional
        ),
        'multichannel_chirp_test':
        RadarChirpParameters( # TODO: Find realistic specs from some source
              amp           = 1,          # Amp
              fs            = 400e6,      # Sample rate
              pulse_time    = 50e-6,      # Length of the radar pulse in seconds
              prf           = [14e3],     # Pulse repetition frequency
              bw            = 100e6,        # Bandwidth of the chirp linear frequency modulation
              ch            = [-100e6, 0, 100e6],        # List of channels
              nsamp         = 2**10,      # Number of samples in the output (total length of the generated signal)
              phase         = np.pi/2,    # Phase offset
              window        = 'tukey',    # Windowing algorithm for smooth transients,
        ),
        'multichannel_tri_chirp_test':
        RadarTriangularChirpParameters(
              amp           = 1,          # Amp
              fs            = 100e6,      # Sample rate
              pulse_time    = 50e-6,      # Length of the radar pulse in seconds
              prf           = [5e3],      # Pulse repetition frequency
              bw            = 2e6,        # Bandwidth of the chirp linear frequency modulation
              ch            = [-50e9, 0, 50e9],        # List of channels
              nsamp         = 2**11,      # Number of samples in the output (total length of the generated signal)
              phase         = np.pi/2,    # Phase offset
              window        = 'tukey',    # Windowing algorithm for smooth transients,
              mode          = "linear",   # {‘linear’, ‘quadratic’, ‘hyperbolic’}, optional
        ),
    }
    

    list_of_tests = [
        #'chirp_test',
        #'tri_chirp_test_ref',
        #'tri_chirp_test',
        'multichannel_chirp_test',
        #'multichannel_tri_chirp_test',
        ]
    for chosen_test_case in list_of_tests:
        dut = radar_signal_generator()
        dut.params = test_cases[chosen_test_case]
        dut.run()

        r = np.complex128(dut.IOS.Members['IQ_OUT'].Data)

        # Genearte header file
        # TODO: Adjust scaling coefficient for header generation
        header_scaling_coeff = 65536
        r_scaled = scale_dbm(r, -30)
        if False:
            dsp_toolkit.generate_header(
                r,
                I = r_scaled.real*header_scaling_coeff,
                Q = r_scaled.imag*header_scaling_coeff,
                header_file_path = "generated_header.h"
        )

        # TODO: plotting with signal analyser

        plot_simple(r, dut.params.fs, figsize=(8, 4), label=chosen_test_case, x_lim_factor=2)
        
        plot_bb_spectrum(r, 400e6, scale='dbfs', window='rect', color='black', ylim=[-100,10])
        dut.print_signal_specs()


    # Show plot
    plt.show()

    input()

