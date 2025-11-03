import scipy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from radar_signal_generator import RadarChirpParameters

# set sensible default values for matplotlib style
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['figure.figsize'] = (3.5, 2.2)
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['figure.autolayout'] = True

def plot_bb_spectrum(x, fs, scale='dbfs', annotate_max=False, window='rect', color='red', ylim=[-100,10]):
  plt.figure()
  N = len(x)

  # Apply windowing function
  # use sym=False for spectral analysis
  if window == 'rect':
    window = np.ones(N)
  elif window == 'flattop':
    # for power estimation
    window = scipy.signal.windows.flattop(N, sym=False)
  elif window == 'hann':
    window = scipy.signal.windows.hann(N, sym=False)
  else:
    raise NotImplementedError("window type not implemented")
  # normalize by average value of windowing function to preserve amplitude
  window /= np.mean(window)
  x_windowed = x*window

  # normalization of X is not strictly necessary if plotting in dBFS
  f = scipy.fft.fftfreq(N, d=1/fs)
  X = scipy.fft.fft(x_windowed) / N
  f2 = scipy.fft.fftshift(f)
  X2 = scipy.fft.fftshift(X)

  ax = plt.gca()

  # amplitude to dB conversion
  if scale=='abs':
    P_plot = np.abs(X2)
  if scale=='dbfs':
    P = np.abs(X2)**2
    P_norm = P / np.max(P)
    P_plot = 10*np.log10(P_norm)
    ax.set_ylim(ylim[0], ylim[1])
  if scale=='v2dbm':
    # amplitude to dbm, assuming np.abs(X2) is voltage (amplitude)
    voltage = np.abs(X2)
    power = (voltage**2)/(2*50)
    P_dbm = 10*np.log10(power/0.001)
    P_plot = P_dbm
    ax.set_ylim(ylim[0], ylim[1])


  # formatter = EngFormatter(unit='Hz')
  # ax.xaxis.set_major_formatter(formatter)

  # adjust freq unit
  ax.set_xlabel("Frequency [MHz]")
  f2 /= 1e6

  # zoom in
  # todo: parameterize
  width = 256
  start = N//2 - width
  end = N//2 + width

  # don't zoom in
  start = 0
  end = len(x)

  # find peak
  if annotate_max:
    i_max = np.argmax(P_plot)
    P_max = P_plot[i_max]

  # plot spectrum
  plt.plot(f2[start:end], P_plot[start:end], linewidth=2, color=color)

  # annotate peak
  if annotate_max:
    plt.scatter(f2[i_max], P_max)
    plt.annotate(f"({f2[i_max]:.3f}, {P_max:.1f})", xy=(f2[i_max],P_max), xycoords='data', xytext=(5,5), textcoords='offset points', fontsize=8)
    print(f"peak at f={f2[i_max]}")

  return (f2, X2)

def scale_dbm(x, dbm):
    """Scale signal amplitude so that its peak power has `dbm` power"""
    P_abs = 0.001 * 10**(dbm/10)

    # measure peak power of original signal
    N = len(x)
    # no need to window because we are analyzing periodic signals with transients
    X = scipy.fft.fft(x) / N
    P = (np.abs(X)**2)/(2*50)
    P_max = np.max(P)

    # required gain
    gain_abs = P_abs / P_max
    gain_ampl = np.sqrt(gain_abs)

    return x * gain_ampl

def plot_simple(x, fs, label="Time-domain signal"):
    t = np.arange(len(x))/fs
    plt.figure()
    plt.plot(t*1e6, np.real(x), 'k', marker='o', markersize=1)
    plt.xlabel("Time (µs)")
    plt.ylabel("Value")
    plt.title(label)
    plt.grid(True)
    #plt.show()

def plot_spectrum(s, fs, title="Spectrum"):
    L = len(s)
    S = np.fft.fftshift(np.fft.fft(s))
    f = np.fft.fftshift(np.fft.fftfreq(L, 1/fs))
    plt.figure()
    plt.plot(f/1e6, 20*np.log10(np.abs(S)+1e-12), 'k', marker='o', markersize=1)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True)
    #plt.show()

if __name__ == "__main__":
  # test scale_dbm
  fs = 2**20
  n = np.arange(2**20)
  # 0dBm sine wave
  # x = 0.224*np.sqrt(2)*np.exp(2j*np.pi*200e3*n/fs)
  x = np.concatenate((np.ones(10), np.zeros(len(n)-10))) + 0j*np.zeros(len(n))
  # rect pulse
  x_scaled = x
  x_scaled = scale_dbm(x, -30)
  # time-domain
  plt.figure()
  plt.plot(x_scaled.real[:64], 'blue')
  plt.plot(x_scaled.imag[:64], 'red')
  plt.legend(['I','Q'])
  # spectrum
  plt.figure()
  plt.grid()
  plt.ylabel("dBm")
  plot_bb_spectrum(x_scaled,fs,scale='v2dbm',window='rect', color='black', ylim=[-70,10])
  #plot_bb_spectrum(x_scaled,fs,scale='dbfs',window='rect', color='black', ylim=[-70,10])

  # Show plot
  plt.show()
