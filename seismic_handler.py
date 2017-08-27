import numpy as np
import matplotlib.pyplot as plt

class SeismicPrestack:
    def __init__(self, filename):
        from obspy.io.segy.segy import _read_segy
        self.stream = _read_segy(filename)
        self.dt = self.stream.binary_file_header.sample_interval_in_microseconds/1000000
        self.nsamp = len (self.stream.traces[0].data)
        self.dt_synt = 0.004
        self.dx_synt = 1
   
    def readPartOrig (self, tr1, tr2, time1, time2):
        t1 = int(time1/self.dt) 
        t2 = int(time2/self.dt) 
        tr2 = min (tr2, len (self.stream.traces))
        t2 = min (t2, self.nsamp)
        
        data = []
        for tr in range (tr1, tr2):
            d = self.stream.traces[tr].data
            data.append (d[t1:t2])
            
        data = np.array(data)
        return data
        
    def readPart (self, tr1, tr2, time1, time2):
        tr2 = min (tr2, len (self.stream.traces))
        time2 = min (time2, self.dt * self.nsamp)
    
        data = self.readPartOrig (tr1, tr2, time1, time2).T
        from scipy import interpolate
        x = np.arange(tr1, tr2, 1)
        y = np.arange(time1, time2, self.dt)
        xx, yy = np.meshgrid(x, y)
        f = interpolate.interp2d(x, y, data, kind='cubic')
        
        xnew = np.arange(tr1, tr2, self.dx_synt)
        ynew = np.arange(time1, time2, self.dt_synt)
        data_new = f(xnew, ynew)
        return data_new.T
        
    @staticmethod
    def spectrum(signal, taper = True):
        windowed = signal
        if taper:
            windowed = windowed * np.blackman(len(windowed))
        a = abs(np.fft.rfft(windowed))

        #db = 20 * np.log10(a)

        return a

    @staticmethod
    def spectrogram (data):
        spec_matrix = []
        db_matrix = []
        for i in range(len(data)):
            a = SeismicPrestack.spectrum(data[i], False)
            spec_matrix.append (a)
            db = 20 * np.log10(a)        
            db_matrix.append (db)
            
        spec_matrix = np.array(spec_matrix)
        db_matrix = db_matrix - np.amax(db_matrix)
        return db_matrix

    @staticmethod
    def fk (data):
        #data = data*np.blackman(len(data[0]))
        data = data.T
        freq = np.fft.fft2(data)
        freq = np.fft.fftshift(freq)
        freq = freq[int(len(freq)/2):,:]

        #print (np.fft.rfftfreq(freq.shape[0], self.dt_synt))
        #print (np.fft.fftfreq(freq.shape[1], self.dx_synt))

        freq = np.abs(freq)
        freq = 20 * np.log10(freq)
        freq = freq - np.amax(freq)
        return freq
        
    @staticmethod
    def plot(data):
        data = data.T
        vm = np.percentile(data, 99)
        imparams = {'interpolation': 'none',
                    'cmap': "gray",
                    'vmin': -vm,
                    'vmax': vm,
                    'aspect': 'auto'
                    }
        plt.imshow(data, **imparams)
        plt.colorbar()
        plt.show()
        return
    
    def wiggle_plot(self, data,
                    ax=None,
                    skip=1,
                    perc=99.0,
                    gain=1.0,
                    rgb=(0, 0, 0),
                    alpha=0.5,
                    lw=0.2,
                    ):
                    
        if ax is None:
            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(111)
            
        # How to remove decorations from matplotlib: https://stackoverflow.com/questions/38411226/matplotlib-get-clean-plot-remove-all-decorations
        ax.set_axis_off()
        plt.gca().invert_yaxis()
        
        tbasis = np.arange(0, len(data[0]) * self.dt, self.dt)

        rgba = list(rgb) + [alpha]
        sc = np.percentile(data, perc)  # Normalization factor
        wigdata = data[::skip, :]
        xpos = np.arange(len(data))[::skip]

        for x, trace in zip(xpos, wigdata):
            # Compute high resolution trace.
            amp = gain * trace / sc + x 
            t = 1000 * tbasis
            hypertime = np.linspace(t[0], t[-1], (10 * t.size - 1) + 1)
            hyperamp = np.interp(hypertime, t, amp)

            # Plot the line, then the fill.
            ax.plot(hyperamp, hypertime, 'k', lw=lw)
            ax.fill_betweenx(hypertime, hyperamp, x,
                             where=hyperamp > x,
                             facecolor=rgba,
                             lw=0,
                             )
        return ax
