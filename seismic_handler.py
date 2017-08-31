import numpy as np
import matplotlib.pyplot as plt

def convert_to_image (data, shape = None):
    import numpy as np
    data = data * 255 / np.max(data)
    
    from scipy.misc import toimage
    im = toimage(data)
    if shape != None:
        im = im.resize(shape)
    return im

        
class FreqNoiser:
    def __init__(self):
        pass
        
    def run (self, handler):
        handler.data = self.addNoiseStatic(handler.data)
        
    @staticmethod 
    def addNoiseStatic (data):
        import numpy as np
        data_new = []
        for t in data:
            f = np.fft.rfft(t)
            l = len (f)
            llow = int(l*0.5);
            lhi = int(l*0.75)
            for i in range (l):
                if i < llow:
                    sc = 0
                if i in range(llow, lhi):
                    sc = (i - llow)/(lhi - llow)
                if i > lhi:
                    sc = 1
                
                f [i] *= 1 + sc*10
            
            t_new = np.fft.irfft(f)
            data_new.append (t_new)
        return np.array(data_new)
    
class SeismicPrestack:
    def __init__(self, filename, processor = None):
        from obspy.io.segy.segy import _read_segy
        self.stream = _read_segy(filename, unpack_headers=True)
        self.dt = self.stream.binary_file_header.sample_interval_in_microseconds/1000000
        self.nsamp = len (self.stream.traces[0].data)
        self.dt_synt = 0.004
        self.dx_synt = 1
        self.processor = processor
        
    def getHeaders (self):
        return [k for k in self.stream.traces[0].header.__dict__]
    
    def getHeaderVals (self, header):
        self.header_vals = [t.header.__dict__[header] for t in self.stream.traces]        
        return np.unique (self.header_vals)
        
    def getHeaderIndex (self, v) : 
        return [index for index, value in enumerate(self.header_vals) if value == v]
   
    def getPartOrig (self, tr1, tr2, s1, s2):
#        tr1 = int (tr1)
#        tr2 = int (tr2)
#        t1 = int(time1/self.dt) 
#        t2 = int(time2/self.dt) 
#        tr2 = min (tr2, len (self.stream.traces))
#        t2 = min (t2, self.nsamp)

        tr2 = min (tr2, len (self.stream.traces))
        s2 = min (s2, self.nsamp)        
        return self.data[tr1:tr2, s1:s2]
        
    def getPart (self, tr1, tr2, s1, s2):
        tr2 = min (tr2, len (self.stream.traces))
        s2 = min (s2, self.nsamp)
    
        data = self.getPartOrig (tr1, tr2, s1, s2)
        from scipy import interpolate
        x = range(tr1, tr2, 1)
        y = range(s1, s2, 1)
        
        if len(x) != data.shape[0] or len(y) != data.shape[1]:
            print ("len(x) != data.shape[0] or len(y) != data.shape[1]")
            print (len(x), len(y), data.shape)
            exit (0)
            
        return data
#        from scipy.interpolate import RegularGridInterpolator
#        my_interpolating_function = RegularGridInterpolator((x, y), data)
#        
#        xnew = np.arange(tr1, tr2, self.dx_synt)
#        ynew = np.arange(time1, time2, self.dt_synt)
#        
#        pts = []
#        for tr in xnew:
#            for t in ynew:
#                pts.append ([tr, t])
#                #print (tr, t, my_interpolating_function([tr, t]))
#                
#        
#        interp = my_interpolating_function(pts)
#        return interp.reshape ((len(xnew), len(ynew)))
#        
#        xx, yy = np.meshgrid(x, y)
#        f = interpolate.interp2d(xx, yy, data, kind='cubic')
#        
#        data_new = f(xnew, ynew)
#        return data_new.T
        
    def readGather (self, gather_value):
        index = self.getHeaderIndex(gather_value)
        self.data = []
        for i in index:
            self.data.append(self.stream.traces[i].data)
        self.data = np.array(self.data)
        if self.processor:
            self.processor.run(self)
        
    def readGatherParts (self, gather_value, xwin, twin):
        xwin = min (xwin, len (self.stream.traces))
        twin = min (twin, self.dt * self.nsamp)
        trwin = int (xwin)
        swin = int (twin/self.dt)
        
        self.readGather(gather_value)
        smax = self.nsamp
        xmax = len (self.data)
        parts = []
        for x in range(0,int(xmax-trwin/4.),int(trwin/2)):
            xend = x + trwin
            if xend > xmax:
                if (xend - xmax) > int(trwin/4.):
                    continue
                x = xmax - trwin
                xend = xmax
                
            for s in range(0,int(smax-swin/4.),int(swin/2)):
                send = s + swin
                if send > smax:
                    if (send - smax) > int(swin/4.):
                        continue
                    s = smax - swin
                    send = smax
                    
                d = self.getPart (x, xend, s, send)
                if np.amax(d) == np.amin(d):
                    continue
                parts.append(d)
                
        sh = parts[0].shape
        for data in parts:
            if sh != data.shape:
                throw ('wrong shapes')
            
        return parts
            
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
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)

        # How to remove decorations from matplotlib: https://stackoverflow.com/questions/38411226/matplotlib-get-clean-plot-remove-all-decorations
        #ax.set_axis_off()

        data = data.T
        vm = np.percentile(data, 99)
        imparams = {
                    #'interpolation': 'none',
                    'cmap': "gray",
                    'vmin': -vm,
                    'vmax': vm,
                    'aspect': 'auto'
                    }
        plt.imshow(data, **imparams)
        #plt.colorbar()
        #plt.show()
        return plt

    @staticmethod
    def plot_spec(data):
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
            
        # How to remove decorations from matplotlib: https://stackoverflow.com/questions/38411226/matplotlib-get-clean-plot-remove-all-decorations
        #ax.set_axis_off()

        imparams = {
                    #'interpolation': 'none',
                    'cmap': "gray",
                    'aspect': 'auto'
                    }
        plt.imshow(data, **imparams)
        #plt.colorbar()
        #plt.show()
        return plt

        
    def wiggle_plot(self, data,
                    skip=1,
                    perc=99.0,
                    gain=1.0,
                    rgb=(0, 0, 0),
                    alpha=0.5,
                    lw=0.2,
                    ):
                    
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
            
        # How to remove decorations from matplotlib: https://stackoverflow.com/questions/38411226/matplotlib-get-clean-plot-remove-all-decorations
        #ax.set_axis_off()
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
        return plt