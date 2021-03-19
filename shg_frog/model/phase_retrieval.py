"""
Module that implements the phase retrieval algorithm used by the FROG software


*********
The realization of the GP (general projections) phase retrieval algorithm
used here is based on the Matlab code from Steven Byrnes who wrote an extension
of Adam Wyatt's MATLAB FROG program. Various features include anti-aliasing
algorithm. The original MATLAB code underlies the following license:

Copyright (c) 2012, Steven Byrnes
Copyright (c) 2009, Adam Wyatt
All rights reserved.

Disclaimer:
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*********

*********
The realization of the Ptychographic Reconstruction algorithm, also used in here
is based on the paper of Sidorenko et al. (2016) and their Matlab implementation.
*********

Julian Krauth translated and adapted the original Matlab code pieces for the use
in the shg_frog python package.

File name: phase_retrieval.py
Author: Julian Krauth
Date created: 2019/11/27
"""
import numpy as np

from ..helpers.file_handler import FileHandler


def rms_diff(F1: np.ndarray, F2: np.ndarray) -> float:
    """ Calculates RMS difference in the entries of two real matrices/vectors. """
    result = np.sqrt(np.mean(np.square(F1-F2)))
    return result

def calc_alpha(Fm, Fr):
    """ Calculates alpha, the positive number that minimizes
    rms_diff(Fm,alpha*Fr). See DeLong1996
    """
    result = np.sum(Fm*Fr)/np.sum(np.square(Fr))
    return result

def parity(x):
    """
    1 for odd, 0 for even. Don't delete! It looks like it's not used
    in the program, but it can be called by the 'method' strings.
    """
    result = int(x-2*np.floor(x/2.))
    return result

def make_axis(length: int, step: float) -> np.ndarray:
    """ Create an array that contains the values of a time or frequency axis,
    given the step size and the length of the array. The values are centered
    around zero.

    Arguments:
        length -- length of array
        step -- step size

    Returns:
        axis -- horizontal array for use as a plot axis
    """
    axis = np.arange(-length/2 * step, length/2*step, step)
    return axis

def shift_signal(sig_in: np.ndarray, shift: float, freq_axis: np.ndarray):
    """
    Shift pulse in time by a given delay.

    Arguments:
    sig_in -- complex float dim 128, pulse field (in time)
    d -- float, a delay picked from the delay vector
    F -- float dim 128, vector of frequencies

    Returns:
    sig_out -- complex float dim 128, pulse field
    """
    sig_freq_domain = np.fft.fft(sig_in, axis=0)
    sig_out = np.fft.ifft(sig_freq_domain * np.exp(1.j*2*np.pi*shift*freq_axis), axis=0)
    return sig_out

def get_norm_intensity(pulse_field: np.ndarray) -> np.ndarray:
    """ Convert field to intensity and normalize such that max is 1. """
    intensity = np.square(np.abs(pulse_field))
    return intensity/np.amax(intensity)

def get_fwhm(intensity: np.ndarray, axis: np.ndarray) -> float:
    """ Calculates FWHM of a signal in units of the given axis.
    This function uses interpolation between data points and a threshold approach.
    No fitting, since this would be less robust for general pulse shapes.
    Arguments:
    intensity -- Array with intensity information of a pulse
    axis -- Array with time information of a pulse
    Returns:
    float -- full width half maximum value"""
    threshold = np.amax(intensity)/2
    step_size = axis[1]-axis[0]
    # Get indices of values that are above half the maximum
    indices = np.nonzero(intensity>threshold)[0]
    # Since the resolution is low, we do some geometry (interpolating):
    ## Interpolate the left side
    index_before_threshold = indices[0] - 1
    index_right_after = indices[0]
    left_delta = step_size\
        *(threshold-intensity[index_before_threshold])\
            /(intensity[index_right_after]-intensity[index_before_threshold])
    left_side = axis[index_before_threshold] + left_delta
    ## Interpolate the right side
    index_after_threshold = indices[-1] + 1
    index_right_before = indices[-1]
    right_delta = step_size*(threshold-intensity[index_right_before])\
        /(intensity[index_after_threshold]-intensity[index_right_before])
    right_side = axis[index_right_before] + right_delta
    ## Get the difference
    fwhm = right_side - left_side
    return fwhm

def print_started_message():
    print('Reconstruction started...')

def print_finished_message():
    print('Phase retrieval finished!')

def print_iter_update(index: int, frog_error: float):
    """ print the current iteration index and error to the terminal. """
    print(f'Iteration:{index:3d}  Error={frog_error:.4f}')


class PhaseRetrieval:
    """
    The two main methods of this class are:
    1. prepFROG(): Preparing the measured FROG trace
       to be used for the phase retrieval
    2. retrievePhase(): Uses the prepared FROG trace and retrieves
       the pulse shape in time and frequency domain using a generalized projections
       algorithm
    3. ePIE_fun_FROG(): uses prepared FROG trace and retrieves the pulse
       shape and phase in time using a ptychographic algorithm.

    The retrievePhase() method uses two additional methods for
    the algorithm:
    - makeFROG()
    - guessPusle()
    """

    def __init__(
        self,
        max_iter: int=200,
        prep_size: int=128,
        GTol: float=0.001,
    ):
        """
        Arguments:
        max_iter: maximum of iterations for the algorithm
        prep_size: size of the frog traces prepared by prepfrog()
        GTol: tolerance level to stop a reconstruction.
        """
        ### Parameters used by prepFROG ###

        # Pixel number of preparated FROG trace, which will be used by
        # phase retrieval algorithm, sometimes later also called N
        self.prep_size = prep_size

        ### Parameters used by phaseRetrieve ###

        # Difference in time-delay (dt) between consecutive pixels. This
        # automatically fixes frequency units, too. Doesn't affect algorithm,
        # only plots and such. This value will be updated by prepFROG
        self.dtperpx = None
        # previously I had here as a default value 0.0308
        # Frequency interval per pixel, given by self.dtperpx
        #self.dvperpx = 1 / (self.N*self.dtperpx)

        # Cell-array: units[0] is units of dtperpx, units[1] is units[0]^-1,
        # the units of frequency.
        self.units = ['ps','THz']

        ### FROG traces ###
        # Preparated trace, created by self.setPrepFrogTrace()
        self.Fm = np.zeros((self.prep_size,self.prep_size)).astype(np.float64)
        # Reconstructed electric field in time domain, created by retrieval function
        self.Pt = None
        # Reconstructed trace, created by retrieval function
        self.Fr = None

        ### Parameters used by retrievePhase method

        # Maximum number of iterations allowed
        self.max_iter = max_iter
        # Mode for initial guess for the pulse electric field.
        # Possible values are "autocorr", "gauss", "custom"
        self._seed_mode = "autocorr"
        # Tolerance on the error
        self._GTol = GTol
        # Algorithm choice
        # method[0]: makeFROGdomain      # 0,1,2,3
        # method[1]: makeFROGantialias   # 1 = antialias
        # method[2]: guessPulsedomain    # not used
        # method[3]: guessPulseantialias # not used
        self.method = [0,0,0,0]


    def set_size(self, val: int):
        """ Sets the size that is used by prepFROG to prepare the
        FROG trace.
        """
        self.prep_size = val

    def set_max_iterations(self, val: int):
        """ Sets the maximum iterations of the phase retrieval. """
        self.max_iter = val

    def set_tolerance(self, val: float):
        """ Sets the tolerance on the error between reconstructed and
        original FROG trace. If the error is smaller than the tolerance
        the retrieval ist stopped. """
        self._GTol = val

    def set_seed_mode(self, mode: str):
        """ Choose what seed is used in the retrieval. """
        if mode in set(["autocorr", "gauss", "custom"]):
            print(f"Set mode to '{mode}'")
            self._seed_mode = mode
        else:
            raise ValueError(f"mode={mode} is not a valid value.")

    def get_seed(self, mode: str=None, frog: np.ndarray=None) -> np.ndarray:
        """ Returns a seed function according to the self._seed_mode
        attribute.
        """
        if mode is None:
            mode = self._seed_mode
        if frog is not None and mode == "autocorr":
            # Sum over frequency axis yields the initial guess for the algorithm.
            # This corresponds to the intensity autocorrelation of the pulse.
            seed = np.sum(frog, axis=0) \
                / np.sqrt(np.sum(np.abs( np.sum(frog, axis=0) )**2))
            return seed.reshape(-1, 1)
        if mode == "gauss":
            # Generate initial guess of gate and pulse from noise times
            # a gaussian envelope function. Don't use complex phase 0 or
            # it gets stuck in real numbers, but don't let the complex
            # phase vary too much or it has aliasing problems.
            N = self.prep_size
            seed = (np.exp(
                -2. * np.log(2.) * np.square( (np.arange(0, N) - N/2.) / (N/10.) )
                ) * np.exp(0.1*2.*np.pi*1.j*np.random.rand(1, N)))
            return seed.reshape(-1, 1)
        if mode == "custom":
            # Loading a seed array that was the result of a measurement
            # and has been saved at some point.
            return FileHandler().load_seed()
        raise ValueError

    def save_pulse_as_seed(self):
        """ Save the last retrieved pulse as seed for another
        retrieval.
        """
        FileHandler().save_seed(self.Pt)


    def prepFROG(
        self,
        ccddt: float,
        ccddv: float,
        ccdimg: np.ndarray,
        showprogress: int=0,
        showautocor: int=0,
        flip: int=2,
    ):
        """
        prepFROG: Cleans, smooths, and downsamples data in preparation for
        running the FROG algorithm on it.
        In the end it saves the time steps per pixel into self.dtperpx
        and the prepared frog trace into self.Fm
        Arguments:
        ccddt -- time step per pixel, in ps
        ccddv -- frequency step per pixel, in THz
        ccdimg -- frog trace, frequencies on axis 0, delays on axis 1.
        """
        print('Prepare FROG trace...')

        # ccddtdv = ccddt * ccddv, with units of "cycles per horizontal-pixel
        # per vertical-pixel". This product ccddtdv is an important parameter
        # for the FROG algorithm, but ccddt and ccddv are NOT themselves
        # important. They are only used for graph labels.
        ccddtdv = ccddt * ccddv

        # Choose correct image orientation
        if flip in (1, 3):
            ccdimg = np.transpose(ccdimg)
        if flip in (2, 3):
            ccdimg = np.flipud(ccdimg)

        # Find the approximate center of the spot, by calculating an average
        # coordinate weighted by row-sums or column-sums.
        ccdsizev = np.size(ccdimg,0) #ccdsizev is how many rows
        ccdsizet = np.size(ccdimg,1) #ccdsizet is how many cols
        colsums = np.sum(ccdimg,0)
        centercol = np.inner(np.arange(1,ccdsizet+1),colsums) / np.sum(colsums)
        rowsums = np.sum(ccdimg,1)
        centerrow = np.inner(np.arange(1,ccdsizev+1),rowsums) / np.sum(rowsums)


        # Find the (very) approximate width of the spot in each dimension
        spotwidth = (2*np.inner(np.abs(np.arange(1,ccdsizet+1)-centercol),colsums)
                     / np.sum(colsums))
        spotheight = (2*np.inner(np.abs(np.arange(1,ccdsizev+1)-centerrow),rowsums)
                      / np.sum(rowsums))


        # Large "aspectratio" means vertical-stripe original image. Can also
        # input this or modify it by hand depending on what works best. This
        # is relevant because the final image will scale the dimensions to
        # make the final image aspect ratio roughly 1. (This helps accuracy).
        aspectratio=spotheight/spotwidth


        # vpxpersample and tpxpersample are the separation between consecutive
        # "samples" to be fed into the FROG algorithm. There are N*N=N^2
        # "samples" total, each is a pixel taken from the CCD image. They
        # satisfy these two equations:
        # (A): vpxpersample / tpxpersample = aspectratio (this helps accuracy)
        # (B): (vpxpersample * ccddv) * (tpxpersample * ccddt) = 1/N (this is
        # FFT requirement)
        vpxpersample = np.sqrt((1/(self.prep_size*ccddtdv))*aspectratio)
        tpxpersample = np.sqrt((1/(self.prep_size*ccddtdv))/aspectratio)

        #if showprogress:
        print('Vertical pixels per freq (v) sample: %.3f' % vpxpersample)
        print('Horizontal pixels per delay (t) sample: %.3f' % tpxpersample)

        # For me these are around 5 pixels.


        ################# IMAGE FILTERING #################
        if showprogress:
            plt.figure('prepFROG',figsize=(7,6))
            plt.subplot(221)
            plt.imshow(ccdimg)
            plt.title('(1) Original')

        #### LOW-PASS FOURIER FILTERING ####
        # See Taft and DeLong, chapter 10 in FROG textbook
        rho=0.3 # Lower rho means more extreme filtering. Make sure image looks OK.
        maxtimesrho = max([ccdsizev, ccdsizet])*rho
        ccdimgfft=np.fft.fftshift(np.fft.fft2(ccdimg))
        tophatfilter=np.zeros((ccdsizev, ccdsizet))
        for ii in range(ccdsizev):
            for jj in range(ccdsizet):
                if(np.sqrt(
                        np.square((ii+1)-ccdsizev/2.)+np.square((jj+1)-ccdsizet/2.))
                   <maxtimesrho):
                    tophatfilter[ii,jj]=1

        ccdimgfft = tophatfilter * ccdimgfft
        ccdimg = abs(np.fft.ifft2(np.fft.ifftshift(ccdimgfft)))
        if showprogress:
            plt.subplot(222)
            plt.imshow(ccdimg)
            plt.title('(2) After Fourier filter')

        #### BACKGROUND SUBTRACTION ####
        # The lowest-average-intensity 8x8 block of pixels is assumed to be the
        # background and is subtracted off
        imgblocks = np.zeros((ccdsizev, ccdsizet))
        for ii in range(8):
            ccdimg_rollv = np.roll(ccdimg, ii+1, axis=0)
            for jj in range(8):
                ccdimg_rollvt = np.roll(ccdimg_rollv, jj+1, axis=1)
                imgblocks = imgblocks + ccdimg_rollvt

        background = np.amin(imgblocks)/(8.*8.)

        ccdimg = ccdimg - background
        ccdimg[ccdimg<0] = 0 # Negative values are set to zero
        if showprogress:
            plt.subplot(223)
            plt.imshow(ccdimg)
            plt.title('(3) After background subtraction')

        #### DOWNSAMPLING TO NxN ####
        # Want an NxN pixel image to process. Go through each pixel of the
        # original, and have it contribute to the nearest pixel of the final
        # (in an average).
        if(ccdimg.shape==(self.prep_size, self.prep_size) and ccddt * ccddv == 1./self.prep_size):
            # Skip downsampling if ccdimg is already sampled correctly.
            fnlimg = ccdimg
            fnldt = ccddt
        else:
            fnlimg = np.zeros((self.prep_size, self.prep_size))
            # How many times you've added onto that pixel
            fnlimgcount = np.zeros((self.prep_size, self.prep_size))
            for ii in range(ccdsizev):  # Which row? (which freq?)
                rowinfinal = int(round(self.prep_size/2.+((ii+1)-centerrow)/vpxpersample))-1
                if(rowinfinal<0 or rowinfinal>=self.prep_size):
                    continue
                for jj in range(ccdsizet):
                    colinfinal = int(round(self.prep_size/2.+((jj+1)-centercol)/tpxpersample))-1
                    if(colinfinal<0 or colinfinal>self.prep_size-1):
                        continue
                    fnlimgcount[rowinfinal, colinfinal] += 1
                    fnlimg[rowinfinal, colinfinal] += ccdimg[ii,jj]
            fnlimgcount[fnlimgcount==0] = 1 # Avoid dividing by zero.
                                            # Pixels that haven't been written
                                            # into should be set to zero, and
                                            # they are.

            fnlimg = fnlimg / fnlimgcount
            fnldt = ccddt * tpxpersample
            #print(f'fnldt: {fnldt}  ccddt: {ccddt} tpxpersample: {tpxpersample}')


        #### Save results in corresponding attributes ####
        self.dtperpx = fnldt # set freq interval per pixel.
        self.Fm = fnlimg # set prep. Frog image for phase retrieval.

        if showprogress:
            plt.subplot(224)
            plt.imshow(fnlimg)
            plt.title('(4) After downsampling to %dx%d' % (self.prep_size, self.prep_size))
            plt.subplots_adjust(
                left=0.06, bottom=0.06, right=0.94, top=0.94, wspace=0.1, hspace=0.3
                )

        if showautocor:
            plt.figure('Autocorrelation',figsize=(6, 4))
            plt.plot(
                np.arange(-(ccdsizet/2.)*ccddt, ccdsizet/2.*ccddt, ccddt),
                np.sum(ccdimg, 0)
                )
            plt.title('Autocorrelation')
            plt.xlabel('Delay')
            plt.ylim((0, 1.05*max(np.sum(ccdimg, 0))))

        if showprogress or showautocor:
            plt.show()




    def makeFROG(self, Pt: np.ndarray, domain: int=0, antialias: int=0):
        """
        makeFROG: Reads in the (complex) electric field as a function of time,
        and computes the expected SHG-FROG trace.

        Arguments:
        Pt -- vertical array, complex electric field
        domain -- 0: delay space, 1: frequency space
        antialias --

        Return:
        [F, EF] -- frog intensity trace, frog complex field trace
        """

        N = len(Pt) # or: N = self.N

        if domain==0:
            EF = np.outer(Pt, Pt)

            if antialias:
    	        # Anti-alias: Delete entries that come from spurious
    	        # wrapping-around. For example, an entry like P2*G(N-1) is spurious
    	        # because we did not measure such a large delay. For even N, there
    	        # are terms like P_i*G_(i+N/2) and P_(i+N/2)*G_i, which correspond
    	        # to a delay of +N/2 or -N/2. I'm deleting these terms. They are
    	        # sort of out-of-range, sort of in-range, because the maximal delay
    	        # in the FFT can be considered to have either positive or negative
    	        # frequency. This is the outer edge of the FROG trace so it
    	        # should be zero anyway. Deleting both halves keeps everything
    	        # symmetric when sign of delay is flipped.
                EF = EF - np.tril(EF,-np.ceil(N/2.)) - np.triu(EF,np.ceil(N/2.))

            for n in range(0, N):
                # Row rotation...Eqs.(10)-->(11) of Kane1999
                EF[n,:] = np.roll(EF[n,:], -n)

            # EF is eqn (11) of Kane1999. From left column to right column, it's
            # tau=0,-1,-2...3,2,1

            # Permute the columns to the right order, tau=...,-1,0,1,...
            EF = np.fliplr(np.fft.fftshift(EF, 1))

	        # FFT each column and put 0 frequency in the correct place:
            EF = np.roll(np.fft.fft(EF, None, axis=0), int(np.ceil(N/2.)-1), 0)

            # Generate FROG trace (= |field|^2)
            F = np.square(np.absolute(EF))

            return F, EF


        if domain==1:
  	        # Frequency-domain integration; in other words,
	        # integral[P(w') * P(w-w')e^(-iw'tau)dw']. We follow Kane1999,
            # but starting in frequency-frequency space rather than
            # delay-delay space.
            PtFFT = np.fft.fft(Pt, axis=0)
            #EF = PtFFT*np.transpose(PtFFT)
            EF = np.outer(PtFFT, PtFFT)

  	        # Right now the (i,j) (i'th row and j'th column) entry of EF
            # corresponds to the product of the i'th Fourier component
            # with the j'th.
	        # But keep in mind "1st Fourier component" is v=0, the 2nd is v=1,
	        # etc., in the order 0,1,2,...,-2,-1.

            if antialias:
   	            # Anti-alias. The product P(v1)*P(v2) contributes to the signal at
	            # frequency v1+v2. This can only be relevant if v1+v2 is in the
	            # range of the FROG trace. Otherwise the term is deleted. When N is
	            # even, there is a "maximal frequency", which can correspond
	            # equally well to +N/2 or -N/2 frequency. We assume it's positive.
	            # That's consistent with the convention above for the FROG trace,
	            # that places zero frequency in such a way that more positive
	            # frequencies are visible than negative frequencies for even N.
                vmax = int(np.floor(N/2.)) # Which index in EF uses
                                           # the max positive freq?
                vmin = vmax+1 # Which index in EF uses the most negative freq?
                for n in range(1,vmax+1):
                    EF[n,vmax-(n-1):vmax+1] = 0
                for n in range(vmin,N):
                    EF[n,vmin:vmin+(N-n)] = 0

            EF = np.fliplr(EF)

            # Row rotation...analogous to Eqns (10)-->(11) of Kane1999
            for n in range(0, N):
                EF[n,:] = np.roll(EF[n,:], -n)

 	        # FT each column
            EF = np.fft.ifft(EF, None, axis=0)
	        # Right now the columns are in the order N*v=(N-1,N-2,...,1,0), and rows
	        # are in the order tau=0,-1,...,1. (These are all mod N.)
            EF = np.flipud(np.fliplr(EF))
	        # Now columns are N*v=(0,1,2,...-1) and rows are tau=1,2,3,...,0.
            EF = np.roll(EF, int(np.ceil(N/2.)), 0)
            EF = np.roll(EF, int(np.ceil(N/2.)-1), 1)

	        # Now zero frequency & zero delay is at (ceil(N/2.),ceil(N/2.))
            # as desired
            EF = np.transpose(EF)

	        # Generate FROG trace (= |field|^2)
            F = np.square(np.absolute(EF))

            return F, EF



    def guessPulse(
        self,
        EF: np.ndarray,
        lastPt: np.ndarray,
        domain: int=0,
        antialias: int=0,
        PowerOrSVD: int=0):
        """
        guesspulse: Extracts the pulse as a function of time, starting with a FROG
        FIELD (i.e. complex amplitude), and the previous best-guess pulse. Uses
        either "power method" from Kane1999 or SVD method from Kane1998.
        """

        N = len(EF[0])


        if domain==0:
            # Do the exact inverse of the procedure in makeFROG...
            # Undo the line:
            # EF = np.roll(np.fft.fft(EF,None,axis=0),int(np.ceil(N/2.)-1),0)
            EF = np.fft.ifft(np.roll(EF, -int(np.ceil(N/2.)-1),0), None, axis=0)

            # Undo the line: EF = np.fliplr(np.fft.fftshift(EF,1))
            EF = np.fft.ifftshift(np.fliplr(EF),1)

            # Undo the lines:
            #      for n in range(0,N):
            #          EF[n,:] = np.roll(EF[n,:],-n)
            for n in range(0,N):
                EF[n,:] = np.roll(EF[n,:],n)
            # Now EF is the "outer product form", see Kane1999

            if antialias:
                # Anti-alias in time domain. See makeFROG for explanation.
                EF = EF - np.tril(EF,-np.ceil(N/2.)) - np.triu(EF,np.ceil(N/2.))

            if PowerOrSVD==0: # Power method
                #lastPt = np.transpose(lastPt) # Make column array
                Pt = np.dot(EF,np.dot(np.conjugate(np.transpose(EF)),lastPt))
                #Pt = np.transpose(Pt) # Change to row array again
            else:             # SVD method (does not work???)
                U, S, V = np.linalg.svd(EF)
                Pt = U[:,0]
                Pt = Pt.reshape(N, 1)

            # Normalize to Euclidean norm 1
            Pt = Pt / np.linalg.norm(Pt)
            return Pt


        if domain==1:
	        # Do the exact inverse of the procedure in makeFROG...
	        # Undo the line: EF = np.transpose(EF)
            EF = np.transpose(EF)

            # Undo the lines:
            # EF = np.roll(EF,int(np.ceil(N/2.)),0)
            # EF = np.roll(EF,int(np.ceil(N/2.)-1),1)
            EF = np.roll(EF,-int(np.ceil(N/2.)-1),1)
            EF = np.roll(EF,-int(np.ceil(N/2.)),0)
            # Undo the line: EF = np.flipud(np.fliplr(EF))
            EF = np.flipud(np.fliplr(EF))
            # Undo the line: EF = np.fft.ifft(EF,None,axis=0)
            EF = np.fft.fft(EF,None,axis=0)
            # Undo the lines:
            # for n in range(0,N):
            #     EF[n,:] = np.roll(EF[n,:],-n)
            for n in range(0,N):
                EF[n,:] = np.roll(EF[n,:],n)
	        # Undo the line: EF = np.fliplr(EF)
            EF = np.fliplr(EF)
            # Now we're up to the lines in makeFROG:
            # PtFFT=np.fft.fft(Pt,axis=0); EF = np.outer(PtFFT,PtFFT)

            if antialias:
	            # Anti-alias in frequency domain. See makeFROG for explanation.
                vmax = np.floor(N/2.)
                vmin = vmax+1
                for n in range(2,vmax+1):
                    EF[n,vmax-(n-1):vmax+1] = 0
                for n in range(vmin,N):
                    EF[n,vmin:vmin+(N-n)] = 0


            if PowerOrSVD==0: # Power method
                lastPtFFT = np.fft.fft(lastPt,axis=0)
                Pt = np.fft.ifft(np.dot(EF,np.dot(np.conjugate(np.transpose(EF)),
                                                  lastPtFFT)),axis=0)
            else: # SVD method
                U, S, V = np.linalg.svd(EF)
                Pt = np.fft.ifft(U[:,0])
                Pt = Pt.reshape(N,1)

            Pt = Pt / np.linalg.norm(Pt)  # Normalize to Euclidean norm 1

            return Pt




    def retrievePhase(
        self, Fm: np.ndarray=None, dtperpx: float=None, # Will be set by prepFROG
        units=None, signal_data=None, signal_label=None, signal_title=None,
        signal_axis=None):
        """
        Retrieves the phase using the prepared FROG trace Fm.
        This method makes use of makeFROG and prepFROG.
        Arguments:
        Fm -- prepared frog trace from prepFROG
        dtperpx -- time step between pixels
        units --
        signal_data --
        signal_label --
        signal_title --
        signal_axis --
        """

        print_started_message()

        if Fm is None:
            Fm = self.Fm

        N = len(Fm[0])

        # Time interval per pixel
        if dtperpx is None: dtperpx = self.dtperpx

        # Frequency interval per pixel
        dvperpx = 1 / (N*dtperpx)

        if units is None:
            dtunit = self.units[0]
            dvunit = self.units[1]
        print(f'prepFROG: dt: {dtperpx}{dtunit} dv: {dvperpx}{dvunit}')

        # x-axis labels for plots
        tpxls = make_axis(N, dtperpx)
        # y-axis labels for plots
        vpxls = make_axis(N, dvperpx)

        # Emit axis scale information
        if signal_axis is not None:
            signal_axis.emit(tpxls, vpxls)

        # Maybe you only want to display part of the plot range, to zoom in
        # on the interesting stuff. If so, edit the following lines...
        tplotrange = [np.min(tpxls), np.max(tpxls)]
        vplotrange = [np.min(vpxls), np.max(vpxls)]


        # Normalize FROG trace to unity max intensity
        Fm = Fm/np.amax(Fm)

        # # Make a randam guess for a seed if no external seed has been loaded.
        # Pt = self.get_seed(mode="gauss")
        Pt = self.get_seed(frog=Fm)

        ###################
        # Start main part #
        ###################

        # Generate FROG trace
        iteration = 0

        makeFROGdomain = self.method[0]
        makeFROGantialias = self.method[1]
        #guesspulsedomain = self.method[2] # not used
        #guesspulseantialias = self.method[3] # not used

        # EFr is reconstructed FROG trace complex amplitudes ( Fr=|EFr|^2 )
        Fr, EFr = self.makeFROG(Pt, makeFROGdomain, makeFROGantialias)

        # Calculate FROG error G, see DeLong1996
        Fr = Fr * calc_alpha(Fm, Fr) #scale Fr to best match Fm, see DeLong1996
        # G=RMS difference in entries of Fm and alpha*Fr
        # (where Fm is normalized so max(Fm)=1 and alpha is whatever
        # value minimizes G.) See DeLong1996
        G = rms_diff(Fm, Fr)

        if signal_data is not None and signal_label is not None:
            signal_data.emit(0, Fm)
            signal_label.emit(self.units)

        #  --------------------------------------------------
        #  F R O G   I T E R A T I O N   A L G O R I T H M
        #  --------------------------------------------------

        while G>self._GTol and iteration<self.max_iter:
            # Keep count of no. of iterations
            iteration += 1

            # Check method to use. Have to run this inside the loop because method
            # may vary depending on iter.
            makeFROGdomain = self.method[0]
            makeFROGantialias = self.method[1]
            guesspulsedomain = self.method[2]
            guesspulseantialias = self.method[3]

            # Update best-guess EFr: Phase from last makeFROG, amplitudes from Fm.
            # Change absolute values of EFr to match Fm (keep phase the same)
            # and avoid dividing by zero.
            EFr = EFr * np.sqrt(
                np.divide(Fm, Fr, out=np.zeros_like(Fm), where=Fr!=0)
                )
            # Extract pulse field from FROG complex amplitude
            #testPt = Pt

            Pt = self.guessPulse(EFr, Pt, guesspulsedomain, guesspulseantialias)

            ### Keep peak centered... not necessary, but this helps when visually
            ### comparing and understanding reconstructions.
            if True:
                # Weighted average to find center of peak
                centerindex = (
                    np.sum(
                        np.arange(1, N+1).reshape(N, 1) * np.absolute(np.power(Pt, 4))
                        ) / np.sum(np.absolute(np.power(Pt, 4))))
                Pt = np.roll(Pt, -int(np.round(centerindex-N/2.)))

            # Make a FROG trace from new fields
            Fr, EFr = self.makeFROG(Pt, makeFROGdomain, makeFROGantialias)

            # Calculate FROG error G, see DeLong1996
            # Scale Fr to best match Fm, see DeLong1996
            Fr = Fr * calc_alpha(Fm, Fr)
            G = rms_diff(Fm, Fr)

            print_iter_update(iteration, G)

            # Create plotting data
            # time domain
            tPt_data = 2*np.pi*get_norm_intensity(Pt[:, 0])
            tPt_angle = np.angle(Pt[:,0])+np.pi
            # frequency domain
            FFTPt = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Pt), axis=0))
            vPt_data = 2*np.pi*get_norm_intensity(FFTPt[:,0])
            vPt_angle = np.angle(FFTPt[:,0])+np.pi
            if signal_data is not None and signal_title is not None:
                signal_data.emit(1, Fr)
                signal_data.emit(2, tPt_data)
                signal_data.emit(3, tPt_angle)
                signal_data.emit(4, vPt_data)
                signal_data.emit(5, vPt_angle)
                signal_title.emit(iteration, G)

        #    self.screenshot()
        #  ------------------------------------------------------------
        #  E N D   O F   A L G O R I T H M
        #  ------------------------------------------------------------


        # Save reconstructed FROG trace in attributes
        self.Pt = Pt
        self.Fr = Fr
        print_finished_message()

        intensity = get_norm_intensity(Pt.reshape(N,))
        print(f'Pulse width: {get_fwhm(intensity, tpxls):.3f} ps.')

        return Pt, G, Fr

    def ePIE_fun_FROG(
        self, I: np.ndarray=None, dt: np.ndarray=None, df: np.ndarray=None,
        signal_data=None, signal_label=None, signal_title=None,
        signal_axis=None):
        """
        Function the reconstructs a pulse function (in time) from a
        SHG FROG trace by use of the Ptychographic algorithm.
        No prior knowledge needed.

        Arguments:
        I       =   float dim 128x128, Experimental / Simulated SHG FROG Trace
        dt       =   float dim 128, vector of delays that coresponds to trace.
        df       =   float dim 128, vector of frequencies.

        Returns:
        Obj     =   complex float dim 128, Reconstructed pulse field (in time).
        Ir      =   float dim 128x128, reconstructed FROG trace.
        error   =   float dim 200, vector of errors for each iteration
        """

        if I is None and dt is None and df is None:
            # Use trace prepared by prepFROG
            I = self.Fm
            dt = self.dtperpx
            df = 1/(I.shape[1]*dt)

        # (Frequencies, Delays) = shape
        (N, K) = I.shape
        # Make a time axis
        D = make_axis(K, dt)
        # We need a vertical frequency axis here
        F = make_axis(N, df).reshape(N, 1)
        # Create bool array that yields which frequencies from the frequency axis
        # are good to use. One could for example only measure every second frequency.
        # then we would use the modulo with 2 here.
        # Originally Fsupp was an argument of this function.
        # It is currently not implemented.
        Fsupp = np.array([True if i%1 == 0 else False for i in range(N)])

        # Check input:
        n_freq = N
        n_delay = K
        assert I.shape == (n_freq, n_delay)
        assert I.dtype == np.dtype('float64')
        assert D.shape == (n_delay,)
        assert D.dtype == np.dtype('float64')
        assert Fsupp.shape == (n_freq,)
        assert Fsupp.dtype == np.dtype('bool')
        assert F.shape == (n_freq, 1)
        assert F.dtype == np.dtype('float64')

        # Obj = self.get_seed(mode="autocorr", frog=I)
        Obj = self.get_seed(frog=I)

        # del1 = 1e-3
        # del2 = 2e-6
        error = 1
        # This will be the reconstructed trace
        Ir = np.zeros(I.shape)

        # Send axis to plot
        if signal_axis is not None:
            signal_axis.emit(D, np.concatenate(F))

        if signal_data is not None and signal_label is not None:
            signal_data.emit(0, I)
            signal_label.emit(self.units)

        print_started_message()

        i = 1
        while error > self._GTol and i <= self.max_iter:
            # Produce random array of integers from 0 to K-1
            s = np.random.permutation(range(K))

            # Parameter that controls the strength of the update
            # and is selected randomly in each iteration.
            alpha = np.abs( 0.2 + np.random.randn()/20 )

            for iterK in range(K):
                # Calculate the SHG signal of the field
                temp = shift_signal(Obj, D[s[iterK]], F)
                psi = Obj * temp
                # Fourier transform SHG signal
                psi_n = np.fft.fft(psi, axis=0) / N
                phase = np.exp(1.j*np.angle(psi_n))
                amp = np.fft.fftshift( I[:, s[iterK]].reshape(K, 1) )
                psi_n[Fsupp] = amp[Fsupp] * phase[Fsupp]
                # Experimental soft thresholding, uncomment 2 following lines for try
                # psi_n[~Fsupp] = (np.real(psi_n[~Fsupp]) - del2 * np.sign(np.real(psi_n[~Fsupp]))) \
                #         * (np.abs(psi_n[~Fsupp]) >= del2) \
                #     + 1.j*(np.imag(psi_n[~Fsupp]) - del2 * np.sign(np.imag(psi_n[~Fsupp]))) \
                #         * (np.abs(psi_n[~Fsupp]) >= del2)

                # Get the updated SHG signal
                psi_n = np.fft.ifft(psi_n, axis=0)*N

                # Update the pulse with a weight function
                Uo = temp.conjugate() / np.max( (np.abs(temp)**2) )
                Up = Obj.conjugate() / np.max( (np.abs(Obj)**2) )

                Corr1 = alpha * Uo * (psi_n - psi)
                Corr2 = shift_signal(alpha * Up * (psi_n - psi), -D[s[iterK]], F)

                Obj = Obj +  Corr1 + Corr2
                Ir[:, s[iterK]] = np.abs( np.fft.fftshift( np.fft.fft(Obj * temp, axis=0)/N ) )[:,0]

                if iterK % K == 0:
                    error = np.sqrt(np.sum(np.abs( Ir[np.fft.fftshift(Fsupp),:] - I[np.fft.fftshift(Fsupp),:] )**2 )) \
                        / np.sqrt(np.sum(np.abs(I[np.fft.fftshift(Fsupp),:] )**2 ))
                    #print(f'Iter:{i:3d}  IterK:{iterK}  alpha={alpha:.4f} Error={error:.4f}')
                    print_iter_update(i, error)

                    if signal_data is not None and signal_title is not None:
                        # Prepare and send data for plotting
                        time_field = Obj.reshape(N,)
                        time_int_scaled = 2*np.pi*get_norm_intensity(time_field)
                        time_phase = np.angle(time_field)+np.pi
                        signal_data.emit(1, Ir)
                        signal_data.emit(2, time_int_scaled)
                        signal_data.emit(3, time_phase)
                        signal_title.emit(i, error)

            i += 1


        self.Fr = Ir
        self.Pt = Obj
        print_finished_message()

        intensity = get_norm_intensity(Obj.reshape(N,))
        print(f'Pulse width: {get_fwhm(intensity, D):.3f} ps.')

        return Obj, error, Ir


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import pathlib
    import imageio
    import yaml


    data_path = pathlib.Path(__file__).parents[1] / 'data'

    # make Phase retrieval instance
    pr = PhaseRetrieval()
    pr.max_iter = 50
    trace = np.array(imageio.imread(data_path / 'prep_frog.tiff')).astype('float64')
    with open(data_path / 'prep_meta.yml', 'r') as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)
    dt = meta['ccddt']
    dF = meta['ccddv']
    t = make_axis(128, dt)

    #pr.prepFROG(ccddt=dt, ccddv=dF, ccdimg=trace)
    pr.load_seed(data_path / 'seed')
    field, error, frog_reconstructed = pr.retrievePhase(Fm=trace, dtperpx=dt)
    #field, error, frog_reconstructed = pr.ePIE_fun_FROG(I=trace, dt=dt, df=dF)
    #FileHandler().save_seed(field)
    plt.figure('Frog reconstructed')
    plt.imshow(frog_reconstructed)
    plt.figure('amplitude')
    plt.plot(t, 2*np.pi*get_norm_intensity(field.reshape(128,)))
    plt.plot(t, np.angle(field)+np.pi)
    plt.show()
