# README #

Software for a pulse measurement using the __SHG FROG__ technique.

> Want to know more about FROG?   
> R. Trebino, Frequency-Resolved Optical Gating: the Measurement of Ultrashort Laser Pulses, Kluwer, Boston (2002)

### What is this repository for? ###

* This repository aims to provide the code necessary to perform a pulse measurement using the 
  second-harmonic generation type of frequency-resolved optical gating, in short SHG FROG.
  Commercial devices exist, but are rather expensive. A home-build device can save a lot of 
  money.
  The code in this repository provides: 
    * Connection with, control over and readout of a motorized translation stage 
    * Connection with and readout of a frequency resolved detection device
    * Measurement of the so-called FROG trace
    * Phase retrieval algorithm from the trace, in order to obtain the original pulse shape.
      The phase retrieval algorithm is based on the Matlab package by Steven Byrnes.
* The software runs in two modes: The frequency-resolved detection can be either a commercial spectrometer (e.g. ANDO), or a grating-CCD combination.


### How to set up? ###

#### Software ####
* Python 3.7
* the following packages are used in the repo:
    * pyQt5
    * pyQtGraph v0.10.0
    * the pyVISA package with the pyVISA-py backend
      (connection to a Newport translation stage via a Serial Port)
    * the plx_gpib_ethernet package [GitHub](https://github.com/nelsond/prologix-gpib-ethernet)
      (connection to the Spectrum Analyzer (GPIB) via an Ethernet Adapter)
    * the pymba package [https://github.com/morefigs/pymba]
      (connection to the ALLIED VISION CCD camera via Ethernet)
    * pyYAML
* run `python Examples/start_gui.py` to start the software, 
the option `python Examples/start_gui.py test` uses virtual devices.
* calibration
    * The calibration of the camera is done via the config.yml file in the Examples/config folder,
      where the according numbers have to be set.


#### Hardware ####

##### Newport Stage #####
* The best and cheapest solution to get a motorized translation stage in our case was to buy an
  actuator (Newport TRA25-CC) and combine it with an existing manual stage.
* The minimum incremental motion of this device is 0.2 micron. For laser pulses with a pulse length
  of <20fs a better actuator might be considered.

##### ANDO Spectrometer #####
We use the ANDO Spectrometer AQ-6315A.  
* In the case of a low repetition rate the ANDO spectrometer has to run in pulsed mode. 
* The spectrometer is also used to calibrate the grating-CCD version of the FROG. 

##### Allied Vision CCD Camera #####
We use the Allied Vision Manta G235B CCD camera. Other Allied Vision Manta GigE cameras should
also work. The original software for this camera is the Vimba Software, written in C/C++. The camera is
included in the FROG software via the Pymba package, a python wrapper for Vimba.  
* Trigger the camera externally if laser has a low repetition rate (e.g. 30Hz). 
  External trigger (TTL) has to be at least 6 microseconds long with an amplitude of min. 2V.
  More information on triggering the camera is found in the Manta Tech Manual from page 193 on.

#### Start the program

The software is started with 

```bash
python Examples/start_gui.py
```

A testing mode can be used by

```bash
python Examples/start_gui.py test
```

In this case virtual devices are used and no connection to real devices is required.

### Contact ###

* Repo owner:  Julian Krauth, j.krauth@vu.nl
* Institution: Vrije Universiteit Amsterdam, Faculty of Sciences, The Netherlands
