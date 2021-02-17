# TODO

## Functionality

- add save or load a custom seed function
- save and load last personal settings
- fully implement ANDO spectrometer option
- Add option to symmetrize the trace
- Add a [ptychographic](https://oren.net.technion.ac.il/homepage/) algorithm as option
- Maybe add an input for the center wavelength
- Add warning if camera is saturating.

## Structure and Repository

- set up automated testing
- Add Docs with drawing of example hardware setup
- Restructure for clear MVC pattern, currently controller is split into the other two.
- Subclass camera and Ando from a parent spectrometer class
- Create parameter tree in a more elegant way
- Should secondary windows become QDialogs? Maybe then no extra custom threads are needed?