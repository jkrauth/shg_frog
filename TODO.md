# TODO

## Functionality

- add save or load a custom seed function
- fully implement ANDO spectrometer option
- Add option to symmetrize the trace
- Maybe add an input for the center wavelength
- Add warning if camera is saturating.

## Structure and Repository

- add more custom types, e.g. of vertical and horizontal np.arrays, frog trace, etc...
- set up automated testing
- Add Docs with drawing of example hardware setup
- Would it make sense to move from a MV pattern to MVC?
- Create parameter tree in a more elegant way, then one could also better include the 'save settings' point from above
- example data files have to be included inside the package with mainfest.in and can be loaded into the frog as a standard from the beginning.
- Seeds should be saved in .frog_config/seeds/...