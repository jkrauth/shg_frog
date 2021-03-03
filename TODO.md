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
- Would it make sense to move from a MV pattern to MVC?
- Subclass camera and Ando from a parent spectrometer class
- Create parameter tree in a more elegant way, then one could also better include the 'save settings' point from above
- The first time shg_frog is started, ask to enter config into a form and save that in .shg_frog/config.yml in home directory. Always check first if it exists, otherwise ask to give that info. Allow for loading defaults. Those have to be included inside the package with mainfest.in
- Example and seed data also has to be inside the package in a data folder. At startup they should be put into a shg_frog_data folder in the home folder. Seeds should be saved in .shg_frog/seeds/...
- Load measurement should then check if shg_frog_data/example or shg_frog_data/<date>/<measurement> exists, if so show open window of data directory.