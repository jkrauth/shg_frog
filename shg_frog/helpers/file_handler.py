"""
Responsible for loading from files and saving to files.
"""
from time import strftime

import pathlib
import yaml
import imageio
import numpy as np

from .data_types import Data


HOME_DIR = pathlib.Path.home()
# These folders will be created if not existent.
CONFIG_DIR = HOME_DIR / ".frog_config"
DATA_DIR = HOME_DIR / "frog_data"


DEFAULT_CONFIG = {
    "camera model": "Manta G-234B NIR",
    "camera id": "DEV_000F314E1E59", # Necessary to select the camera
    "stage port": "/dev/ttyUSB0",
    "pixel size": 5.86, # Given in micron
    "pxls height": 1216, # Number of pixels in vertical
    "pxls width": 1936, # Number of pixels in horizontal
    "center wavelength": 345, # in nm, center wavelength of the pulse
    "focal length": 200, # in mm, lens between grating and camera
    "grating": 0.81, # Grating specified with 0.81nm/mrad
}


def get_unique_path(directory: pathlib.Path, name_pattern: str) -> pathlib.Path:
    """ Creates a unique path with a given pattern, using integer numbering.
    Arguments:
    directory -- pathlib.Path, path to where numbered folders should be
    name_pattern -- str, name of the folder, containing the curly format brackets for numbering.
    """
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


class FileHandler:
    """ Saving and loading data into and from files. """
    name_meta = 'meta.yml'
    name_frog = 'frog.tiff'
    name_config = 'config.yml'
    name_seed = 'seed.dat'

    def _get_new_measurement_path(self) -> pathlib.Path:
        """Returns a path for the next measurement."""
        today = strftime("%Y%m%d")
        today_path = DATA_DIR / today
        new_path = get_unique_path(today_path, 'measurement_{:03d}')
        return new_path

    def save_new_measurement(self, data: Data, config: dict):
        """ Saves data and configuration into a new measurement folder
        Arguments:
        data -- data and metadata of a measurement
        config -- configuration file data of the frog_software
        """
        # Get unique path for new measurement
        measurement_path = self._get_new_measurement_path()
        measurement_path.mkdir(parents=True)
        # Save Frog image with correct bit depth
        if data.meta['bit depth'] == 'Mono8':
            bit_type = np.uint8
        elif data.meta['bit depth'] == 'Mono12':
            bit_type = np.uint16
        imageio.imsave(measurement_path / self.name_frog, data.image.astype(bit_type))
        # Save settings
        with open(measurement_path / self.name_meta, 'w') as f:
            yaml.dump(data.meta, f, default_flow_style=False)
        # Save configuration
        with open(measurement_path / self.name_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def get_main_config(self) -> dict:
        """Load default from custom file if it exists, otherwise create file and
        return default config."""
        config_path = CONFIG_DIR / self.name_config
        if not CONFIG_DIR.exists():
            CONFIG_DIR.mkdir()
            with open(config_path, 'w') as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
            return DEFAULT_CONFIG
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def save_main_config(self, config: dict):
        """ Save current config to file in CONFIG_DIR """
        pass

    def get_measurement_data(self, measurement_path: pathlib.Path) -> Data:
        """ Get config, settings (meta), and image data of an old measurement. """
        # Load settings
        with open(measurement_path / self.name_meta, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        # Load frog image
        frog_image = imageio.imread(measurement_path / self.name_frog)
        data = Data(frog_image, meta)
        # Load configuration
        #with open(measurement_path / self.name_config, 'r') as f:
        #    config = yaml.load(f, Loader=yaml.FullLoader)
        return data

    def load_seed(self) -> np.ndarray:
        """ For the use in the phase retrieval module.
        Load a custom seed for the retrieval class from a file
        Real and Imaginary part need to be in 2 space-separated columns.
        Returns:
        vertical complex array, that can be used as a seed array.
        """
        return np.loadtxt(CONFIG_DIR / self.name_seed).view(complex).reshape(-1, 1)

    def save_seed(self, seed: np.ndarray):
        """ For the use in the phase retrieval module
        Takes the electric field of the reconstructed pulse
        and writes it to a file.
        Real and Imaginary part are written into 2 space-separated columns.
        Argument:
        seed -- vertical complex numpy array: the complex field of a
                retrieved pulse.
        """
        np.savetxt(CONFIG_DIR / self.name_seed, seed.view(float).reshape(-1, 2))
