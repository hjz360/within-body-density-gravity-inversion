"""Data root for all analysis scripts. Set the environment variable VDNET_DATA to the folder unpacked from the
Zenodo data record; by default <repo>/data is used."""
import os
DATA = os.environ.get('VDNET_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
