import pickle
from importlib.resources import files

def FakeTorino():
    resource_path = files('src.clp_zne.qpu_info').joinpath('torino_backend.pkl')
    with open(resource_path, 'rb') as f:
        backend = pickle.load(f)
    return backend