from .beauty import BeautyDataset
from .games import GamesDataset
from .auto import AutoDataset
from .toys_new import ToysNewDataset
from .sports import SportsDataset
from .office import OfficeDataset

DATASETS = {
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    AutoDataset.code(): AutoDataset,
    ToysNewDataset.code(): ToysNewDataset,
    SportsDataset.code(): SportsDataset,
    OfficeDataset.code(): OfficeDataset,

}

def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
