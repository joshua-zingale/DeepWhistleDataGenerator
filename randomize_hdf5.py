import argparse
import random
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_hdf5', type=str, help='The hdf5 file to be shuffled')
    config = parser.parse_args()
    
    h5file = h5py.File(config.input_hdf5, "r+")

    seed = random.randrange(1,1e10)
    for dataset in h5file:
        random.seed(seed)
        random.shuffle(h5file[dataset])

    h5file.close()

if __name__ == "__main__":
    main()
