#! /usr/bin/env/python3

import h5py

def main():
    training_data_path = "../data/grimp/HS4_training_data_v01.h5"
    noise_data_path = "../data/grimp/HS4_noise_event_data.h5"
    with h5py.File(training_data_path, "r") as training_data, h5py.File(noise_data_path, "r") as noise_data:
        print(list(training_data.keys()))
        print(list(noise_data.keys()))


if __name__ == "__main__":
    main()
