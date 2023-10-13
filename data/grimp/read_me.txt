File explanation

HS4_noise_event_data.h5:
Contains the data of 14837 noise events (102179 readings) induced during experiment HS4 at the Grimsel Test Site. Each reading is stored in X with 801 datapoints resulting in an array of 102179x801. Y is the waveform identifier and set to: 3 (Noise)
Keys and dimensions of arrays are ['SNR'(102179,), 'X', 'Y'(102179,), 'dist'(102179,), 'eve_id'(102179,), 'mags'(102179,)]

And contains the data of all located 3094 events (20383 readings) induced during experiment HS4 at the Grimsel Test Site. Each reading is stored in X with 801 datapoints resulting in an array of 20383x801. Y is the waveform identifier and set to: 1 (P-wave)
Keys are ['SNR'(20383,), 'X', 'Y'(20383,), 'dist'(20383,), 'eve_id'(20383,), 'mags'(20383,)]

HS4_training_data_v01.h5:
Contains training data, 85% of 20383 readings of each noise and P-waves randomly distributed and normalized.

HS4_testing_data_v01.h5:
Contains testing data, 15% of 20383 readings of each noise and P-waves randomly distributed and normalized.
