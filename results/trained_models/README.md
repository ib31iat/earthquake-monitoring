# Trained Models
This folder contains our trained models

## EQT Basic
This model is the standard EQT model from seisbench, first trained for four epochs on the full STEAD dataset before being transferlearned for 10 epochs on the limited bedretto data we have.

## EQT Normal
This model is the standard EQT model from seisbench.

## EQT Reduced Encoder
This model has a reduced encoder: only two Res CNN layers and one Bi LSTM layer remain.

## EQT no Res CNN and no Bi LSTM
This model does not contain any resdiual nn layers and no Bi LSTM layers, not even in the encoder.
