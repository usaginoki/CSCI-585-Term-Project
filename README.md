This repository contains the term project for CSCI 585 Computer Vision course.

We should have forked [https://github.com/rishiswethan/Video-Audio-Face-Emotion-Recognition/tree/main] but had LFS issues, so here we are.

Installation:

We created a virtual environment and handled dependencies with poetry: .toml file is already here.

For prediction:

1. Train model (read further) or download weghts from: [https://drive.google.com/drive/folders/1CUcuT7AiyLA0SZAOTdwEeu_APJlzwN60?usp=sharing]
2. Place the file for prediction into input_files folder in the main directory of the project
3. execute run.py and select Transformer model (number 4)

For hyperparameter tuning:

1. structure data accordingly (read further)
2. execute transformer_optimizer.py (or you could execute run.py and follow prompts, but our script is more handy since it can be launched in the background)

For training:

1. data should be structured like shown below
2. execute transformer_model_trainer.py (or you could execute run.py and follow prompts, but our script is more handy since it can be launched in the background)



Note for professor:

Code for the model is in folder source/audio_face_transformer/

Definition of the model in model.py and configurations are in tr_config.py



Data structure for training and hyperparameter tuning:
```html
   data 
    ├───training_AV                   // Audio visual data used for training the combined model
    │   ├───RAVDESS
    │   │   ├───train
    │   │   │  ├───RAVDESS_0_Neurtal.mp4                  // Example
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.mp4    // Correct format
    │   │   │  ├───...other videos
    │   │   ├───test
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.mp4
    │   // Other datasets must be added in the same format and added in the config.py file
    │    
    ├───training_faces                // Facial data used for training the landmarks and image based emotion detection model
    │   ├───FER
    │   │   ├───train
    │   │   │  ├───FER_0_Angry.png
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.png    // Correct format
    │   │   │  ├───...other images
    │   │   ├───test
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.png    // Test data from all faces datasets are used
    │            
    ├───extracted_audio               // Audio data used for training the audio model
    │   ├───RAVDESS
    │   │   ├───train
    │   │   │  ├───RAVDESS_0_Neurtal.wav                  // Example
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.wav    // Correct format
    │   │   │  ├───...other audio
    │   │   ├───test
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.wav    // We only use testing data from CREMA-D and TESS, this is just an example. The downloaded data will be empty here.
```
