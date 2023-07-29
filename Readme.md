## 0. Authors:
Ketan V.,
Nanda K.,
S. Gowtham,
Karthik C.,
Dheeraj V.,

## I. Spoken Language Identification
### Project Description
```
This project, developed for the NVIDIA AI Hackathon 2019, is a real-time spoken language identification system for diverse Indian languages. Built using Python and Flask for the web interface, it uses a Convolutional Recurrent Neural Network (CRNN) for language detection.

We tackled a unique challenge of identifying and transitioning between multiple Indian languages in real-time during a conversation. This hands-free multilingual conversation facilitator provides an impressive and practical solution to language barriers, particularly in a linguistically diverse nation like India.

The project stands as a testament to our team's innovative problem-solving abilities, as we effectively addressed a significant data scarcity issue. We conducted on-the-spot interviews to collect live language samples, thereby creating a dataset that enhanced the performance of our language detection model.

This innovative solution earned us the 3rd position at the NVIDIA AI Hackathon 2019.
```
### Features
```
Real-time spoken language identification for diverse Indian languages
Live audio capture and analysis
Quick language transition (within 1.5 seconds)
Hands-free multilingual conversations
```
### Technologies Used
```
Python
Flask
Convolutional Recurrent Neural Network (CRNN)
```
### Acknowledgements
```
We're grateful to everyone who contributed to this project and those who provided language samples. This project wouldn't have been possible without the NVIDIA AI Hackathon 2019 platform, which gave us the opportunity to solve a unique and challenging problem..
```
## II. File Structure:

    AIH19T-0161
    |-- README.txt
    |-- requirements.txt
    |-- Model
    |-- |-- Tuning
    |-- |-- |-- Models
    |-- |-- |-- `-- *Weights.pth     '*' represents something before
    |-- |-- |-- `-- *Confusion.npy
    |-- |-- |-- `-- *Metrics.txt
    |-- |-- |-- createCombinations.py
    |-- |-- |-- config.yaml
    |-- |-- |-- CRNN.py
    |-- |-- |-- dataLoader.py
    |-- |-- |-- train.py
    |-- |-- |-- test.py
    |-- |-- |-- main.py
    |-- |-- |-- train.sh
    |-- |-- |-- test.sh
    |-- |-- Models
    |-- |-- `-- *Weights.pth     '*' represents something before
    |-- |-- `-- *Confusion.npy
    |-- |-- `-- *Metrics.txt
    |-- |-- config.yaml
    |-- |-- CRNN.py
    |-- |-- dataLoader.py
    |-- |-- main.py
    |-- |-- train.py
    |-- |-- test.py
    |-- |-- train.sh
    |-- |-- test.sh
    |-- UI
    |   |-- gui.py
    |   |-- templates
    |   |   |-- upload.html
    |   |   `-- gui.html
    |   |-- uploads
    |   |   `-- sample_recordings.wav
    |   |-- main.py
    |   |-- CRNN.py
    |   |-- data_upload.py
    |   `-- weights
    |       `-- weights.pth
    `-- wheel_dependencies
    |-- `-- <.whl> files
    |-- Dataset
    |-- |-- TAM
    |-- |-- `-- <Extracted Data>
    |-- |-- GUJ
    |-- |-- `-- <Extracted Data>
    |-- |-- MAR
    |-- |-- `-- <Extracted Data>
    |-- |-- HIN
    |-- |-- `-- <Extracted Data>
    |-- |-- TEL
    |-- |-- `-- <Extracted Data>
    |-- |-- TestData
    |-- |-- |-- <*.png> files         Spectrogram images  
    |-- |-- TrainData
    |-- |-- |-- <*.png> files         Spectrogram images
    |-- |-- audio2images.py
    |-- |-- TestInput.txt
    |-- |-- TrainInput.txt
    |-- |-- noise.wav
    |-- Trained
    |-- |-- Models
    |-- |-- |-- <*.pth> files     Weights
    |-- |-- ConfusionMatrics
    |-- |-- |-- <*.npy> files     Confusion Matrix
    |-- |-- Metrics
    |-- |-- |-- <*.txt> files     F1 Score
    |-- |-- |-- OriginalTestResult.out 
    |-- |-- |-- 
    |-- |-- |-- 

## III. Installing the dependencies:
    
    Install the requirements using the command*:
```
        pip3 install -r requirements.txt
	conda install -c conda-forge pyaudio         // Try installing them with pip3 we got error so we installed using conda
	conda install pyqt4
	pip3 install flask
	pip3 install librosa
```

    We got many permission issues:
	So we had to create a new conda env and install everything using conda.
	We hope the person who is checking will have all permissions in the system and could install all dependencies.

## IV. Running the Web-App:
    P.S. The cluster didn't have a display environment and hence couldn't load the web application. Ensure that the web app is run on a system with a proper graphics interface.
    To run the web appliation go to ./UI/ folder and run the following command:
	module load python3.6
        python3 main.py
    This starts a local developemnt server at local host: 5000. Navigate to local host:5000 to use the web application.
    Go to a browser open and type localhost:5000/ to get the page.

    There are two options in the web app:
	1) For offline, click on browse to select the audio file only in the .wav format, click submit then the top three languages are shown in the left.
	2) For online, click on online option , you will be redirected to popup 'python gui' click on record option it will display the latest languages in the front.

## V. Reproducing the models:

    All the preprocessing, Training, Testing and Hyper parameter tuning will be done inside AIH19T-0161/Model folder.

    Stage 0: Preprocessing
	Note: The Dataset folder actually has the data we need as input. So if you are taking the whole folder, this data preprocess step is not mandatory.
	Move into the folder (cd Dataset)
        The folder AIH19T-0161/Dataset/<Corresponding Lanuguages> contains all the raw audio samples in .wav format
        These audio samples are fed to a script named audio2images.py which produces spectrogram files and divides them into "TrainInput" and "TestInput".
        This also produces two text files named "TrainData.txt", "TestData.txt" which contain the labels for the corresponding .png files.
        To run the preprocessing step use the following command:
                module load python3.6
                python3 audio2images.py
	Get back to the main folder by running cd ..

    Stage 1: Training
        All the parameters related to python scripts are configured in config.yaml file. 
        Any parameters that need to be changed, should be changed in config.yaml file(for example, file paths).
        The file dataLoader.py contains a class DataLoader which is used to create objects called 'TrainData' and 'TestData'.
        The CRNN.py contains the class named CRNN which is used to create an instance of the untrained model.
        The train.py contains all the scripts required for training(defined as a function).
        Training can be done using the job scripts:
		vi config.yaml and assure that train is 1 and test is 0
		vi main.py ensure that import test is commented out
                sbatch train.sh
	Note: All sbatch parameters are set in the train.sh. (Assuming that the reservation name still exists, if not replace the commands with the proper ones).
        This will call the main.py which creates dataLoader object for train data and initiates the training.
        After training is done, model weights are stored in the Model/Models/ in the name of the 'runName' field specified in config.yaml.

    Stage 2: Testing
        The procedure for testing is similar to training. A dataLoader object is created to test the data called 'TestData'.
        The test.py contains all the scripts required for testing(defined as a function).
        Testing can be done using the job scripts:
		vi config.yaml and assure that train is 0 and test is 1
		vi main.py and uncomment the import test line
                test.sh
        This will call the main.py which creates dataLoader object for test data and initiates the testing.
        Now, a CRNN object is created and is loaded with weights stored in the ./Model/in the name of the 'runName' field specified in config.yaml.
        This prints and stores F1 score and confusion matrix between the predicted and the ground truth value in a separate file 'trialTest.txt'.

    Stage 3: Hyper parameter tuning
        All the parameters related to python scripts are configured in config.yaml file. 
        Any parameters that need to be changed, should be changed in config.yaml file(for example, file paths).
        The file dataLoader.py contains a class DataLoader which is used to create objects called 'TrainData' and 'TestData'.
        The CRNN.py contains the class named CRNN which is used to create an instance of the untrained model.
        The train.py contains all the scripts required for training(defined as a function).
        Training can be done using the job scripts:
		vi config.yaml and assure that train is 1 and test is 0
		vi main.py ensure that import test is commented out
                sbatch train.sh
        This will call the main.py which creates dataLoader object for train data and initiates the training.
        After training is done model weights are stored in the ./Models/in the name of the their index in the combinations list.
	This combinations list can be retrived by pickle loading combos.pkl
		import pickle
		combinations = pickle.load(open('combos.pkl',wb))
	Then testing the run
		vi config.yaml and assure that train is 0 and test is 1
		vi main.py ensure that import test is commented out
                sbatch train.sh
        This will call the main.py which creates dataLoader object for test data and initiates the testing.
        Now, a CRNN object is created and is loaded with weights stored in the ./Model/in the name of the their index in the combinations list.
        This prints and stores F1 score and confusion matrix between the predicted and the ground truth value in a separate file 'tuneTest30.txt'.


## Dependencies
-----------------------------------------

All the dependencies are packaged into requirements.txt file.

Some wheel files are included in the root of the directory for installation. 

https://download.pytorch.org/whl/cu90/torch_stable.html








