# Dissertation
Learning to estimate 3D ground reaction forces from multi-view 2D human motion.

To run the code, you need to first download the ForcePose data set from:
https://drive.google.com/file/d/16gE9JlcLt1QWJ3woDCYFLfOCduC2WOeL/view

Extract the force_pose.tar GZ File. 
For help to do this, visit: https://www.wikihow.com/Extract-a-Gz-File

Once extracted, you will see a folder named "force_pose". The folder should contain a "readme.txt" along with "train.json" and "val.json".

Please drag the "force_pose" folder into the empty Datasets folder. 

Once completed, you should be able to run trainMultvariantLSTM.py to train an LSTM regression model if you have the correct python libaries installed. 

Please ensure you specifiy the desired training parameters located on lines 14-16.

Please run testMultvariantLSTM.py to test your trained model. However please make sure you specifiy the correct model name in the python variable "model_names" located on line 156. Model names can be found in the "Models" folder, just copy the folder name of the model you want to test.


