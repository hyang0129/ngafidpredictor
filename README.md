# NGAFID Flight Maintenance Predictor

This tool currently predicts whether or not a flight is before or after maintenance.
It has been trained on flights that lasted at least 1 hour, so it may not perform
well for very short flights. 

To use the predictor, follow these steps. 

1. Clone the repo 
2. Install the requirements.txt to your python (version 3.7+)
3. cd to the repo directory and run ```python main.py``` to process the examples
4. run ```python main.py --inputdirectory example_flights``` to process all flights in
a particular directory (replace example_flights with your directory)

All results will be written to the results.csv file. Subsequent runs will overwrite
previous results.