# GaTp

Python Version: >= 3.7

For required libraries -> 'pip install -r GaTp/scripts/requirements.txt'\
In addition, pytorch == 1.11.0 with cuda toolkit == 11.3 is required

Datasets folder is in the Shared Drive Folder.\
Needs to be downloaded and placed under 'GaTp/' 

To run a training session from scratch, with default params:\
'python GaTp/main.py -agent_type magat -mode train'  

Model parameters are configured in 'GaTp/yaml_configs/magat.yaml'

If a path error at the start is raised (no config file found), change in GaTp/utils/config.py:\
DATA_ROOT -> path to the 'GaTp/' folder\
APPEND_DATA_ROOT -> set to True
