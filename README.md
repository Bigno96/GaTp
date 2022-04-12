# GaTp

Python Version: >= 3.8

For required libraries -> pip install -r GaTp/requirements.txt

Datasets folder is in the Shared Drive Folder.\
Needs to be downloaded and placed under GaTp/ 

To run a training session from scratch, with default params:\
python GaTp/main.py -agent_type magat -mode train  

Model parameters are configured in GaTp/yaml_configs/magat.yaml

If a path error at the start is raised (no config file found), change in GaTp/utils/config.py:\
PROJECT_ROOT -> path to the GaTp/ folder\
APPEND_PROJECT_ROOT -> set to True
