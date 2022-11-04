# mdiabetes-analysis-utils
This git repo stores the code for handling the mdiabetes AI-experiment
data for analyzing message transmission, response rates, and participant behavior

### To use these utils
1) create your analytics directory, for example: `/home> mkdir myanalytics && cd myanalytics` 
2) in your directory clone the repo as `utils`: `/home/myanalytics> gh repo clone skippyelvis/mdiabetes-analysis-utils utils`
3) create symlink to arogya_content folder: `/home/myanalytics> ln -s /home/users/jwolf5/mdiabetes/PROD/arogya_content .`
4) create symlink to local_storage folder: `/home/myanalytics> ln -s /home/users/jwolf5/mdiabetes/PROD/local_storage .`
5) make sure your directory is set up correctly
```
/home/myanalytics> ls
local_storage/* arogya_content/* utils/
```
6) create new file `myfile.py` and import the `utils` module
```
#/home/myanalytics/myfile.py
from utils.behavior_data import BehaviorData
bd = BehaviorData()
print(bd.dimensions)
```
