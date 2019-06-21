#Bound box and time

boundingbox='-61.0 -83.0 -34 -76.9'
time='2018-10-29T00:00:00,2018-10-31T00:00:00'
output='C:/Users/mjvw74/Documents/somedata/' #change output
pathtopythonfile = 'C:/Users/mjvw74/Documents/ground2float/getATL/run_ATL06_query.py'


#import
#!/usr/bin/env python
import os
import sys
import getpass
import socket
import requests
import json
import subprocess
import zipfile

try:
    os.makedirs(output)
except:
    pass

#####get token

# Earthdata Login credentials

# Enter your Earthdata Login user name
uid = 'bertiemiles1990'
# Enter your email address associated with your Earthdata Login account
email = 'a.w.j.miles@durham.ac.uk'
pswd = getpass.getpass('')


# Request token from Common Metadata Repository using Earthdata credentials
token_api_url = 'https://cmr.earthdata.nasa.gov/legacy-services/rest/tokens'
hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)

data = {
    'token': {
        'username': uid,
        'password': pswd,
        'client_id': 'NSIDC_client_id',
        'user_ip_address': ip
    }
}
headers={'Accept': 'application/json'}
response = requests.post(token_api_url, json=data, headers=headers)
token = json.loads(response.content)['token']['id']
print(token)


#make token correct format
tokennumber='"<id>'+token+'</id>"'
tokennumber


rundownload = 'python '+pathtopythonfile+' -s -token '+tokennumber+' -b '+boundingbox+' -v 001 -t '+time+' -o '+output

p=subprocess.Popen(rundownload, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

import os, zipfile

dir_name = output
extension = ".zip"

os.chdir(dir_name) # change directory from working dir to dir with files

for item in os.listdir(dir_name): # loop through items in dir
    if item.endswith(extension): # check for ".zip" extension
        file_name = os.path.abspath(item) # get full path of files
        zip_ref = zipfile.ZipFile(file_name) # create zipfile object
        zip_ref.extractall(dir_name) # extract file to dir
        zip_ref.close() # close file
        os.remove(file_name) # delete zipped file
