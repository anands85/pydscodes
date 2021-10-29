#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pymongo
import pandas as pd
import numpy as np
import io, os, sys, types
import datetime
import nbimporter


# In[2]:


def mongodb_ingest(mongodb_url, 
                    db_name, 
                    collection_name,
                    dir_file_name,
                    ingest_dir=True):
    process = True
    
    start = datetime.datetime.now()
    
    try:
        
        # Making Connection
        myclient = pymongo.MongoClient(mongodb_url)

        # database
        db = myclient[db_name]

        # Created or Switched to collection
        # names: GeeksForGeeks
        Collection = db[collection_name]

        #if ingestion of single file thenn ingest_dir=False and only
        #one file ingested
    
        if ingest_dir:

            files = [f for f in os.listdir(dir_file_name) if os.path.isfile(f)]

            for f in files:

                # Loading or Opening the json file
                with open(f) as file:

                    file_data = json.load(file)

                # Inserting the loaded data in the Collection
                # if JSON contains data more than one entry
                # insert_many is used else inser_one is used
                if isinstance(file_data, list):

                    Collection.insert_many(file_data)

                else:

                    Collection.insert_one(file_data)
        else:

            # Loading or Opening the json file
            with open(dir_file_name) as file:
                
                file_data = json.load(file)

            # Inserting the loaded data in the Collection
            # if JSON contains data more than one entry
            # insert_many is used else inser_one is used
            if isinstance(file_data, list):
                
                Collection.insert_many(file_data)
                
            else:
                
                Collection.insert_one(file_data)
                
    except Exception as err:
        
        print(err)
        
        process = False
       
    end = datetime.datetime.now()
    
    diff = end - start
    
    return process, diff.seconds 


# In[3]:


def mongodb_retrieve(mongodb_url, 
                    db_name, 
                    collection_name,
                    collection_koi):

    process = True
    
    start = datetime.datetime.now()
    
    try:
        
        #https://www.geeksforgeeks.org/how-to-fetch-data-from-mongodb-using-python/

        client = pymongo.MongoClient(mongodb_url)

        # Database Name
        db = client[db_name]

        # Collection Name
        col = db[collection_name]

        y_collect = col.find_one()[collection_koi]

        print(json.dumps(y_collect, indent=4, sort_keys=True))
    
    except Exception as err:
        
        print(err)
        
        process = False
       
    end = datetime.datetime.now()
    
    diff = end - start
    
    return y_collect, process, diff.seconds

