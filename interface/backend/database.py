from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.server import server_config

class Database:
    def __init__(self):
        self.uri = server_config.db_uri
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
    
    def get_client(self):
        return self.client
    
    def get_database(self):
        db_name = "Fault_History"
        db = self.client[db_name]
        
        # Ensure database exists by checking and creating if needed
        # MongoDB creates databases lazily when you first write to a collection
        # So we'll create an empty collection if the database doesn't exist yet
        if db_name not in self.client.list_database_names():
            # Create a collection to ensure the database is created
            # We'll use a system collection or create a placeholder
            try:
                db.create_collection("_metadata")
                print(f"Created database: {db_name}")
            except Exception as e:
                # Collection might already exist or database was created by another process
                pass
        
        return db
    
    def insert_alert(self, alert: dict):
        asset_id = alert["asset_id"]
        # Extract fan number from "Fan_1" format
        fan_id = int(asset_id.split("_")[1])
        fault_type = alert["message"]
        timestamp = alert.get("ts") or datetime.now().timestamp()
        document = {
            "fan_id": fan_id,
            "fault_type": fault_type,
            "timestamp": timestamp
        }
        try:
            self.get_database().get_collection("Fault_History").insert_one(document)
            print(f"Alert inserted successfully: {document}")
        except Exception as e:
            print(f"Error inserting alert: {e}")
