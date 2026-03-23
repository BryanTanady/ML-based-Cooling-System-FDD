from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import sys
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
load_dotenv()

class Database:
    def __init__(self):
        self.uri = os.getenv("DB_URL")
        # Ensure that the MongoDB server is running and accessible
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))
        try:
            self.client.admin.command('ping')
            print("================================================DATABASE CONNECTED================================================")
            print("Pinged your deployment. You successfully connected to MongoDB!")
            print("================================================================================================================")
            db_name = os.getenv("DB_NAME")
            if db_name not in self.client.list_database_names():
                self.client.create_database(db_name)
                print("================================================DATABASE CREATED================================================")
                print(f"Database {db_name} created successfully.")
                print("================================================================================================================")
        except Exception as e:
            print("Error connecting to MongoDB:", e)
            print("================================================DATABASE NOT CONNECTED================================================")
            print("Please check if the MongoDB server is running and accessible. EXITING...")
            print("================================================================================================================")
            sys.exit(1)
    
    def get_client(self):
        return self.client
    
    def get_database(self):
        """Single database with collections Fault_History and Fault_Periods."""
        db_name = os.getenv("DB_NAME")
        if db_name not in self.client.list_database_names():
            print("================================================DATABASE NOT CREATED================================================")
            print(f"Database {db_name} not found. Please check if the database is created. EXITING...")
            print("================================================================================================================")
            sys.exit(1)
        return self.client[db_name]

    

    # Raw Data Commit to DB
    def insert_alert(self, alert: dict):
        document = dict(alert)
        document["timestamp"] = datetime.utcnow()
        document["_id"] = str(uuid.uuid4())
        collection_name = os.getenv("DB_RAW_DATA")
        try:
            self.get_database().get_collection(collection_name).insert_one(document)
            print("================================================RAW DATA SUCCESSFULLY INSERTED INTO DB================================================\n")
            print(f"Alert inserted successfully: {document}")
            print("================================================================================================================")
        except Exception as e:
            print("================================================ERROR INSERTING RAW DATA INTO DB================================================\n")
            print(f"Error inserting raw data into DB: {e}")
            print("================================================================================================================")
            raise HTTPException(
            status_code=500,
            detail=str(e)
            )


    def insert_fault_period_start(self, id: str, asset_id: str, fault_type: str, start_ts: float):
        """Insert a fault period when it starts (end_ts set when period ends)."""
        collection_name = os.getenv("DB_FAULT_PERIODS")
        document = {
            "_id": id,
            "asset_id": asset_id,
            "fault_type": fault_type,
            "start_ts": start_ts,
            "end_ts": None,
            "acknowledged": False,
            "acknowledged_at": None,
        }
        try:
            self.get_database().get_collection(collection_name).insert_one(document)
            print(f"Fault period started: {document}")
        except Exception as e:
            print(f"Error inserting fault period start: {e}")

    def update_fault_period_end(self, id: str, end_ts: float):
        """Update fault period with end_ts when fault ends."""
        collection_name = os.getenv("DB_FAULT_PERIODS")
        coll = self.get_database().get_collection(collection_name)
        try:
            result = coll.update_one({"_id": id}, {"$set": {"end_ts": end_ts}})
            if result.modified_count:
                print(f"Fault period ended: id={id}, end_ts={end_ts}")
            else:
                print(f"Fault period end update missed: id={id}")
        except Exception as e:
            print(f"Error updating fault period end: {e}")

    def acknowledge_fault_period(self, id: str, acknowledged_at: datetime | None = None) -> bool:
        """Mark a fault period as acknowledged."""
        acknowledged_at = acknowledged_at or datetime.utcnow()
        collection_name = os.getenv("DB_FAULT_PERIODS")
        coll = self.get_database().get_collection(collection_name)
        try:
            result = coll.update_one(
                {"_id": id},
                {"$set": {"acknowledged": True, "acknowledged_at": acknowledged_at}},
            )
            if result.matched_count == 0:
                print(f"Fault period acknowledge missed: id={id}")
                return False
            print(f"Fault period acknowledged: id={id}, acknowledged_at={acknowledged_at}")
            return True
        except Exception as e:
            print(f"Error acknowledging fault period: {e}")
            return False
