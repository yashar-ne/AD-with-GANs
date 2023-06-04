from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def get_database():
    connection_string = "mongodb+srv://admin_lse:5bKIwnZbTM7sGDjh@cluster0.6gyi8ct.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(connection_string)
    return client['latent_space_explorer']


def get_client():
    dbname = get_database()
    return dbname["labels"]


def save_to_db(z, shifts_range, shifts_count, dim, is_anomaly):
    try:
        data = {
            "z": z,
            "shifts_range": shifts_range,
            "shifts_count": shifts_count,
            "dim": dim,
            "is_anomaly": is_anomaly
        }
        get_client().insert_one(data)
        return True
    except Exception as e:
        print(e)
        return False
