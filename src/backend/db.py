from pymongo.mongo_client import MongoClient

from src.backend.models.SessionLabelsModel import SessionLabelsModel


def get_database():
    connection_string = "mongodb+srv://admin_lse:5bKIwnZbTM7sGDjh@cluster0.6gyi8ct.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(connection_string)
    return client['latent_space_explorer']


def get_client():
    dbname = get_database()
    return dbname["labels"]


def save_session_labels_to_db(session_labels: SessionLabelsModel):
    try:
        data = {
            "z": session_labels.z,
            "anomalous_dims": session_labels.anomalous_dims,
            "shifts_range": session_labels.shifts_range,
            "shifts_count": session_labels.shifts_count,
        }
        get_client().insert_one(data)
        return True
    except Exception as e:
        print(e)
        return False
