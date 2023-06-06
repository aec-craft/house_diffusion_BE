from mongoengine import connect


def create_db():
    db_str = "mongodb://localhost:27017"
    return connect(db="fyp", host=db_str)
