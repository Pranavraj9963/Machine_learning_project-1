import mysql.connector

connection = mysql.connector.connect(
    host ="localhost",
    user ="root",
    password ="Sairaj@123",
    database ="spam_detection"
)

cursor = connection.cursor()

def save_prediction(
        email,
        model,
        prediction,
        probability
):
    sql = """
        INSERT INTO prediction_history(
        email,
        model_name,
        prediction,
        probability
        )
        VALUES(
        %s,
        %s,
        %s,
        %s
        )"""
    

    values = (email,
                model,
                prediction,
                probability
                )
    
    cursor.execute(sql,values)

    connection.commit()