"""
Fichero que reune el codigo necesario para montar la API REST donde se despliegan los modelos en produccion.
"""

## IMPORTS -----
import connexion
from flask import render_template


## Creo la App
app = connexion.App(__name__, specification_dir="./")
app.add_api("openapi.yaml")

## ROUTES -----
@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)