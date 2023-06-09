import json
import requests
import numpy as np
from flask import Flask, request, make_response
from flask_cors import CORS

import QuoraServer as qs

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


@app.route('/search')
def search():
    query = request.args.get("query")
    page = request.args.get("page")
    response = requests.get("http://127.0.0.1:8080/query-processing?query=" + query)

    results, total = qs.run_query_with_index(response.json()['result'], False, page)
    # results, total = qs.run_query_with_cluster(response.json()['result'], False, page)

    response = make_response(json.dumps({"result": results, "total_count": total}, cls=NumpyEncoder))
    response.headers.set('Content-Type', 'application/json')
    return response


@app.route('/suggest')
def suggest():
    query = request.args.get("query")

    response = requests.get("http://127.0.0.1:8080/query-processing?query=" + query)
    results = qs.get_suggestions(response.json()['result'])

    response = make_response(json.dumps({"result": results}, cls=NumpyEncoder))
    response.headers.set('Content-Type', 'application/json')
    return response


@app.route('/query-processing')
def query_processing():
    query = request.args.get("query")
    results = qs.query_processing(query)

    response = make_response(json.dumps({"result": results}, cls=NumpyEncoder))
    response.headers.set('Content-Type', 'application/json')
    return response


if __name__ == "__main__":
    app.run(port=8080, debug=True)
