
from flask import Flask, render_template, request

import hr

app = Flask(__name__)

@app.route("/", methods = ['GET'])
def hello():
    return render_template('index.html')

@app.route("/dataTest", methods = ['POST'])
def dataTest():

    if request.method == "POST":
        data = hr.department_data
        emp_detail = hr.employee_details_data
        emp_data = hr.employee_data
        mod = hr.data_model
        score = hr.accuracy_sc
        classification_rep = hr.class_rep
        f1_sc = hr.fi_sc

    return render_template('index1.html', model = mod , score = score, clas = classification_rep, f1 = f1_sc)

@app.route("/sub", methods = ['POST'])
def submit():
    
    if request.method == "POST":
       name = request.form["name"]
    
    return render_template("sub.html", n = name)

if __name__==   "__main__":
    app.run(debug=True)