# coding: utf-8

from flask import Flask
from flask import render_template
from flask_sockets import Sockets

import sqlite3
import os
import json
from flask import request
from flask import g

import leancloud

leancloud.init("El4C8thRibJnmYN7eke4iSIY-gzGzoHsz", "15erFcxwUmwrbe5uPhDTyLag")

app = Flask(__name__, static_url_path='/static')
sockets = Sockets(app)

# 动态路由

survive_mapping = ['Perished','Survived']
pclass_mapping = ["","Class1","Class2","Class3"]
gender_mapping = {"female":"Female","male":"Male"}
path = os.getcwd() + "/static/titanic_data_db.db"
#path = "titanic_data_db.db"

def age_mapping(age_value):       
        if age_value > 0 and age_value <= 16:
            age_group = "0-16"
        elif age_value > 16 and age_value <= 25:
            age_group = "16-25"
        elif age_value >25 and age_value <= 40:
            age_group = "25-40"
        elif age_value >40 and age_value <= 60:
            age_group = "40-60"
        else:
            age_group = "60 and overs"
        return age_group

#@app.before_request
#def before_request():
#    g.db = sqlite3.connect(path)
#
#
#@app.teardown_request
#def teardown_request(exception):
#    if hasattr(g, 'db'):
#        g.db.close()

@app.route('/')
def titanic_data():
    return render_template('story_board.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/survive')
def sql(**params):
    pclass = request.args.get('pclass')
    sex = request.args.get('sex')
    min_age = request.args.get('min_age')
    max_age = request.args.get('max_age')
    g.db = sqlite3.connect(path)
    cursor = g.db.execute("select survived,count(*) from titanic_data_edit\
    where Pclass = {}\
    and Sex ='{}'\
    and Age >{}\
    and Age <{}\
    group by survived".format(pclass,sex,min_age,max_age))
    result = []
    for survive, count in cursor:
        result.append({"Survive": 'Survived' if survive == 1 else 'Perished',"Count":count})
    return json.dumps(result)


@app.route('/pclass')
def pclass_sql(**params):
    sql = "select survived, pclass from titanic_data_edit"
    g.db = sqlite3.connect(path)
    cursor = g.db.execute(sql)
    result = []
    for survive_value, pclass_value in cursor:
        result.append({"Survived":survive_mapping[survive_value],
        "Pclass":pclass_mapping[pclass_value],
        "count":1})
    return json.dumps(result)

@app.route('/sex')
def sex_sql(**params):
    sql = "select survived, sex from titanic_data_edit;"
    g.db = sqlite3.connect(path)
    cursor = g.db.execute(sql)
    result = []
    for survive_value, sex_value in cursor:
        result.append({"Survived":survive_mapping[survive_value],
        "Gender":gender_mapping[sex_value],
        "count":1})
    return json.dumps(result)

@app.route('/age')
def age_sql(**params):
    sql = "select survived, age from titanic_data_edit"
    survive_mapping = ['Perished','Survived']
    g.db = sqlite3.connect(path)
    cursor = g.db.execute(sql)
    result = []
    for survive_value,age_value in cursor:
        if age_value != "":
            result.append({"Survived":survive_mapping[survive_value],
            "Age Group":age_mapping(age_value),
            "count":1})
    return json.dumps(result)

@app.route('/comprehensive')
def comprehensive_sql(**params):
    sql = "select survived,sex,pclass from titanic_data_edit;"
    g.db = sqlite3.connect(path)
    cursor = g.db.execute(sql)
    result = []
    for survive_value,sex_value,pclass_value in cursor:
        result.append({"Survived":survive_mapping[survive_value],
        "Gender":gender_mapping[sex_value],
        "Pclass":pclass_mapping[pclass_value],
        "count":1})
    return json.dumps(result)


@sockets.route('/echo')
def echo_socket(ws):
    while True:
        message = ws.receive()
        ws.send(message)
