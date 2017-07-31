#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 06:04:22 2017

@author: newchama
"""

from flask import Flask
from flask import abort
from flask import redirect

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Hello World!</h1>'

@app.route('/user/<name>')
def sayHello(name):
    if name == 'baidu':
        return redirect('http://www.baidu.com')
    elif name == 'NO':
        return abort(404)

    return '<h1> Hello,%s </h1>' % name

if __name__ == '__main__':
    app.run(debug=True)
