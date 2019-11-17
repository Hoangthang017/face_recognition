from flask import Flask, render_template, Response,url_for, jsonify, request, redirect,json
import cv2



app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
    