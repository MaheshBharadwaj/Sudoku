from flask import *
from werkzeug import secure_filename
import os

app = Flask(__name__)

@app.route('/')
def upload_file():
	return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload():
	if request.method == 'POST':
		f = request.files['file']
		f.save(secure_filename(f.filename))
		os.system("python3 Solver.py "+f.filename)
		return "solved"
		
if __name__ == '__main__':
   app.run(debug = True)