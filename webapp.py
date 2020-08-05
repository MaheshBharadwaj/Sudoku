from flask import *
from static.Solver import main as solve
from werkzeug.utils import secure_filename
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = ROOT_DIR + '/static/temp/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_folder='static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_file():
	return render_template('index.html')
	
@app.route('/uploader', methods = ['POST'])
def upload():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

		print('Filename: ', f.filename,'#')
		solve(f.filename)
		return "solved"
		
if __name__ == '__main__':
	app.run(debug = True)