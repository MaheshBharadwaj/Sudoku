from flask import *
from static.Solver import main as solve
from werkzeug.utils import secure_filename
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = ROOT_DIR + '/static/temp/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_folder='static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
board = None
solved_board = None


@app.route('/', methods=['GET'])
def upload_file():
	if request.method == 'GET':
		return render_template('index.html', invalid=0)


@app.route('/puzzle', methods=['GET', 'POST'])
def upload():
	global board
	global solved_board
	global ALLOWED_EXTENSIONS
	if request.method == 'POST':
		f = request.files['file']
		name, ext = f.filename.split('.')
		if ext not in ALLOWED_EXTENSIONS:
			return render_template('index.html', invalid_file = 1)
		f.save(os.path.join(
			app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

		 
		board, solved_board = solve(f.filename)
		if board is None: 
			return render_template('index.html', invalid_file = 1) 
		return render_template('solution.html', board=board, button=1)
	else:
		return render_template('index.html', invalid_file = 0)


@app.route('/check', methods=['GET', 'POST'])
def check():
	global board
	if request.method == 'POST':
		user_input = request.form

		for key in user_input.keys():
			i, j = tuple(map(int, key.split('-')))
			try:
				board[i][j] = int(user_input[key])
			except:
				board[i][j] = 0

		print(board)
		correct = 1
		for i in range(9):
			for j in range(9):
				if board[i][j] != solved_board[i][j]:
					correct = 0
					break
		print(correct)
		if correct:
			return render_template('success.html', board=solved_board)
		return render_template('failure.html', board=solved_board)

	if request.method == 'GET':
		return redirect(url_for('upload_file'))


@app.route('/solution', methods=['GET'])
def solution():
	if request.method == 'GET':
		return render_template('solution.html', board=solved_board, button=0)


if __name__ == '__main__':
	app.run(debug=True)
