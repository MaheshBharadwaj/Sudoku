from flask import *
app = Flask(__name__)

@app.route('/')
def home():
	board = [[9,7,0,0,0,2,0,0,0],[1,0,8,0,0,6,0,0,4],[0,0,2,0,0,0,7,0,0],
			[0,0,1,8,5,0,4,0,0],[5,4,7,0,2,1,0,0,9],[8,0,6,7,9,0,1,0,5],
			[3,0,0,5,0,8,0,0,0],[0,0,0,0,0,0,5,4,0],[7,2,0,0,4,0,0,8,0]]
	session['board'] = board
	return redirect(url_for('print_board'))

@app.route('/print')
def print_board():
	return render_template('login.html', board = session.get('board'))

if __name__ == '__main__':
	app.secret_key = 'some key'
	app.run(debug = True)