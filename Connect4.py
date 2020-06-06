from time import *

dp = dict()

def Conv(l):
	# Returns the given list as a tuple.
	t = tuple()
	for i in range(len(l)):
		t += tuple(l[i])
	return t

def PrintBoard(l):
	#Prints the Board.
	for i in range(len(l)):
		print(*l[i])
	print()

def Win(l, x, y):
	st = l[x][y]
	if(st == '.'): return False
	for i in range(x,x+4):
		if(i >= len(l) or l[i][y] != st):
			break
	else:
		return True

	for i in range(y,y+4):
		if(i >= len(l[1]) or l[x][i] != st):
			break
	else:
		return True

	for i in range(4):
		dx = x - i
		dy = y - i
		if(dx < 0  or dy < 0 or l[dx][dy] != st):
			break
	else:
		return True

	for i in range(4):
		dx = x - i
		dy = y + i
		if(dx < 0  or dy >= len(l[1]) or l[dx][dy] != st):
			break
	else:
		return True

	return False

def check(l):
	#Checks if there is a winner.
	ch = False
	for i in range(len(l)):
		for j in range(len(l[i])):
			ch |= Win(l, i, j)
	return ch

def mem(l, no, coins):
	t = Conv(l)
	if(check(l) == True):
		dp[t] = no%2
		return dp[t]
	if(no == 42): return 0
	if(t in dp): return dp[t]
	ans = 0
	for i in range(len(coins)):
		if(coins[i] >= 0):
			l[coins[i]][i] = str(no%2)
			coins[i] -= 1
			ans = max(ans,mem(l, no+1, coins))
			coins[i] += 1
			l[coins[i]][i] = '.'
	dp[t] = ans
	return dp[t]

l = [['.' for i in range(7)] for j in range(6)] #The Board.
coins = [5 for i in range(7)] #Which row of each column the coin next coin will go to.
#PrintBoard(l)
#print(mem(l, 0 , coins))

no = 0
while(check(l) == False):
	PrintBoard(l)
	r = int(input("Player %d's turn : "%(no%2+1)))
	r -= 1
	l[coins[r]][r] = str(no%2)
	coins[r] -= 1
	no += 1
PrintBoard(l)
if(no%2 == 0):
	print("Player 2 WINS!!")
else:
	print("Player 1 WINS!!")