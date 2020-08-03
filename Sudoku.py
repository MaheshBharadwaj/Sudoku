from copy import *
from time import *

class Sudoku():
    def __init__(self, board):
        self.board = board
        self.rows = [0 for i in range(9)]
        self.cols = [0 for i in range(9)]
        self.boxes = [0 for i in range(9)]
        self.n = len(board)
        self.solvable = True
        for i in range(self.n):
            for j in range(self.n):
                val = self.board[i][j]
                if(val != 0):
                    msk = (1<<(val-1))
                    self.rows[i] ^= msk
                    self.cols[j] ^= msk
                    self.boxes[(i//3)*3+(j//3)] ^= msk
                    if(self.rows[i] or self.cols[j] or self.boxes[(i//3)*3+(j//3)]): self.solvable = False

    def Solvable(self):
        return self.solvable

    def Print_board(self):
        for i in range(self.n):
            for j in range(self.n):
                print(self.board[i][j], end = " ")
                if(j%3 == 2 and j != 8): print("|", end = " ")
            print()
            if(i%3 == 2 and i != 8): print(("-"*6)+"+"+("-"*7)+"+"+("-"*7))

    def options(self, i, j):
        msk = self.rows[i]|self.cols[j]|self.boxes[(i//3)*3+(j//3)]
        left = []
        for i in range(9):
            if (msk&(1<<i)) == 0: left.append(i+1)
        return left

    def place(self, i, j, val):
        self.board[i][j] = val
        msk = (1<<(val-1))
        self.rows[i] ^= msk
        self.cols[j] ^= msk
        self.boxes[(i//3)*3+(j//3)] ^= msk

def solve(puzzle):
    solved = False
    while not solved:
        solved = True
        changed = False
        for i in range(puzzle.n):
            for j in range(puzzle.n):
                if(puzzle.board[i][j] == 0):
                    solved = False
                    left = puzzle.options(i,j)
                    if(len(left) == 0): return False
                    if(len(left) == 1):
                        puzzle.place(i, j, left[0])
                        changed = True
        if not changed:
            if solved:
                puzzle.Print_board()
                return True
            min_needed = ((puzzle.n+1),[],(-1,-1))
            for i in range(puzzle.n):
                for j in range(puzzle.n):
                    if(puzzle.board[i][j] == 0):
                        left = puzzle.options(i,j)
                        if(len(left) > 1): min_needed = min(min_needed, (len(left),left,(i,j)))
            i, j = min_needed[2]
            for val in min_needed[1]:
                puzzle_next = deepcopy(puzzle)
                puzzle_next.place(i, j, val)
                solved = solve(puzzle_next)
                if solved: break
            return solved
    return solved

def make_board(string):
    board = []
    for i in range(81):
        val = string[i]
        if(val == '.'): val = '0'
        if(i%9 == 0):
            l = []
        l.append(ord(val)-ord('0'))
        if(i%9 == 8): board.append(l)
    return board

def get_input():
    string = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'

    board = [[9,7,0,0,0,2,0,0,0],[1,0,8,0,0,6,0,0,4],[0,0,2,0,0,0,7,0,0],
        [0,0,1,8,5,0,4,0,0],[5,4,7,0,2,1,0,0,9],[8,0,6,7,9,0,1,0,5],
        [3,0,0,5,0,8,0,0,0],[0,0,0,0,0,0,5,4,0],[7,2,0,0,4,0,0,8,0]]
    #make_board(string)
    return board

def main():
    board = get_input()
    puzzle = Sudoku(board)
    puzzle.Print_board()
    print()
    st = time()
    solve(puzzle)
    en = time()
    print()
    print(en-st,"seconds")

if __name__ == '__main__':
    main()
