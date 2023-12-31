import random
import copy
import time

class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        drop_phase = True  # TODO: detect drop phase
        pin = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                if(state[i][j] != ' '):
                    pin+= 1
        if(pin >= 8):
            drop_phase = False
        
        
        move = []
        index = 0
        successor_list = self.succ(state)
        value = self.eval_succ(successor_list)

        # randomize starting pin for variety
        if pin < 2:
            successor = successor_list[random.randint(0, len(successor_list)-1)]
            for i in range(len(state)):
                for j in range(len(state[i])):
                    if(state[i][j] == ' ' and successor[i][j] == self.my_piece):
                        # ensure the destination (row,col) tuple is at the beginning of the move list
                        move.insert(0, (i, j))
                        return move

        high = value[0]
        for i in range(len(value)):
            if(high < value[i]):
                high = value[i]
                index = i
        successor = successor_list[index]
        if not drop_phase:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    if(state[i][j] == self.my_piece and successor[i][j] == ' '):
                        move.append((i, j))

        for i in range(len(state)):
                for j in range(len(state[i])):
                    if(state[i][j] == ' ' and successor[i][j] == self.my_piece):
                        # ensure the destination (row,col) tuple is at the beginning of the move list
                        move.insert(0, (i, j))
        # TODO: implement a minimax algorithm to play better

        return move

    def succ(self, state):
        drop_phase = True  
        pin = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                if(state[i][j] == self.my_piece):
                    pin+= 1
        if(pin >= 4):
            drop_phase = False
        successor = []
        temp = copy.deepcopy(state)
        if(drop_phase == True):
            for i in range(len(temp)):
                for j in range(len(temp[i])):
                    if(temp[i][j] == ' '):
                        temp[i][j] = self.my_piece
                        successor.append(copy.deepcopy(temp))
                        temp[i][j] = ' ' 
        else:
            for i in range(len(temp)):
                for j in range(len(temp[i])):
                    if(temp[i][j] == self.my_piece):
                        if(i != 0 and temp[i-1][j] == ' '): #up
                            temp[i-1][j] = self.my_piece
                            temp[i][j] = ' '
                            successor.append(copy.deepcopy(temp))
                            temp[i-1][j] = ' '
                            temp[i][j] = self.my_piece  
                        if(i != 4 and temp[i+1][j] == ' '): #down
                            temp[i+1][j] = self.my_piece
                            temp[i][j] = ' '
                            successor.append(copy.deepcopy(temp))
                            temp[i+1][j] = ' '
                            temp[i][j] = self.my_piece    
                        if(j != 0 and temp[i][j-1] == ' '): #left
                            temp[i][j-1] = self.my_piece
                            temp[i][j] = ' '
                            successor.append(copy.deepcopy(temp))
                            temp[i][j-1] = ' '
                            temp[i][j] = self.my_piece
                        if(j != 4 and temp[i][j+1] == ' '): #right
                            temp[i][j+1] = self.my_piece
                            temp[i][j] = ' '
                            successor.append(copy.deepcopy(temp))
                            temp[i][j+1] = ' '
                            temp[i][j] = self.my_piece
                        if(i != 0 and j != 4 and temp[i-1][j+1] == ' '): #up right 
                            temp[i-1][j+1] = self.my_piece
                            temp[i][j] = ' '
                            successor.append(copy.deepcopy(temp))
                            temp[i-1][j+1] = ' '
                            temp[i][j] = self.my_piece
                        if(i != 0 and j != 0 and temp[i-1][j-1] == ' '): #up left
                            temp[i-1][j-1] = self.my_piece
                            temp[i][j] = ' '
                            successor.append(copy.deepcopy(temp))
                            temp[i-1][j-1] = ' '
                            temp[i][j] = self.my_piece
                        if(i != 4 and j != 4 and temp[i+1][j+1] == ' '): #down right 
                            temp[i+1][j+1] = self.my_piece
                            temp[i][j] = ' '
                            successor.append(copy.deepcopy(temp))
                            temp[i+1][j+1] = ' '
                            temp[i][j] = self.my_piece
                        if(i != 4 and j != 0 and temp[i+1][j-1] == ' '): #down left
                            temp[i+1][j-1] = self.my_piece
                            temp[i][j] = ' '
                            successor.append(copy.deepcopy(temp))
                            temp[i+1][j-1] = ' '
                            temp[i][j] = self.my_piece
        return successor

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    """ Function to evaluate successor """
    def eval_succ(self, successor):
        result = []
        for i in range(len(successor)):
            if(self.game_value(successor[i]) != 0):
                result.append(self.game_value(successor[i]))
            else:
                result.append(self.heuristic_game_value(successor[i]))
        return result

    def heuristic_game_value(self, state):
        score = 0
        update = False
        update_opp = False
        # Horizontal line 3 in a row
        for row in range(5):
            for col in range(3):
                if state[row][col] != ' ' and state[row][col] == state[row][col+1] == state[row][col+2]:
                    if state[row][col] == self.my_piece and update == False:
                        update = True
                        score += 100
                    elif state[row][col] != self.my_piece and state[row][col] != ' ' and update_opp == False:
                        update_opp = True
                        score -= 150
        
        #vertical line 3 in a row
        for col in range(5):
            for row in range(3):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col] == state[row+2][col]:
                    if state[row][col] == self.my_piece and update == False:
                        update = True
                        score += 100
                    elif state[row][col] != self.my_piece and state[row][col] != ' ' and update_opp == False:
                        update_opp = True
                        score -= 150
        
        # Horizontal line 2 in a row
        for row in range(5):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row][col+1]:
                    if state[row][col] == self.my_piece and update == False:
                        update = True
                        score += 50
                    elif state[row][col] != self.my_piece and state[row][col] != ' ' and update_opp == False:
                        update_opp = True
                        score -= 25

        #vertical line 2 in a row
        for col in range(5):
            for row in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col]:
                    if state[row][col] == self.my_piece and update == False:
                        update = True
                        score += 50
                    elif state[row][col] != self.my_piece and state[row][col] != ' ' and update_opp == False:
                        update_opp = True
                        score -= 25

        #diagonal \ line 3 in a row
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2]:
                    if state[row][col] == self.my_piece and update == False:
                        update = True
                        score += 100
                    elif state[row][col] != self.my_piece and state[row][col] != ' ' and update_opp == False:
                        update_opp = True
                        score -= 150

        #diagonal \ line 2 in a row
        for row in range(3):
            for col in range(3):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1]:
                    if state[row][col] == self.my_piece and update == False:
                        update = True
                        score += 50
                    elif state[row][col] != self.my_piece and state[row][col] != ' ' and update_opp == False:
                        update_opp = True
                        score -= 25
        
        #diagonal / line 3 in a row
        x = len(state)-1
        for row in range(2):
            for col in range(2):
                if state[row][x-col] != ' ' and state[row][x-col] == state[row+1][x-col-1] == state[row+2][x-col-2]:
                    if state[row][x-col] == self.my_piece and update == False:
                        update = True
                        score += 100
                    elif state[row][x-col] != self.my_piece and state[row][x-col] != ' ' and update_opp == False:
                        update_opp = True
                        score -= 150

        #diagonal / line 2 in a row
        x = len(state)-1
        for row in range(3):
            for col in range(3):
                if state[row][x-col] != ' ' and state[row][x-col] == state[row+1][x-col-1]:
                    if state[row][x-col] == self.my_piece and update == False:
                        update = True
                        score += 50
                    elif state[row][x-col] != self.my_piece and state[row][x-col] != ' ' and update_opp == False:
                        update_opp = True
                        score -= 25
        return (score/200)

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and 3x3 square corners wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == state[row+3][col+3]:
                    return 1 if state[row][col] == self.my_piece else -1 
        # TODO: check / diagonal wins
        x = len(state)-1
        for row in range(2):
            for col in range(2):
                if state[row][x-col] != ' ' and state[row][x-col] == state[row+1][x-col-1] == state[row+2][x-col-2] == state[row+3][x-col-3]:
                    return 1 if state[row][x-col] == self.my_piece else -1
        # TODO: check 3x3 square corners wins
        for row in range(3):
            for col in range(3):
                if state[row][col] != ' ' and state[row][col] == state[row+2][col] == state[row][col+2] == state[row+2][col+2] and state[row+1][col+1] == ' ':
                    return 1 if state[row][col] == self.my_piece else -1
        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)
                

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
