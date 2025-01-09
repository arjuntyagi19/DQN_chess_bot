import chess
import numpy as np

def board_to_fen(b):
    # returns a shortened FEN without counters
    f = b.fen()
    f = f.split(" ")
    return " ".join(f[:4])

def fen_to_board(f):
    return chess.Board(f)

def is_terminal(b):
    # returns the result from White's perspective
    outcome = b.outcome()
    if outcome == None:
        return False, 0
    res = outcome.result()

    if res == "1-0": return True, 1
    if res == "0-1": return True, -1
    return True, 0

def find_possible_moves(b):
    moves = b.legal_moves
    return [b.san(m) for m in moves]

def apply_move(b , move):
    b.push_san(move)


def piece_to_int(piece):
    types = "rnbqkpRNBQKP"
    s = piece.symbol()
    return types.index(s)


def board_to_BaseBoard(b):
    f = b.fen()
    f = f.split(" ")[0]
    return chess.BaseBoard(f)


def get_attackers(b):
    res = np.zeros(12 * 64)
    bb = board_to_BaseBoard(b)
     
    d = bb.piece_map()

    for k in d.keys():
        row = 7 - k // 8
        col = k % 8
        p = piece_to_int(d[k])
        
        attacked = bb.attacks(k)

        for i in attacked:
            row = 7 - i//8
            col = i % 8
            res[p * 64 + row * 8 + col] = 1

    return res



def get_pawns(b):
    res = np.zeros(2 * 64)
    bb = board_to_BaseBoard(b)
     
    d = bb.piece_map()

    for k in d.keys():
        row = 7 - k // 8
        col = k % 8
        p = piece_to_int(d[k])

        if p == 5: # black pawn
            res[64 + (row+1) * 8 + col] = 1

        if p == 11: # white pawn
            res[(row-1) * 8 + col] = 1

    return res



def board_to_features(b):
    assert b.turn == chess.WHITE # evaluate only white
    
    res = np.zeros(64 * 12 * 2)
    
    d = b.piece_map()

    for k in d.keys():
        row = 7 - k // 8
        col = k % 8
        p = piece_to_int(d[k])        
        res[p * 64 + row * 8 + col] = 1

    res[64*12:] = get_attackers(b)

    pawns = get_pawns(b)
    res = np.concatenate((res, pawns))

    res2 = np.zeros((8,8,24 + 2))

    for i in range(8):
        for j in range(8):
            res2[i][j] = res[i*8 + j : : 64]
    
    return res2


