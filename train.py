import numpy as np

from collections import deque
import joblib
import random
import time

import chess

import board
import evaluator
from evaluator import evaluator_class


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow import keras

tf.compat.v1.disable_eager_execution() # otherwise it is slow !!!



def get_scores_of_moves(b, moves, e):
    # scores are from white's perspective
    n = len(moves)
    scores = np.zeros(n)

    for i in range(n):
        board.apply_move(b, moves[i])
        scores[i] = e.eval(b)
        b.pop()

    return scores    




class agent_class():

    def __init__(self):

        self.value = evaluator_class()
        
        self.memory = deque(maxlen= 2000000)
        self.loss_history = deque(maxlen= 1000)
        self.same_move_history = deque(maxlen= 1000)

        
    def learn_from_memory(self, batch_size):
        if len(self.memory) <= 0 : return         

        inp = np.zeros((batch_size, 8, 8, 26))

        out = np.zeros(batch_size)
        p = 0

        same_moves = 0
        same_moves_counter = 0
        
        for i in range(batch_size):
            index = np.random.choice(len(self.memory))
            fen, prev_move = self.memory[index]
            b = chess.Board(fen)

            inp[p] = board.board_to_features(b)
            
            terminal, score =  board.is_terminal(b)
        
            if terminal:
                out[p] = score
            else:
                moves  = board.find_possible_moves(b)
                scores = get_scores_of_moves(b, moves, self.value)            
                out[p] = np.max(scores)
                current_move = np.argmax(scores)

                if prev_move != -1:
                    same_moves_counter += 1
                    if prev_move == current_move:
                        same_moves += 1
                self.memory[index] = (fen, current_move)
                
            p += 1

        assert p == batch_size  

        print("mean= ", np.mean(out))

        history = self.value.value.fit(inp,out, verbose=0)

        loss = history.history["loss"][0] * 1000
        self.loss_history.append(loss)

        if same_moves_counter > 20:
            self.same_move_history.append(same_moves / same_moves_counter)


    def report_loss(self):
        n = len(self.loss_history)
        if n == 0 : return 0
        x = np.zeros(n)
        for i in range(n):
            x[i] = self.loss_history[i]
        return np.mean(x)
           
    def report_same_move(self):
        n = len(self.same_move_history)
        if n == 0 : return 0
        x = np.zeros(n)
        for i in range(n):
            x[i] = self.same_move_history[i]
        return np.mean(x)



def get_move_to_play(b, agent, i, explore):
    e = agent.value
    
    moves = board.find_possible_moves(b)

    if explore:
        return moves[np.random.choice(len(moves))]
    
    scores = get_scores_of_moves(b, moves, e)

    if i < 10: scores += np.random.normal(0, 0.04, len(moves))
        
    if b.turn == chess.WHITE:
        best_move = np.argmax(scores)
    else:
        best_move = np.argmin(scores)
        
    return moves[best_move]

            
def play_game(agent):
    # returns the game and whether it was decided
    game = []
    explores = []
    b = chess.Board()
    game_explore = np.random.rand() < 0.5

    past_positions = {""}
    reps_num = 0

    for i in range(150):

        explore = np.random.rand() < 0.1 and game_explore
        to_play = get_move_to_play(b, agent, i, explore)

        game.append(to_play)
        explores.append(explore)

        board.apply_move(b, to_play)

        f = board.board_to_fen(b)
        if f in past_positions: reps_num += 1
        if reps_num >= 1: break
        past_positions.add(f)

        terminal, score = board.is_terminal(b)
        
        if terminal: return game, True, explores

    return game, False, explores



def nice_print(game, explores):
    for i in range(len(game)):
        print(game[i] , end=" ")
        if explores[i]: print("{e}", end=" ")
    print("\n")

    
def get_decidable_game(agent):
    total = 0
    while True:
        game, decided, explores = play_game(agent)
        nice_print(game, explores)
        total = len(game)
        if decided:
            return game, total


def insert_game_to_memory(game, agent):
    b = chess.Board()
    for m in game[:-1]: # don't insert last position, which is terminal
        board.apply_move(b, m)

        if b.turn == chess.WHITE:
            fen = board.board_to_fen(b)
        else:
            c = b.mirror()
            fen = board.board_to_fen(c)
        agent.memory.append((fen, -1))


    
def save_model(agent, train_iteration):
    if (train_iteration % 1000) != 0:
        return

    data = (agent.memory, agent.loss_history, agent.same_move_history)
    
    joblib.dump(data,      ".\\agents\\agent_" + str(train_iteration // 1000))
    agent.value.value.save(".\\agents\\value_" + str(train_iteration // 1000))



def load_agent(agent, i):
    data = joblib.load(".\\agents\\agent_" + str(i))
    agent.memory = data[0]
    agent.loss_history = data[1]
    agent.same_move_history = data[2]

    tmp =  keras.models.load_model(".\\agents\\value_" + str(i))
    weights = tmp.get_weights()
    agent.value.value.set_weights(weights)
    

test_positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq",
"r1bqk2r/p1p2ppp/2p2n2/4p1N1/1bB5/8/PPPP1PPP/RNBQK2R w KQkq",
"r2qr1k1/1bp2ppp/p1p5/3np1NQ/1PB5/2N5/PP1P1PPP/R1B2RK1 w -" ]

def eval_test_positions():
    print("  pos= ", end="")
    for x in test_positions:
        b = chess.Board(x)
        v = agent.value.eval(b)
        print("%6.4f " % v , end="")
    print("\n")


    
def consider_stopping():
    while True:
        lines = open("stop.txt","r").read()
        if not "stop" in lines: return
        print("sleeping")
        time.sleep(30)
    

agent = agent_class()

inp = np.zeros((1,8,8,26))
out = np.zeros(1)
agent.value.value.fit(inp, out, verbose = 0)


#load_agent(agent, 222) # to continue from a saved model


for i in range(1, 5):    
    game, total = get_decidable_game(agent)

    insert_game_to_memory(game, agent)

    agent.learn_from_memory(256)

    loss = agent.report_loss()
    same = agent.report_same_move()

    print ("i= ", i , "loss= ", "%7.3f" % loss,
           "same= ", "%7.3f " % same, end= "")
    eval_test_positions()
        
    save_model(agent, i + 1)
    consider_stopping()





