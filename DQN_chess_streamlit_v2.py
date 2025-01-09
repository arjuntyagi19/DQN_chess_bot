import streamlit as st
import chess
import numpy as np
from collections import deque
import joblib
import random
import time
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow import keras
import chess.svg
import base64
import pandas as pd



def render_svg(svg, placeholder, width=None, height=None):
    """Render SVG on Streamlit using a placeholder with optional width, height, and center alignment."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    # Define style for img tag with optional width and height
    img_style = "display:block; margin:auto;"  # This centers the image
    if width:
        img_style += f"width:{width};"
    if height:
        img_style += f"height:{height};"
    html = f'<img src="data:image/svg+xml;base64,{b64}" style="{img_style}"/>'
    placeholder.markdown(html, unsafe_allow_html=True)
    

# Importing from your custom modules
import board
from evaluator import evaluator_class

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Function to get scores of moves
def get_scores_of_moves(b, moves, e):
    # Scores are from white's perspective
    n = len(moves)
    scores = np.zeros(n)

    for i in range(n):
        board.apply_move(b, moves[i])
        scores[i] = e.eval(b)
        b.pop()

    return scores

# Function to get the move to play
def get_move_to_play(b, agent):
    e = agent.value
    moves = board.find_possible_moves(b)
    scores = get_scores_of_moves(b, moves, e)
    
    df = pd.DataFrame({"Move": moves, "Score": scores})
        
    if b.turn == chess.WHITE:
        best_move_idx = np.argmax(scores)
    else:
        best_move_idx = np.argmin(scores)
        
    return moves[best_move_idx] , df

# Class for the agent
class agent_class():
    def __init__(self):
        self.value = evaluator_class()
        self.memory = deque(maxlen=2000000)
        self.loss_history = deque(maxlen=1000)
        self.same_move_history = deque(maxlen=1000)

    def learn_from_memory(self, batch_size):
        if len(self.memory) <= 0:
            return         

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

# Load agent function
def load_agent(agent, i):
    data = joblib.load("./agents/agent_" + str(i))
    agent.memory = data[0]
    agent.loss_history = data[1]
    agent.same_move_history = data[2]

    tmp =  keras.models.load_model("./agents/value_" + str(i))
    weights = tmp.get_weights()
    agent.value.value.set_weights(weights)


st.set_page_config(
    page_title="DQN Chess using RL",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def main():
    st.title("DQN Chess using RL")
    ## wide mode
   
    
    
    
    # st.sidebar.title("Navigation")
    # page = st.sidebar.radio("Go to", ("Play Chess", "Find Mate in One"))

    # if page == "Play Chess":
    #     play_chess()
    # elif page == "Find Mate in One":
    #     find_mate_in_one()
        
        
        
    tab_titles = ["Home","Play Chess", "Find Mate in One","About"]
    tab1, tab2, tab3 ,tab4 = st.tabs(tab_titles)
    
    with tab1:
        st.title("Home")
        st.write("Welcome to the Chess AI app!")
        st.write("Select a tab from the sidebar to get started.")
        
    with tab2:
        play_chess()
        
        
    with tab3:
        find_mate_in_one()
        
    with tab4:
        st.title("About")
        st.write("This app uses a Deep Q-Network (DQN) to play chess.")
        st.write("The DQN is trained using self-play, where the agent plays against itself and learns from the outcomes of the games.")
        st.write("The agent uses a neural network to evaluate board positions and make decisions.")
        st.write("The app allows you to play chess against the AI or find mate in one puzzles.")
        st.write("Enjoy playing chess with the AI!")
    
    
    
 

def play_chess():
    # Initialize or load the chess board state
    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
    
    agent = agent_class()
    inp = np.zeros((1, 8, 8, 26))
    out = np.zeros(1)
    agent.value.value.fit(inp, out, verbose=0)
    load_agent(agent, 1450)  # Load agent from saved model

    
    board_placeholder = st.empty()
    render_svg(chess.svg.board(st.session_state.board), board_placeholder , width="400px", height="400px")

    if st.session_state.board.turn == chess.WHITE:
        move = st.text_input("Enter your move (in algebraic notation):", key="move_input")
        if st.button("Submit Move"):
            try:
                if chess.Move.from_uci(move) in st.session_state.board.legal_moves:
                    st.session_state.board.push_uci(move)
                    render_svg(chess.svg.board(st.session_state.board), board_placeholder , width="400px", height="400px")
                    time.sleep(1)  # Simulate AI thinking time
                    if not st.session_state.board.is_game_over():
                        agent_move , _ = get_move_to_play(st.session_state.board , agent)
                        st.session_state.board.push_san(agent_move)
                        render_svg(chess.svg.board(st.session_state.board), board_placeholder , width="400px", height="400px")
                else:
                    st.error("Invalid move! Try again.")
            except ValueError:
                st.error("Invalid move format! Please enter moves in algebraic notation.")
    else:
        time.sleep(1)  # Simulate AI thinking time
        agent_move , _ = get_move_to_play(st.session_state.board , agent)
        st.session_state.board.push(agent_move)
        render_svg(chess.svg.board(st.session_state.board), board_placeholder , width="400px", height="400px")

    if st.session_state.board.is_game_over():
        st.write("Game over!")
        st.write("Result:", st.session_state.board.result())

def find_mate_in_one():
    st.title("Mate in One Finder")
    fen_position = st.text_input("Enter FEN position:", value="r1b1k3/3pr3/4P2K/1ppP4/Ppn3p1/R4R1p/5Q2/1N2N2B w - - 0 44")
    b = chess.Board(fen_position)
    agent = agent_class()
    inp = np.zeros((1, 8, 8, 26))
    out = np.zeros(1)
    agent.value.value.fit(inp, out, verbose=0)
    load_agent(agent, 3250)  # Load agent from saved model

    def find_mate_in_one_position(agent, fen_position):
        b = chess.Board(fen_position)
        agent_move , df = get_move_to_play(b, agent)
        print("Agent move:", agent_move)
        
        return agent_move , df

    

    mate_in_one_move , df = find_mate_in_one_position(agent, fen_position)
    st.write("Mate in one move:", mate_in_one_move)
    st.write("Board position:")
    
    
    def make_arrows(b, move):
        arrows = []
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color="#0000cccc"))
        return arrows
    
    
    b.push_san(mate_in_one_move)
    
    move = b.pop()
    
    
    board = chess.svg.board(b , arrows=make_arrows(b, move) , size=400)
    
    render_svg(board, st.empty() ,  width="400px", height="400px")
    
    st.dataframe(df.sort_values(by=['Score']))


   

if __name__ == "__main__":
    main()
