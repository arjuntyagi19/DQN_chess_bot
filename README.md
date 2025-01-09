# Chess Bot using Deep Q-Learning (DQN)
## 🌐 Hosted Application
The Chess Bot is deployed using **Streamlit**. Try it here: [Streamlit App](https://dqn-chess-app.streamlit.app/)

This repository contains the implementation and detailed report of a Chess Bot leveraging Deep Q-Learning (DQN). The bot is designed to facilitate interactive gameplay and solve **mate-in-one** challenges, utilizing advanced reinforcement learning techniques.

---

## 📖 Project Overview

Chess is a game of strategy and complexity, making it an ideal domain for reinforcement learning. This project builds a **Chess Bot** capable of adaptive decision-making using a **Deep Q-Network (DQN)**. It is trained to evaluate board states and select optimal moves dynamically.


## 🧠 Model Architecture

The bot is trained using a **Deep Q-Network** that:
- Maps chessboard states (represented as FEN) to Q-values for potential moves.
- Uses a reward system based on the quality of moves (e.g., checkmates, material gain).
- Optimizes policy through experience replay and epsilon-greedy exploration.

---

## 📊 Results

- **Mate-in-One Accuracy:** 95%
- **General Gameplay Performance:** The bot demonstrates competitive play against beginners.
- **Training Performance:** Reward converges after 10,000 episodes.

---


---

## 🚀 Features
- **Reinforcement Learning:** Implementation of DQN to train the bot.
- **Interactive Gameplay:** Users can play against the bot or test it on specific **mate-in-one** puzzles.
- **FEN Input Support:** Accepts Forsyth-Edwards Notation (FEN) to load chessboard positions.
- **Deployment:** Hosted using **Streamlit** for seamless accessibility.

---

## 🛠️ Tools and Technologies
- **Python**
- **PyTorch/TensorFlow** for DQN implementation
- **Chess Libraries:** `python-chess` for board representation and move validation
- **Streamlit** for building the interactive web app
- **Google Colab/Jupyter** for training and experimentation

---

## 📂 Project Structure

 
