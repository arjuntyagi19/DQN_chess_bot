The tensor is structured as follows:

First 12 Indices (0-11): Direct presence of pieces on the board.
Next 12 Indices (12-23): Squares attacked by each type of piece.
Last 2 Indices (24-25): Special pawn structures (pawn attacks).
Here’s a detailed description of each feature map index:

Direct Presence of Pieces on the Board
Index 0: Black Rooks (r)
Index 1: Black Knights (n)
Index 2: Black Bishops (b)
Index 3: Black Queens (q)
Index 4: Black Kings (k)
Index 5: Black Pawns (p)
Index 6: White Rooks (R)
Index 7: White Knights (N)
Index 8: White Bishops (B)
Index 9: White Queens (Q)
Index 10: White Kings (K)
Index 11: White Pawns (P)
Squares Attacked by Each Type of Piece
Index 12: Squares attacked by Black Rooks
Index 13: Squares attacked by Black Knights
Index 14: Squares attacked by Black Bishops
Index 15: Squares attacked by Black Queens
Index 16: Squares attacked by Black Kings
Index 17: Squares attacked by Black Pawns
Index 18: Squares attacked by White Rooks
Index 19: Squares attacked by White Knights
Index 20: Squares attacked by White Bishops
Index 21: Squares attacked by White Queens
Index 22: Squares attacked by White Kings
Index 23: Squares attacked by White Pawns
Special Pawn Structures
Index 24: Black Pawn Structures (typically these could represent black pawn attacks or configurations in advanced analyses)
Index 25: White Pawn Structures (similarly for white pawns)
