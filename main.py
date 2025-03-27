import sys
import chess
import chess.svg
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox)
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QTimer
from NigelBotData.chess_ai import NigelChessBot, train_set
class ChessGUI(QMainWindow):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.board = chess.Board()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('NigelChessBot')
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for board display
        self.board_panel = QWidget()
        board_layout = QVBoxLayout(self.board_panel)
        
        self.svg_widget = QSvgWidget()
        self.update_board_display()
        board_layout.addWidget(self.svg_widget)
        
        # Right panel for controls and info
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Game controls
        self.reset_button = QPushButton("New Game")
        self.reset_button.clicked.connect(self.reset_game)
        
        self.undo_button = QPushButton("Undo Move")
        self.undo_button.clicked.connect(self.undo_move)
        
        self.bot_move_button = QPushButton("Make Bot Move")
        self.bot_move_button.clicked.connect(self.make_bot_move)
        
        self.auto_play_button = QPushButton("Auto Play")
        self.auto_play_button.clicked.connect(self.toggle_auto_play)
        
        
        # Status display
        self.status_label = QLabel()
        self.eval_label = QLabel()  # Moved this line up before update_status()
        self.eval_label.setWordWrap(True)
        self.move_counter = 0
        self.move_counter_label = QLabel()
        self.move_counter_label.setText(f"Move Counter: {self.move_counter}")
        self.move_counter_label.setWordWrap(True)
        self.move_counter_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to control layout
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.undo_button)
        control_layout.addWidget(self.bot_move_button)
        control_layout.addWidget(self.auto_play_button)
        control_layout.addStretch()
        control_layout.addWidget(self.move_counter_label)
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.eval_label)
        
        # Add panels to main layout
        main_layout.addWidget(self.board_panel, 3)
        main_layout.addWidget(control_panel, 1)
        
        # Auto-play timer
        self.auto_play_timer = QTimer()
        self.auto_play_timer.timeout.connect(self.auto_play_step)
        self.auto_playing = False
        
        # Initialize status after all widgets are created
        self.update_status()
        
    def update_board_display(self):
        """Update the SVG board display"""
        svg = chess.svg.board(
            board=self.board,
            size=600,
            lastmove=self.board.move_stack[-1] if self.board.move_stack else None,
            check=self.board.king(self.board.turn) if self.board.is_check() else None
        )
        self.svg_widget.load(svg.encode('utf-8'))
        
    def update_status(self):
        """Update the status label with game information"""
        status = []
        status.append(f"Turn: {'White' if self.board.turn else 'Black'}")
        
        if self.board.is_checkmate():
            status.append("Checkmate!")
        elif self.board.is_check():
            status.append("Check!")
        elif self.board.is_stalemate():
            status.append("Stalemate")
        elif self.board.is_insufficient_material():
            status.append("Draw by insufficient material")
        elif self.board.is_game_over():
            status.append("Game over")
            
        self.status_label.setText("\n".join(status))
        
        # Update evaluation
        features = self.bot.extract_features(self.board)
        feature_text = "\n".join([f"{name}: {value:.2f}" 
                                for name, value in zip(self.bot.feature_names, features)])
        self.eval_label.setText(f"Position Evaluation:\n{feature_text}")
    
    def reset_game(self):
        """Reset the game to starting position"""
        self.board.reset()
        self.update_board_display()
        self.update_status()
        
    def undo_move(self):
        """Undo the last move"""
        if self.board.move_stack:
            self.board.pop()
            self.update_board_display()
            self.update_status()
    
    def make_bot_move(self):
        """Make the bot play a move"""
        if not self.board.is_game_over():
            move = self.bot.predict_move(self.board)
            self.board.push(move)
            self.update_board_display()
            self.update_status()
    
    def toggle_auto_play(self):
        """Toggle auto-play mode"""
        self.auto_playing = not self.auto_playing
        if self.auto_playing:
            self.auto_play_button.setText("Stop Auto Play")
            self.auto_play_timer.start(1000)  # 1 second between moves
        else:
            self.auto_play_button.setText("Auto Play")
            self.auto_play_timer.stop()
    
    def auto_play_step(self):
        """Perform one step of auto-play"""
        if not self.board.is_game_over():
            self.make_bot_move()
            self.move_counter += 1
            self.move_counter_label.setText(f"Move Counter: {self.move_counter}")
        else:
            self.toggle_auto_play()
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for making moves"""
        if event.button() == Qt.LeftButton:
            # Get the clicked square
            pos = self.svg_widget.mapFromParent(event.pos())
            if 0 <= pos.x() < self.svg_widget.width() and 0 <= pos.y() < self.svg_widget.height():
                # Calculate square coordinates
                square_size = self.svg_widget.width() / 8
                file = int(pos.x() / square_size)
                rank = 7 - int(pos.y() / square_size)
                square = chess.square(file, rank)
                
                # Handle move selection
                if hasattr(self, 'selected_square'):
                    # Try to make a move
                    move = chess.Move(self.selected_square, square)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        self.update_board_display()
                        self.update_status()
                        del self.selected_square
                        
                        # Bot can play automatically after human move
                        if self.auto_playing and not self.board.is_game_over():
                            QTimer.singleShot(500, self.make_bot_move)
                    else:
                        # Select a different piece
                        self.selected_square = square
                else:
                    # Select a piece
                    piece = self.board.piece_at(square)
                    if piece and piece.color == self.board.turn:
                        self.selected_square = square
                
                self.update_board_display()

def main():
    # Initialize the chess bot
    bot = NigelChessBot()
    # Train the bot
    bot = train_set(bot)
    bot.save("NigelBotData/chess_bot.joblib")
    bot = NigelChessBot.load("NigelBotData/chess_bot.joblib")

    
    # Create and show the GUI
    app = QApplication(sys.argv)
    gui = ChessGUI(bot)
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()