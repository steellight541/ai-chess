import chess
import chess.svg
import chess.pgn
import random
import json
from collections import Counter
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
)
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import QByteArray, QTimer, pyqtSignal, QObject, QThread

PAWN_TABLE = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    10,
    10,
    20,
    30,
    30,
    20,
    10,
    10,
    5,
    5,
    10,
    25,
    25,
    10,
    5,
    5,
    0,
    0,
    0,
    20,
    20,
    0,
    0,
    0,
    5,
    -5,
    -10,
    0,
    0,
    -10,
    -5,
    5,
    5,
    10,
    10,
    -20,
    -20,
    10,
    10,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]
KNIGHT_TABLE = [
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
    -40,
    -20,
    0,
    0,
    0,
    0,
    -20,
    -40,
    -30,
    0,
    10,
    15,
    15,
    10,
    0,
    -30,
    -30,
    5,
    15,
    20,
    20,
    15,
    5,
    -30,
    -30,
    0,
    15,
    20,
    20,
    15,
    0,
    -30,
    -30,
    5,
    10,
    15,
    15,
    10,
    5,
    -30,
    -40,
    -20,
    0,
    5,
    5,
    0,
    -20,
    -40,
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
]
BISHOP_TABLE = [
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -10,
    0,
    5,
    10,
    10,
    5,
    0,
    -10,
    -10,
    5,
    5,
    10,
    10,
    5,
    5,
    -10,
    -10,
    0,
    10,
    10,
    10,
    10,
    0,
    -10,
    -10,
    10,
    10,
    10,
    10,
    10,
    10,
    -10,
    -10,
    5,
    0,
    0,
    0,
    0,
    5,
    -10,
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
]
ROOK_TABLE = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    5,
    10,
    10,
    10,
    10,
    10,
    10,
    5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    0,
    0,
    0,
    5,
    5,
    0,
    0,
    0,
]
QUEEN_TABLE = [
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -10,
    0,
    5,
    5,
    5,
    5,
    0,
    -10,
    -5,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    0,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    -10,
    5,
    5,
    5,
    5,
    5,
    0,
    -10,
    -10,
    0,
    5,
    0,
    0,
    0,
    0,
    -10,
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
]
KING_TABLE = [
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -20,
    -30,
    -30,
    -40,
    -40,
    -30,
    -30,
    -20,
    -10,
    -20,
    -20,
    -20,
    -20,
    -20,
    -20,
    -10,
    20,
    20,
    0,
    0,
    0,
    0,
    20,
    20,
    20,
    30,
    10,
    0,
    0,
    10,
    30,
    20,
]


class ChessBoardWidget(QSvgWidget):
    pieceMoved = pyqtSignal(int, int)
    squareSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.drag_start_square = None

    def mousePressEvent(self, event):
        pos = event.pos()
        square_size = min(self.width(), self.height()) / 8.0
        file = int(pos.x() // square_size)
        rank = 7 - int(pos.y() // square_size)
        square = chess.square(file, rank)
        self.squareSelected.emit(square)
        self.drag_start_square = square
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.drag_start_square is None:
            return
        pos = event.pos()
        square_size = min(self.width(), self.height()) / 8.0
        file = int(pos.x() // square_size)
        rank = 7 - int(pos.y() // square_size)
        drag_end_square = chess.square(file, rank)
        self.pieceMoved.emit(self.drag_start_square, drag_end_square)
        self.drag_start_square = None
        event.accept()


class ChessAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess AI")
        self.setGeometry(100, 100, 600, 650)
        self.layout = QVBoxLayout()
        self.board_widget = ChessBoardWidget()
        self.board_widget.setMinimumSize(600, 600)
        self.layout.addWidget(self.board_widget)
        self.board_widget.pieceMoved.connect(self.handle_drag_move)
        self.board_widget.squareSelected.connect(self.handle_square_click)
        self.move_input = QLineEdit()
        self.move_input.setPlaceholderText("Enter your move in UCI format (e.g. e2e4)")
        self.move_input.returnPressed.connect(self.make_move)
        self.layout.addWidget(self.move_input)
        self.ai_vs_ai_button = QPushButton("Start AI vs AI")
        self.ai_vs_ai_button.clicked.connect(self.start_ai_vs_ai)
        self.layout.addWidget(self.ai_vs_ai_button)
        self.move_counter_label = QLabel("Moves: 0")
        self.layout.addWidget(self.move_counter_label)
        self.setLayout(self.layout)
        self.board = chess.Board()
        self.openings = self.load_openings("NielsBotData/.Dataset/openings.json")
        self.selected_square = None
        self.possible_moves = []
        self.update_board()
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.ai_move)

    def handle_square_click(self, square):
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.possible_moves = [
                    move.to_square
                    for move in self.board.legal_moves
                    if move.from_square == square
                ]
            else:
                self.selected_square = None
                self.possible_moves = []
        else:
            move = chess.Move(self.selected_square, square)
            if self.board.piece_type_at(self.selected_square) == chess.PAWN and (
                (self.board.turn == chess.WHITE and chess.square_rank(square) == 7)
                or (self.board.turn == chess.BLACK and chess.square_rank(square) == 0)
            ):
                move.promotion = chess.QUEEN
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.possible_moves = []
                if not self.board.is_game_over():
                    self.start_ai_worker()
            else:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    self.possible_moves = [
                        move.to_square
                        for move in self.board.legal_moves
                        if move.from_square == square
                    ]
                else:
                    self.selected_square = None
                    self.possible_moves = []
        self.update_board()

    def update_board(self):
        squares = {}
        if self.selected_square is not None:
            squares[self.selected_square] = "rgba(173, 216, 230, 0.5)"
            for move_square in self.possible_moves:
                squares[move_square] = "rgba(128, 128, 128, 0.5)"
        svg = chess.svg.board(self.board, squares=squares, size=600).encode("utf-8")
        self.board_widget.load(QByteArray(svg))
        self.update_move_counter()

    def handle_drag_move(self, from_square, to_square):
        move = chess.Move(from_square, to_square)
        if move not in self.board.legal_moves:
            piece = self.board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                if (
                    piece.color == chess.WHITE and chess.square_rank(to_square) == 7
                ) or (piece.color == chess.BLACK and chess.square_rank(to_square) == 0):
                    move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.selected_square = None
            self.possible_moves = []
            self.update_board()
            if not self.board.is_game_over():
                self.start_ai_worker()

    def make_move(self):
        user_input = self.move_input.text().strip()
        if user_input:
            try:
                move = chess.Move.from_uci(user_input)
            except ValueError:
                return
            if move not in self.board.legal_moves:
                if len(user_input) == 4:
                    move = chess.Move.from_uci(user_input + "q")
                    if move not in self.board.legal_moves:
                        return
                else:
                    return
            self.board.push(move)
            self.selected_square = None
            self.possible_moves = []
            self.update_board()
            self.move_input.clear()
            if not self.board.is_game_over():
                self.start_ai_worker()

    def start_ai_worker(self):
        self.ai_worker = AIWorker(self.board.copy(), self.openings)
        self.ai_thread = QThread()
        self.ai_worker.moveToThread(self.ai_thread)
        self.ai_worker.finished.connect(self.handle_ai_move)
        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_thread.start()

    def handle_ai_move(self, move):
        self.board.push(move)
        self.update_board()
        self.ai_thread.quit()
        self.ai_thread.wait()

    def ai_move(self):
        if not self.board.is_game_over():
            self.board.push(self.best_move())
            self.update_board()

    def start_ai_vs_ai(self):
        self.board.reset()
        self.update_board()
        self.ai_timer.start(1000)

    def load_openings(self, json_filepath):
        try:
            with open(json_filepath, "r") as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            return {}

    def update_move_counter(self):
        self.move_counter_label.setText("Moves: " + str(len(self.board.move_stack)))

    def evaluate_board(self):
        eval_score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                color_factor = 1 if piece.color == chess.WHITE else -1
                table_square = (
                    square
                    if piece.color == chess.WHITE
                    else chess.square_mirror(square)
                )
                if piece.piece_type == chess.PAWN:
                    eval_score += (
                        PAWN_TABLE[table_square] * color_factor + 100 * color_factor
                    )
                elif piece.piece_type == chess.KNIGHT:
                    eval_score += (
                        KNIGHT_TABLE[table_square] * color_factor + 300 * color_factor
                    )
                elif piece.piece_type == chess.BISHOP:
                    eval_score += (
                        BISHOP_TABLE[table_square] * color_factor + 300 * color_factor
                    )
                elif piece.piece_type == chess.ROOK:
                    eval_score += (
                        ROOK_TABLE[table_square] * color_factor + 500 * color_factor
                    )
                elif piece.piece_type == chess.QUEEN:
                    eval_score += (
                        QUEEN_TABLE[table_square] * color_factor + 900 * color_factor
                    )
                elif piece.piece_type == chess.KING:
                    eval_score += KING_TABLE[table_square] * color_factor
        return eval_score

    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.board.is_game_over():
            return self.evaluate_board()
        if maximizing_player:
            max_eval = -float("inf")
            for move in self.order_moves(self.board.legal_moves):
                self.board.push(move)
                eval = self.alpha_beta(depth - 1, alpha, beta, False)
                self.board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in self.order_moves(self.board.legal_moves):
                self.board.push(move)
                eval = self.alpha_beta(depth - 1, alpha, beta, True)
                self.board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def order_moves(self, moves):
        scored_moves = []
        for move in moves:
            score = 0
            if self.board.is_capture(move):
                score += 10
            if self.board.gives_check(move):
                score += 5
            scored_moves.append((score, move))
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]

    def best_move(self, depth=4):
        search_depth = depth
        fen_parts = self.board.fen().split(" ")
        normalized_fen = " ".join(fen_parts[:4])
        if normalized_fen in self.openings:
            possible_moves = [
                chess.Move.from_uci(uci) for uci in self.openings[normalized_fen]
            ]
            legal_moves = [
                move for move in possible_moves if move in self.board.legal_moves
            ]
            if legal_moves:
                return random.choice(legal_moves)  # Changed to random choice
        best_moves = []
        best_eval = -float("inf") if self.board.turn == chess.WHITE else float("inf")
        maximizing = self.board.turn == chess.WHITE
        for move in self.order_moves(self.board.legal_moves):
            self.board.push(move)
            eval = self.alpha_beta(
                search_depth - 1, -float("inf"), float("inf"), not maximizing
            )
            self.board.pop()
            if (maximizing and eval > best_eval) or (
                not maximizing and eval < best_eval
            ):
                best_eval = eval
                best_moves = [move]
            elif eval == best_eval:
                best_moves.append(move)
        if best_moves:
            return random.choice(best_moves)
        else:
            return random.choice(list(self.board.legal_moves))


class AIWorker(QObject):
    finished = pyqtSignal(chess.Move)

    def __init__(self, board, openings):
        super().__init__()
        self.board = board
        self.openings = openings

    def run(self):
        self.finished.emit(self.calculate_best_move())

    def calculate_best_move(self):
        search_depth = 4
        fen_parts = self.board.fen().split(" ")
        normalized_fen = " ".join(fen_parts[:4])
        if normalized_fen in self.openings:
            possible_moves = [
                chess.Move.from_uci(uci) for uci in self.openings[normalized_fen]
            ]
            legal_moves = [
                move for move in possible_moves if move in self.board.legal_moves
            ]
            if legal_moves:
                return random.choice(legal_moves)  # Changed to random choice
        best_moves = []
        best_eval = -float("inf") if self.board.turn == chess.WHITE else float("inf")
        maximizing = self.board.turn == chess.WHITE
        for move in self.order_moves(self.board.legal_moves):
            self.board.push(move)
            eval = self.alpha_beta(
                search_depth - 1, -float("inf"), float("inf"), not maximizing
            )
            self.board.pop()
            if (maximizing and eval > best_eval) or (
                not maximizing and eval < best_eval
            ):
                best_eval = eval
                best_moves = [move]
            elif eval == best_eval:
                best_moves.append(move)
        if best_moves:
            return random.choice(best_moves)
        else:
            return random.choice(list(self.board.legal_moves))

    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.board.is_game_over():
            return self.evaluate_board()
        if maximizing_player:
            max_eval = -float("inf")
            for move in self.order_moves(self.board.legal_moves):
                self.board.push(move)
                eval = self.alpha_beta(depth - 1, alpha, beta, False)
                self.board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in self.order_moves(self.board.legal_moves):
                self.board.push(move)
                eval = self.alpha_beta(depth - 1, alpha, beta, True)
                self.board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_board(self):
        eval_score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                color_factor = 1 if piece.color == chess.WHITE else -1
                table_square = (
                    square
                    if piece.color == chess.WHITE
                    else chess.square_mirror(square)
                )
                if piece.piece_type == chess.PAWN:
                    eval_score += (
                        PAWN_TABLE[table_square] * color_factor + 100 * color_factor
                    )
                elif piece.piece_type == chess.KNIGHT:
                    eval_score += (
                        KNIGHT_TABLE[table_square] * color_factor + 300 * color_factor
                    )
                elif piece.piece_type == chess.BISHOP:
                    eval_score += (
                        BISHOP_TABLE[table_square] * color_factor + 300 * color_factor
                    )
                elif piece.piece_type == chess.ROOK:
                    eval_score += (
                        ROOK_TABLE[table_square] * color_factor + 500 * color_factor
                    )
                elif piece.piece_type == chess.QUEEN:
                    eval_score += (
                        QUEEN_TABLE[table_square] * color_factor + 900 * color_factor
                    )
                elif piece.piece_type == chess.KING:
                    eval_score += KING_TABLE[table_square] * color_factor
        return eval_score

    def order_moves(self, moves):
        scored_moves = []
        for move in moves:
            score = 0
            if self.board.is_capture(move):
                score += 10
            if self.board.gives_check(move):
                score += 5
            scored_moves.append((score, move))
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]


if __name__ == "__main__":
    app = QApplication([])
    window = ChessAI()
    window.show()
    app.exec_()
