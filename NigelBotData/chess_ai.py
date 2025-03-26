import chess
import chess.pgn
import chess.svg
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os






class NigelChessBot:
    def __init__(self) -> None: # type: ignore
        """Initialize with enhanced features and search capabilities"""
        self.scaler = StandardScaler()
        self.feature_names = [
            'material_balance',
            'center_control',
            'piece_mobility',
            'king_safety',
            'pawn_structure',
            'threats',
            'developed_pieces',
            'capture_opportunity',
            'hanging_pieces',
            'move_repetition',
            'position_history',
            'king_proximity'
        ]
        
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.5,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        self.move_history = []
        self.position_counts = {}
        
        self.model = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)


    def update_history(self, board):
        """Update move history and position counts"""
        # Get current position signature (excluding move counters)
        position_key = board._transposition_key()
        
        # Update position counts
        self.position_counts[position_key] = self.position_counts.get(position_key, 0) + 1
        
        # Keep history of recent moves (last 10 moves)
        if len(self.move_history) >= 10:
            self.move_history.pop(0)

    def extract_features(self, board):
        """Extract features including move history considerations."""
        features = np.zeros(len(self.feature_names))

        # Update position history before feature extraction
        self.update_history(board)

        # Material Balance (White - Black)
        material_balance = sum(
            self.piece_values[piece.piece_type] * (1 if piece.color == chess.WHITE else -1)
            for square in chess.SQUARES
            if (piece := board.piece_at(square))
        )
        features[0] = material_balance

        # Center Control (Pieces in d4, d5, e4, e5)
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        features[1] = sum(1 for sq in center_squares if board.piece_at(sq))

        # Piece Mobility (Number of legal moves)
        features[2] = board.legal_moves.count()

        # King Safety (Attackers vs Defenders near king)
        king_square = board.king(board.turn)
        attack_score = 0
        
        if king_square:
            # Check squares in king's vicinity (2-square radius)
            king_zone = [
                sq for sq in chess.SQUARES 
                if chess.square_distance(king_square, sq) <= 2
            ]
            
            # Count attackers vs defenders in king's zone
            attacker_color = not board.turn
            attackers = sum(1 for sq in king_zone if board.is_attacked_by(attacker_color, sq))
            defenders = sum(1 for sq in king_zone if board.is_attacked_by(board.turn, sq))
            attack_score = attackers - (defenders * 0.5)
        
        features[3] = attack_score

        # Pawn Structure (Count of doubled, isolated, backward pawns)
        features[4] = self.evaluate_pawn_structure(board)

        # Threats (Checking moves & attack opportunities)
        features[5] = sum(1 for move in board.legal_moves if board.gives_check(move))

        # Developed Pieces (Non-pawn pieces moved from starting positions)
        features[6] = sum(
            1 for square in chess.SQUARES if board.piece_at(square) and board.piece_at(square).piece_type != chess.PAWN
        )

        # Capture Opportunity (Captures available)
        features[7] = sum(1 for move in board.legal_moves if board.is_capture(move))

        # Hanging Pieces (Pieces attacked but not defended)
        features[8] = self.evaluate_hanging_pieces(board)

        # Move Repetition Count
        position_key = board._transposition_key()
        features[9] = self.position_counts.get(position_key, 0)

        # Move Diversity (Total unique moves played)
        features[10] = len(self.move_history) - len(set(self.move_history))

        # King Proximity (Number of friendly pieces near king)
        enemy_king = board.king(not board.turn)
        proximity_score = 0
        
        if enemy_king:
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == board.turn and piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    distance = chess.square_distance(square, enemy_king)
                    proximity_score += (8 - distance) * 0.5  # Closer = higher score
                    
        features[11] = proximity_score  # Add to feature array

        return features
    
    def evaluate_pawn_structure(self, board):
        """Evaluates weak pawn structures: isolated, doubled, or backward pawns."""
        pawn_files = {file: [] for file in range(8)}
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file = chess.square_file(square)
                pawn_files[file].append(square)
        
        penalty = 0
        for file, squares in pawn_files.items():
            if len(squares) > 1:
                penalty += 1  # Doubled pawns
            
            # Isolated pawn (No friendly pawns on adjacent files)
            if file > 0 and not pawn_files[file - 1] and file < 7 and not pawn_files[file + 1]:
                penalty += 2
        
        return penalty

    def evaluate_hanging_pieces(self, board):
        """Counts pieces that are attacked but not defended."""
        hanging_pieces = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                attackers = board.attackers(not piece.color, square)
                defenders = board.attackers(piece.color, square)
                if attackers and not defenders:
                    hanging_pieces += 1
        return hanging_pieces


    
    def predict_move(self, board):
        """Predict move with robust probability handling"""
        # First check for immediate wins/captures
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()
            
            if board.is_capture(move):
                capturing_piece = board.piece_at(move.from_square)
                captured_piece = board.piece_at(move.to_square)
                if captured_piece and self.piece_values[captured_piece.piece_type] >= \
                self.piece_values[capturing_piece.piece_type]:
                    return move
        
        # If no immediate tactics found, use hybrid approach
        if np.random.random() < 0.7:
            search_result, _ = self.search_move(board, depth=2)  # Unpack tuple
            if search_result:
                return search_result
        
        # Fall back to model prediction
        features = self.extract_features(board)
        scaled_features = self.scaler.transform([features])
        
        probas = self.model.predict_proba(scaled_features)[0]
        classes = self.model.classes_
        
        legal_moves = [move for move in board.legal_moves]
        move_scores = []
        
        for move in legal_moves:
            move_uci = move.uci()
            if move_uci in classes:
                idx = np.where(classes == move_uci)[0][0]
                base_score = probas[idx]
            else:
                base_score = 0.001  # Small baseline probability for unseen moves
            
            # Combine with evaluation score
            board.push(move)
            eval_score = self.evaluate_position(board) * 0.01  # Scale to similar range
            board.pop()
            
            total_score = base_score + eval_score
            move_scores.append((move, total_score))
        
        # Ensure all scores are non-negative
        min_score = min(score for _, score in move_scores)
        if min_score <= 0:
            move_scores = [(move, score - min_score + 0.001) for move, score in move_scores]
        
        move_scores.sort(key=lambda x: x[1], reverse=True)
        top_moves = move_scores[:3]
        
        if len(top_moves) > 1:
            weights = np.array([score for _, score in top_moves])
            # Normalize and ensure no negative values
            weights = np.maximum(weights, 0.001)  # Ensure minimum probability
            weights /= weights.sum()  # Normalize
            
            try:
                best_move = np.random.choice(
                    [move for move, _ in top_moves],
                    p=weights
                )
                self.move_history.append(best_move.uci())
                return best_move
            except ValueError:
                # Fallback if normalization fails
                best_move = max(move_scores, key=lambda x: x[1])[0]
                self.move_history.append(best_move.uci())
                return best_move
        
        best_move = max(move_scores, key=lambda x: x[1])[0]
        self.move_history.append(best_move.uci())
        return best_move
       

    def reset_history(self):
        """Clear move history for new game"""
        self.move_history = []
        self.position_counts = {}

    def evaluate_position(self, board):
        """Simple position evaluation as fallback"""
        if board.is_checkmate():
            return -1000 if board.turn == chess.WHITE else 1000
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0
            
        # Basic material evaluation
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        
        # Add features from our model's perspective
        features = self.extract_features(board)
        score += features[1] * 0.2  # Center control
        score += features[2] * 0.1  # Mobility
        score += features[3] * 0.5  # King safety
        score -= features[4] * 0.2  # Pawn structure
        score += features[5] * 0.3  # Threats 
        score += features[6] * 0.1 # Developed pieces
        score += features[7] * 0.1 # Capture opportunity
        score -= features[8] * 0.2
        score -= features[9] * 0.1  # Move repetition
        score -= features[10] * 0.1 # Move diversity
        score += features[11] * 0.2

        
        return score

    def search_move(self, board, depth=2):
        """Basic search with configurable depth - fixed return types"""
        best_move = None
        best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            
            if depth > 1:
                # Recursive search returns (move, score)
                _, score = self.search_move(board, depth-1)
            else:
                # Base case returns score for current position
                score = self.evaluate_position(board)
            
            board.pop()
            
            # Update best move based on score
            if (board.turn == chess.WHITE and score > best_score) or \
            (board.turn == chess.BLACK and score < best_score):
                best_score = score
                best_move = move
        
        # Always return a tuple at all depths
        return (best_move, best_score) if depth > 1 else (best_move, best_score)

    def train(self, games):
        """Train on a collection of games"""
        X = []
        y = []
        
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                # Get features before move
                features = self.extract_features(board)
                X.append(features)
                
                # Store the move as target
                y.append(move.uci())
                
                board.push(move)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return X.shape[0]

    def save(self, filename):
        """Save model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
        }, filename)

    @classmethod
    def load(cls, filename):
        """Load model from file"""
        data = joblib.load(filename)
        bot = cls()
        bot.model = data['model']
        bot.scaler = data['scaler']
        return bot


class ChessAi:
    def __init__(self):
        self.ai = NigelChessBot()


    def get_move(self, board):
        return self.ai.predict_move(board)
    
    def get_svg(self, board):
        return chess.svg.board(board=board)

def train_set(bot) -> NigelChessBot:
    # Create some training games (in practice, load from PGN)
    training_games = []
    for _ in range(100):  # Create 10 random games for demo
        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        
        while not board.is_game_over() and len(list(board.legal_moves)) > 0:
            move = np.random.choice(list(board.legal_moves))
            board.push(move)
            node = node.add_variation(move)
        
        game.headers["Result"] = board.result()
        training_games.append(game)
    
    # Train the bot
    print(f"Training on {len(training_games)} games...")
    num_positions = bot.train(training_games)
    print(f"Trained on {num_positions} board positions.")
    return bot

# Example usage
if __name__ == "__main__":
    # Initialize bot (try both 'random_forest' and 'logistic')
    # bot = NigelChessBot.load("NigelBotData/chess_bot.joblib")
    bot = NigelChessBot()
    
    # Create some training games (in practice, load from PGN)
    training_games = []
    for _ in range(100):  # Create 10 random games for demo
        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        
        while not board.is_game_over() and len(list(board.legal_moves)) > 0:
            move = np.random.choice(list(board.legal_moves))
            board.push(move)
            node = node.add_variation(move)
        
        game.headers["Result"] = board.result()
        training_games.append(game)
    
    # Train the bot
    print(f"Training on {len(training_games)} games...")
    num_positions = bot.train(training_games)
    print(f"Trained on {num_positions} board positions.")

    # save the model
    bot.save("NigelBotData/chess_bot.joblib")

    

