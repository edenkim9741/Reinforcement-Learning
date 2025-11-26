import chess

# =============================================================================
#  [PART 1] Action Encoding Logic
#  (PettingZoo 호환을 위해 chess.Move를 int형 Action ID로 변환)
# =============================================================================

def square_to_coord(s):
    col = s % 8
    row = s // 8
    return (col, row)

def diff(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return (x2 - x1, y2 - y1)

def sign(v):
    return -1 if v < 0 else (1 if v > 0 else 0)

def mirror_move(move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )

def get_queen_dir(diff):
    dx, dy = diff
    assert dx == 0 or dy == 0 or abs(dx) == abs(dy)
    magnitude = max(abs(dx), abs(dy)) - 1
    counter = 0
    for x in range(-1, 1 + 1):
        for y in range(-1, 1 + 1):
            if x == 0 and y == 0: continue
            if x == sign(dx) and y == sign(dy): return magnitude, counter
            counter += 1
    return 0, 0

def get_queen_plane(diff):
    NUM_COUNTERS = 8
    mag, counter = get_queen_dir(diff)
    return mag * NUM_COUNTERS + counter

def get_knight_dir(diff):
    dx, dy = diff
    counter = 0
    for x in range(-2, 2 + 1):
        for y in range(-2, 2 + 1):
            if abs(x) + abs(y) == 3:
                if dx == x and dy == y: return counter
                counter += 1
    return 0

def is_knight_move(diff):
    dx, dy = diff
    return abs(dx) + abs(dy) == 3 and 1 <= abs(dx) <= 2

def get_pawn_promotion_move(diff):
    dx, dy = diff
    return dx + 1

def get_pawn_promotion_num(promotion):
    return 0 if promotion == chess.KNIGHT else (1 if promotion == chess.BISHOP else 2)

def get_move_plane(move):
    source = move.from_square
    dest = move.to_square
    difference = diff(square_to_coord(source), square_to_coord(dest))

    QUEEN_MOVES = 56
    KNIGHT_MOVES = 8
    QUEEN_OFFSET = 0
    KNIGHT_OFFSET = QUEEN_MOVES
    UNDER_OFFSET = KNIGHT_OFFSET + KNIGHT_MOVES

    if is_knight_move(difference):
        return KNIGHT_OFFSET + get_knight_dir(difference)
    else:
        if move.promotion is not None and move.promotion != chess.QUEEN:
            return (
                UNDER_OFFSET
                + 3 * get_pawn_promotion_move(difference)
                + get_pawn_promotion_num(move.promotion)
            )
        else:
            return QUEEN_OFFSET + get_queen_plane(difference)

def encode_move(move: chess.Move, should_mirror: bool) -> int:
    """
    chess.Move 객체를 PettingZoo Action ID(int)로 변환합니다.
    should_mirror: 현재 턴이 Black이라면 True (PettingZoo는 흑번일 때 보드를 뒤집어서 계산함)
    """
    if should_mirror:
        move = mirror_move(move)

    TOTAL = 73
    source = move.from_square
    coord = square_to_coord(source)
    panel = get_move_plane(move)
    
    # (row * 8 + col) * 73 + panel
    # 주의: square_to_coord의 리턴은 (col, row) 형태
    cur_action = (coord[0] * 8 + coord[1]) * TOTAL + panel
    return cur_action


# =============================================================================
#  [PART 2] Minimax Algorithm
# =============================================================================

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def evaluate_board(board: chess.Board):
    if board.is_checkmate():
        # White 승리시 양수, Black 승리시 음수
        return -9999 if board.turn == chess.WHITE else 9999
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    # 모든 기물 스캔
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    return score

def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    legal_moves = list(board.legal_moves)
    # 간단한 Move Ordering (잡는 수 우선 탐색 - 성능 향상용)
    # legal_moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    if maximizing_player:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def get_best_move_minimax(board: chess.Board, depth=2):
    """
    Minimax를 통해 최적의 chess.Move를 반환합니다.
    """
    best_move = None
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return None
    
    # White 차례면 Maximize, Black 차례면 Minimize
    maximizing_player = (board.turn == chess.WHITE)
    
    if maximizing_player:
        best_value = -float('inf')
        for move in legal_moves:
            board.push(move)
            val = minimax(board, depth - 1, -float('inf'), float('inf'), False)
            board.pop()
            if val > best_value:
                best_value = val
                best_move = move
    else:
        best_value = float('inf')
        for move in legal_moves:
            board.push(move)
            val = minimax(board, depth - 1, -float('inf'), float('inf'), True)
            board.pop()
            if val < best_value:
                best_value = val
                best_move = move
                
    return best_move