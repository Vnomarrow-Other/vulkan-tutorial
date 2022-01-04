#[cfg(test)]
mod chess_lib_test {
    pub extern crate chess_engine;
    use chess_engine::chess_game::*;
    #[test]
    fn tests_working() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn bishop_test() {
        // Make sure it can move sideways
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Bishop, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 5, 6);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);

        // Make sure it cannot move if it is not sideways
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Bishop, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 6, 6);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure it can move sideways
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Bishop, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 2, 5);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);

        // Make sure it cannot move if it is not sideways
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Bishop, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 2, 2);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure it cannot move if there is a piece in the way
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Bishop, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(2, 3), ChessPieceId::Bishop, ChessPieceColor::Black);
        let board_move = BoardMove::new(3, 4, 1, 2);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);
    }
    #[test]
    fn rook_test() {
        // Make sure it cannot move if it is not straight
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Rook, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 5, 6);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure it can move straight along x
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Rook, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 3, 6);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);

        // Make sure it can move straight along y
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Rook, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 6, 4);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);
    }

    #[test]
    fn knight_test() {
        let mut game = Game::new();
        // Make sure it can move a corect move
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Knight, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 5, 5);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);

        // Make sure it cannot move incorrectly
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Knight, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 1, 2);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure it can move a corect move
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Knight, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 1, 3);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);

        // Make sure it cannot move incorrectly
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Knight, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 3, 5);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);
    }

    #[test]
    fn queen_test() {
        // Make sure it cannot move incorrectly
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Queen, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 5, 5);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure it can move diagonally
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Queen, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 1, 2);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);

        // Make sure it cannot move incorrectly
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Queen, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 1, 3);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure it can move straight
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::Queen, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 4, 3, 5);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);
    }

    #[test]
    fn check_test() {
        // Make sure rook is checking king
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::King, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(4, 4), ChessPieceId::Rook, ChessPieceColor::Black);
        assert_eq!(game.is_check().is_some(), true);

        // Make sure check is not given when current player is not in check
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::King, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(4, 4), ChessPieceId::Rook, ChessPieceColor::White);
        assert_eq!(game.is_check().is_some(), false); // Since it is white who starts

        // Make sure it is not check when game starts
        let mut game = Game::new();
        game.set_up_board();
        assert_eq!(game.is_check().is_some(), false);
    }

    #[test]
    fn check_stop_move_test() {
        // Make sure rook cannot move when king is in check
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(1, 1), ChessPieceId::Rook, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(3, 4), ChessPieceId::King, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(4, 4), ChessPieceId::Rook, ChessPieceColor::Black);
        let board_move = BoardMove::new(1, 1, 1, 2);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);
    }

    #[test]
    fn pawn_test() {
        // Make sure pawn can move one forward
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(1, 6), ChessPieceId::Pawn, ChessPieceColor::White);
        let board_move = BoardMove::new(1, 6, 1, 5);
        assert_eq!(game.move_piece(board_move, false, None).is_ok(), true);

        // Make sure pawn can move two forward
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(1, 6), ChessPieceId::Pawn, ChessPieceColor::White);
        let board_move = BoardMove::new(1, 6, 1, 4);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);

        // Make sure white pawn cannot move backward
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(1, 6), ChessPieceId::Pawn, ChessPieceColor::White);
        let board_move = BoardMove::new(1, 6, 1, 7);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure black pawn can move two forward
        let mut game = Game::new();
        game.empty_board();
        game.turn = ChessPieceColor::Black;
        game.set_pos(BoardPosition::new(1, 1), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(1, 1, 1, 3);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);

        // Make sure black pawn can move one forward
        let mut game = Game::new();
        game.empty_board();
        game.turn = ChessPieceColor::Black;
        game.set_pos(BoardPosition::new(1, 1), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(1, 1, 1, 2);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);

        // Make sure black pawn cannot move three forward
        let mut game = Game::new();
        game.empty_board();
        game.turn = ChessPieceColor::Black;
        game.set_pos(BoardPosition::new(1, 1), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(1, 1, 1, 4);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure black pawn cannot move backward
        let mut game = Game::new();
        game.empty_board();
        game.turn = ChessPieceColor::Black;
        game.set_pos(BoardPosition::new(1, 1), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(1, 1, 1, 0);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure black pawn cannot move to other side of board
        let mut game = Game::new();
        game.empty_board();
        game.turn = ChessPieceColor::Black;
        game.set_pos(BoardPosition::new(2, 1), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(2, 1, 3, 7);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure black pawn cannot move 2 after having moved 1
        let mut game = Game::new();
        game.empty_board();
        game.turn = ChessPieceColor::Black;
        game.set_pos(BoardPosition::new(2, 1), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(2, 1, 2, 2);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(2, 2)).is_some(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(2, 2)).unwrap().moved, true);
        game.turn = ChessPieceColor::Black;
        let board_move = BoardMove::new(2, 2, 2, 4);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(2, 2)).is_some(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(2, 4)).is_some(), false);

        // Make sure white pawn cannot move 2 after having moved 1
        let mut game = Game::new();
        game.empty_board();
        game.turn = ChessPieceColor::White;
        game.set_pos(BoardPosition::new(4, 6), ChessPieceId::Pawn, ChessPieceColor::White);
        let board_move = BoardMove::new(4, 6, 4, 5);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(4, 5)).is_some(), true);
        game.turn = ChessPieceColor::White;
        let board_move = BoardMove::new(4, 5, 4, 3);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(4, 5)).is_some(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(4, 3)).is_some(), false);
        
    }

    #[test]
    fn convert_test() {
        // Make sure you cannot convert pawn to pawn
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 1), ChessPieceId::Pawn, ChessPieceColor::White);
        let board_move = BoardMove::new(0, 1, 0, 0);
        assert_eq!(game.move_piece(board_move, true, Some(ChessPieceId::Pawn)).is_ok(), false);

        // Make sure you cannot convert pawn to king
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 1), ChessPieceId::Pawn, ChessPieceColor::White);
        let board_move = BoardMove::new(0, 1, 0, 0);
        assert_eq!(game.move_piece(board_move, true, Some(ChessPieceId::King)).is_ok(), false);

        // Make sure you cannot convert to nothing
        let mut game = Game::new();
        game.empty_board();
        game.turn = ChessPieceColor::Black;
        game.set_pos(BoardPosition::new(0, 6), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(0, 6, 0, 7);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);

        // Make sure white can convert pawn to queen
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 1), ChessPieceId::Pawn, ChessPieceColor::White);
        let board_move = BoardMove::new(0, 1, 0, 0);
        assert_eq!(game.move_piece(board_move, true, Some(ChessPieceId::Queen)).is_ok(), true);

        // Make sure white cannot convert pawn to queen on whites side of the board
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 6), ChessPieceId::Pawn, ChessPieceColor::White);
        let board_move = BoardMove::new(0, 6, 0, 7);
        assert_eq!(game.move_piece(board_move, true, Some(ChessPieceId::Queen)).is_ok(), false);

        // Make sure black can convert pawn to queen on whites side of the board
        let mut game = Game::new();
        game.empty_board();
        game.turn = ChessPieceColor::Black;
        game.set_pos(BoardPosition::new(0, 6), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(0, 6, 0, 7);
        assert_eq!(game.move_piece(board_move, true, Some(ChessPieceId::Queen)).is_ok(), true);

        // Make sure black cannot convert pawn to queen on blacks side of the board
        let mut game = Game::new();
        game.empty_board();
        game.turn = ChessPieceColor::Black;
        game.set_pos(BoardPosition::new(0, 6), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(0, 1, 0, 0);
        assert_eq!(game.move_piece(board_move, true, Some(ChessPieceId::Queen)).is_ok(), false);

        // Make sure black cannot convert pawn to queen when it is not blacks turn
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 6), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(0, 6, 0, 7);
        assert_eq!(game.move_piece(board_move, true, Some(ChessPieceId::Queen)).is_ok(), false);
    }

    #[test]
    fn castle_test() {
        // Make sure you can castle to the left
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 0), ChessPieceId::Rook, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(3, 0), ChessPieceId::King, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 0, 1, 0);
        let move_result = game.move_piece(board_move, false, None);
        assert_eq!(move_result.is_ok(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(0, 0)).is_some(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(3, 0)).is_some(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(1, 0)).unwrap().id == ChessPieceId::King, true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(2, 0)).unwrap().id == ChessPieceId::Rook, true);

        // Make sure you can castle to the right
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(7, 0), ChessPieceId::Rook, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(3, 0), ChessPieceId::King, ChessPieceColor::White);
        let board_move = BoardMove::new(3, 0, 5, 0);
        let move_result = game.move_piece(board_move, false, None);
        assert_eq!(move_result.is_ok(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(7, 0)).is_some(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(3, 0)).is_some(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(5, 0)).unwrap().id == ChessPieceId::King, true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(4, 0)).unwrap().id == ChessPieceId::Rook, true);

        // Make sure you cannot castle if king has moved
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 0), ChessPieceId::Rook, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(3, 0), ChessPieceId::King, ChessPieceColor::White);
        let board_ref = game.get_board_ref(BoardPosition::new(3, 0)).unwrap();
        board_ref.as_mut().unwrap().moved = true;
        let board_move = BoardMove::new(3, 0, 1, 0);
        assert_eq!(game.move_piece(board_move, false, None).is_ok(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(0, 0)).is_some(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(3, 0)).is_some(), true);

        // Make sure you cannot castle if rook has moved
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 0), ChessPieceId::Rook, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(3, 0), ChessPieceId::King, ChessPieceColor::White);
        let board_ref = game.get_board_ref(BoardPosition::new(0, 0)).unwrap();
        board_ref.as_mut().unwrap().moved = true;
        //game.get_board_ref(BoardPosition::new(0, 0)).unwrap().as_mut().unwrap().moved = true;
        let board_move = BoardMove::new(3, 0, 1, 0);
        assert_eq!(game.move_piece(board_move, false, None).is_ok(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(0, 0)).is_some(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(3, 0)).is_some(), true);

        // Make sure you cannot castle when one of the squares is in check
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(7, 0), ChessPieceId::Rook, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(3, 0), ChessPieceId::King, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(4, 2), ChessPieceId::Rook, ChessPieceColor::Black);
        let board_move = BoardMove::new(3, 0, 5, 0);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(7, 0)).is_some(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(3, 0)).is_some(), true);

        // Make sure you can castle when the rook is threatened
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(7, 0), ChessPieceId::Rook, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(3, 0), ChessPieceId::King, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(7, 2), ChessPieceId::Rook, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(6, 2), ChessPieceId::Rook, ChessPieceColor::Black);
        let board_move = BoardMove::new(3, 0, 5, 0);
        assert_eq!(game.move_piece(board_move, true, None).is_ok(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(7, 0)).is_some(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(3, 0)).is_some(), false);
    }

    #[test]
    fn pessant_test() {
        // Make sure you can do pessant
        let mut game = Game::new();
        game.turn = ChessPieceColor::Black;
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 1), ChessPieceId::Pawn, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(1, 3), ChessPieceId::Pawn, ChessPieceColor::White);
        let board_move = BoardMove::new(0, 1, 0, 3);
        assert_eq!(game.move_piece(board_move, false, None).is_ok(), true);
        let board_move = BoardMove::new(1, 3, 0, 2);
        assert_eq!(game.move_piece(board_move, false, None).is_ok(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(0, 3)).is_some(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(1, 3)).is_some(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(0, 2)).unwrap().id == ChessPieceId::Pawn, true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(0, 2)).unwrap().color == ChessPieceColor::White, true);

        // Make sure you cannot do pessant backwards
        let mut game = Game::new();
        game.turn = ChessPieceColor::Black;
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 1), ChessPieceId::Pawn, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(1, 3), ChessPieceId::Pawn, ChessPieceColor::Black);
        let board_move = BoardMove::new(0, 1, 0, 3);
        assert_eq!(game.move_piece(board_move, false, None).is_ok(), false);
        let board_move = BoardMove::new(1, 3, 0, 2);
        assert_eq!(game.move_piece(board_move, false, None).is_ok(), false);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(0, 1)).is_some(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(1, 3)).is_some(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(0, 2)).is_some(), false);

        // Make sure you cannot do pessant with other pieces
        let mut game = Game::new();
        game.turn = ChessPieceColor::Black;
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 1), ChessPieceId::Pawn, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(1, 3), ChessPieceId::Queen, ChessPieceColor::White);
        let board_move = BoardMove::new(0, 1, 0, 3);
        assert_eq!(game.move_piece(board_move, false, None).is_ok(), true);
        let board_move = BoardMove::new(1, 3, 0, 2);
        assert_eq!(game.move_piece(board_move, false, None).is_ok(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(0, 3)).is_some(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(0, 2)).is_some(), true);
        assert_eq!(game.get_board_piece_clone(BoardPosition::new(1, 3)).is_some(), false);
    }

    #[test]
    fn check_mate_test() {
        // Make sure program can detect a check mate
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 0), ChessPieceId::Rook, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(0, 1), ChessPieceId::Rook, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(5, 0), ChessPieceId::King, ChessPieceColor::White);
        assert_eq!(game.is_check().is_some(), true);
        assert_eq!(game.get_possible_moves().is_empty(), true);
        assert_eq!(game.is_check_mate(), true);
        assert_eq!(game.get_winner().unwrap() == ChessPieceColor::Black, true);

        // make sure program does not give false check mates
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 0), ChessPieceId::Rook, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(0, 1), ChessPieceId::Rook, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(5, 1), ChessPieceId::King, ChessPieceColor::White);
        assert_eq!(game.is_check().is_some(), true);
        assert_eq!(game.get_possible_moves().is_empty(), false);
        assert_eq!(game.is_check_mate(), false);
        assert_eq!(game.get_winner() == None, true);

        // Make sure it is not check mate when the game starts
        let mut game = Game::new();
        game.set_up_board();
        assert_eq!(game.is_check_mate(), false);
    }
    #[test]
    fn stale_mate_test() {
        // Make sure a check mate is not a stale mate
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 0), ChessPieceId::Rook, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(0, 1), ChessPieceId::Rook, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(5, 0), ChessPieceId::King, ChessPieceColor::White);
        assert_eq!(game.is_check().is_some(), true);
        assert_eq!(game.get_possible_moves().is_empty(), true);
        assert_eq!(game.is_stale_mate(), false);

        // Make sure program does not give false stale mates
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 0), ChessPieceId::Rook, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(0, 1), ChessPieceId::Rook, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(5, 1), ChessPieceId::King, ChessPieceColor::White);
        assert_eq!(game.is_check().is_some(), true);
        assert_eq!(game.get_possible_moves().is_empty(), false);
        assert_eq!(game.is_stale_mate(), false);

        // Make sure a stale mate is a stale mate
        let mut game = Game::new();
        game.empty_board();
        game.set_pos(BoardPosition::new(0, 0), ChessPieceId::King, ChessPieceColor::White);
        game.set_pos(BoardPosition::new(1, 5), ChessPieceId::Rook, ChessPieceColor::Black);
        game.set_pos(BoardPosition::new(5, 1), ChessPieceId::Rook, ChessPieceColor::Black);
        assert_eq!(game.is_check().is_some(), false);
        assert_eq!(game.get_possible_moves().is_empty(), true);
        assert_eq!(game.is_stale_mate(), true);
        assert_eq!(game.get_winner() == None, true);

        // Make sure it is not stale mate when the game starts
        let mut game = Game::new();
        game.set_up_board();
        assert_eq!(game.is_stale_mate(), false);
    }

    #[test]
    fn algebraic_notation_move_test() {
        // Make sure you can move pawn one forward
        let mut game = Game::new();
        game.set_up_board();
        let pos = BoardPosition::from_algebraic_notation("a3").unwrap();
        assert_eq!(game.algebraic_notation_move("a3".to_string()).is_ok(), true);
        assert_eq!(game.get_board_piece_clone(pos).is_some(), true);

        let mut game = Game::new();
        game.set_up_board();
        let pos = BoardPosition::from_algebraic_notation("h3").unwrap();
        assert_eq!(game.algebraic_notation_move("h3".to_string()).is_ok(), true);
        assert_eq!(game.get_board_piece_clone(pos).is_some(), true);

        // Make sure you cannot move pawn that far
        let mut game = Game::new();
        game.set_up_board();
        let pos = BoardPosition::from_algebraic_notation("a5").unwrap();
        assert_eq!(game.algebraic_notation_move("a5".to_string()).is_ok(), false);
        assert_eq!(game.get_board_piece_clone(pos).is_some(), false);

        // Make sure you can convert pawn
        let mut game = Game::new();
        game.empty_board();
        let from_pos = BoardPosition::from_algebraic_notation("a7").unwrap();
        let to_pos = BoardPosition::from_algebraic_notation("a8").unwrap();
        game.set_pos(from_pos, ChessPieceId::Pawn, ChessPieceColor::White);
        assert_eq!(game.algebraic_notation_move("a8Q".to_string()).is_ok(), true);
        assert_eq!(game.get_board_piece_clone(to_pos).is_some(), true);
        assert_eq!(game.get_board_piece_clone(to_pos).unwrap().id == ChessPieceId::Queen, true);

        // Make sure you cannot move to other side if you do not promote
        let mut game = Game::new();
        game.empty_board();
        let from_pos = BoardPosition::from_algebraic_notation("a7").unwrap();
        let to_pos = BoardPosition::from_algebraic_notation("a8").unwrap();
        game.set_pos(from_pos, ChessPieceId::Pawn, ChessPieceColor::White);
        assert_eq!(game.algebraic_notation_move("a8".to_string()).is_ok(), false);
        assert_eq!(game.get_board_piece_clone(to_pos).is_some(), false);
        assert_eq!(game.get_board_piece_clone(from_pos).is_some(), true);

        // Make sure you cannot specify a promote piece and not promote
        let mut game = Game::new();
        game.set_up_board();
        let pos = BoardPosition::from_algebraic_notation("a3").unwrap();
        assert_eq!(game.algebraic_notation_move("a3Q".to_string()).is_ok(), false);
        assert_eq!(game.get_board_piece_clone(pos).is_some(), false);

        // Make sure you cannot move piece if it is unclear which piece should move
        let mut game = Game::new();
        game.empty_board();
        let to_pos = BoardPosition::from_algebraic_notation("b1").unwrap();
        let from_pos1 = BoardPosition::from_algebraic_notation("a1").unwrap();
        let from_pos2 = BoardPosition::from_algebraic_notation("e1").unwrap();
        game.set_pos(from_pos1, ChessPieceId::Rook, ChessPieceColor::White);
        game.set_pos(from_pos2, ChessPieceId::Rook, ChessPieceColor::White);
        assert_eq!(game.algebraic_notation_move("Rb1".to_string()).is_ok(), false);
        assert_eq!(game.get_board_piece_clone(to_pos).is_some(), false);
        assert_eq!(game.get_board_piece_clone(from_pos1).is_some(), true);
        assert_eq!(game.get_board_piece_clone(from_pos2).is_some(), true);

        // Make sure you can move when piece is specified
        let mut game = Game::new();
        game.empty_board();
        let to_pos = BoardPosition::from_algebraic_notation("b1").unwrap();
        let from_pos1 = BoardPosition::from_algebraic_notation("a1").unwrap();
        let from_pos2 = BoardPosition::from_algebraic_notation("e1").unwrap();
        game.set_pos(from_pos1, ChessPieceId::Rook, ChessPieceColor::White);
        game.set_pos(from_pos2, ChessPieceId::Rook, ChessPieceColor::White);
        assert_eq!(game.algebraic_notation_move("Reb1".to_string()).is_ok(), true);
        assert_eq!(game.get_board_piece_clone(to_pos).is_some(), true);
        assert_eq!(game.get_board_piece_clone(from_pos1).is_some(), true);
        assert_eq!(game.get_board_piece_clone(from_pos2).is_some(), false);


        // Make sure program does not crash with wierd input
        let mut game = Game::new();
        game.set_up_board();
        assert_eq!(game.algebraic_notation_move("a9".to_string()).is_ok(), false);

        // Make sure program does not crash with wierd input
        let mut game = Game::new();
        game.set_up_board();
        assert_eq!(game.algebraic_notation_move("a0".to_string()).is_ok(), false);

        // Make sure program does not crash with wierd input
        let mut game = Game::new();
        game.set_up_board();
        assert_eq!(game.algebraic_notation_move("i1".to_string()).is_ok(), false);

        // Make sure program does not crash with wierd input
        let mut game = Game::new();
        game.set_up_board();
        assert_eq!(game.algebraic_notation_move("a2 a3".to_string()).is_ok(), false);
    }
}
