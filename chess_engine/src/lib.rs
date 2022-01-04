pub mod chess_game {
    use std::{collections::LinkedList};

    #[derive(Clone, Copy)]
    pub enum ColorTerminal {
        Black,
        Red,
        Green,
        Yellow,
        Blue,
        Magenta,
        LightBlue,
        White,
    }
    pub fn print_color(text: &str, text_color: ColorTerminal, background_color: ColorTerminal) {
        //let background_mod_8 = background_color as i32 % 8;
        let text_mod_8 = text_color as i32 % 8;
        let background_mod_8 = background_color as i32 % 8;
        print!("\x1b[{}m\x1b[{}m{}\x1b[m", 40+background_mod_8, 30+text_mod_8 as i32, text);
    }

    type BoardPosType = u8;
    #[derive(Clone, Copy, Debug)]
    pub struct BoardPosition {
        x: BoardPosType,
        y: BoardPosType
    }
    impl BoardPosition {
        pub fn new(x: BoardPosType, y: BoardPosType) -> Self {
            Self {
                x,
                y
            }
        }
        pub fn from_algebraic_notation(text: &str) -> Result<Self, String> {
            if text.len() != 2 {
                return Err("Wring size".to_string());
            }
            let char_vec: Vec<char> = text.chars().collect();
            let x = Self::get_coordinate_from_letter(char_vec[0])?;
            let y = Self::get_coordinate_from_number(char_vec[1])?;
            return Ok(Self::new(x, y));
        }

        pub fn get_coordinate_from_letter(letter: char) -> Result<BoardPosType, String> {
            match letter {
                'a' => return Ok(0),
                'b' => return Ok(1),
                'c' => return Ok(2),
                'd' => return Ok(3),
                'e' => return Ok(4),
                'f' => return Ok(5),
                'g' => return Ok(6),
                'h' => return Ok(7),
                _ => return Err("Could not parse coordinate letter".to_string())
            }
        }

        pub fn get_coordinate_from_number(number: char) -> Result<BoardPosType, String> {
            match number {
                '1' => return Ok(7),
                '2' => return Ok(6),
                '3' => return Ok(5),
                '4' => return Ok(4),
                '5' => return Ok(3),
                '6' => return Ok(2),
                '7' => return Ok(1),
                '8' => return Ok(0),
                _ => return Err("Could not parse coordinate number".to_string())
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct BoardMove {
        pub from_pos: BoardPosition,
        pub to_pos: BoardPosition
    }

    impl BoardMove {
        pub fn new(from_x: BoardPosType, from_y: BoardPosType, to_x: BoardPosType, to_y: BoardPosType) -> Self {
            Self {
                from_pos: BoardPosition::new(from_x, from_y),
                to_pos: BoardPosition::new(to_x, to_y),
            }
        }
    }

    #[allow(dead_code)]
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum ChessPieceId {
        Pawn,
        Knight,
        Rook,
        King,
        Queen,
        Bishop,
    }

    impl ChessPieceId {
        pub fn from_str(text: &str) -> Result<ChessPieceId, ()> {
            let mut text_copy = text.to_string();
            text_copy = text_copy.to_lowercase();
            match text_copy.as_str() {
                "king" => {
                    return Ok(ChessPieceId::King);
                }
                "queen" => {
                    return Ok(ChessPieceId::Queen);
                }
                "rook" => {
                    return Ok(ChessPieceId::Rook);
                }
                "pawn" => {
                    return Ok(ChessPieceId::Pawn);
                }
                "bishop" => {
                    return Ok(ChessPieceId::Bishop);
                }
                "knight" => {
                    return Ok(ChessPieceId::Knight);
                }
                _ => {}
            }
            return Err(());
        }
    }

    #[allow(dead_code)]
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum ChessPieceColor {
        White,
        Black,
    }

    #[allow(dead_code)]
    #[derive(Clone, Copy, Debug)]
    pub struct ChessPiece {
        pub id: ChessPieceId,
        pub color: ChessPieceColor,
        pub unicode_char: char,
        pub moved: bool,
    }

    impl ChessPiece {
        pub fn new(id: ChessPieceId, color: ChessPieceColor) -> Self {
            let unicode_char;
            match id {
                ChessPieceId::Bishop => {
                    unicode_char = '♝';
                }
                ChessPieceId::Rook => {
                    unicode_char = '♜';
                }
                ChessPieceId::King => {
                    unicode_char = '♚';
                }
                ChessPieceId::Queen => {
                    unicode_char = '♛';
                }
                ChessPieceId::Knight => {
                    unicode_char = '♞';
                }
                ChessPieceId::Pawn => {
                    unicode_char = '♟';
                }
            }
            Self {
                id: id,
                color: color,
                unicode_char: unicode_char,
                moved: false,
            }
        }
    }

    #[allow(dead_code)]
    #[derive(Clone, Copy, Debug)]
    pub struct Game {
        board: [Option<ChessPiece>; 8 * 8],
        pub turn: ChessPieceColor,
        last_move: Option<BoardMove>,
        last_move_passant: bool,
        move_count_left: i64, // To make sure player does not play more than 50 moves
    }

    impl Default for Game {
        fn default() -> Self {
            let mut game = Game::new();
            game.set_up_board();
            return game;
        }
    }

    #[allow(dead_code)]
    impl Game {
        pub fn new() -> Self {
            const INIT: Option<ChessPiece> = None;
            Self {
                board: [INIT; 8 * 8],
                turn: ChessPieceColor::White,
                last_move: None,
                last_move_passant: false,
                move_count_left: 100 // Since it is 50 moves per player
            }
        }

        pub fn set_up_board(&mut self) {
            self.empty_board();
            self.set_pos(BoardPosition::new(0, 0), ChessPieceId::Rook, ChessPieceColor::Black);
            self.set_pos(BoardPosition::new(1, 0), ChessPieceId::Knight, ChessPieceColor::Black);
            self.set_pos(BoardPosition::new(2, 0), ChessPieceId::Bishop, ChessPieceColor::Black);
            self.set_pos(BoardPosition::new(3, 0), ChessPieceId::Queen, ChessPieceColor::Black);
            self.set_pos(BoardPosition::new(4, 0), ChessPieceId::King, ChessPieceColor::Black);
            self.set_pos(BoardPosition::new(5, 0), ChessPieceId::Bishop, ChessPieceColor::Black);
            self.set_pos(BoardPosition::new(6, 0), ChessPieceId::Knight, ChessPieceColor::Black);
            self.set_pos(BoardPosition::new(7, 0), ChessPieceId::Rook, ChessPieceColor::Black);
            for x in 0..8 {
                self.set_pos(BoardPosition::new(x, 1), ChessPieceId::Pawn, ChessPieceColor::Black);
            }

            self.set_pos(BoardPosition::new(0, 7), ChessPieceId::Rook, ChessPieceColor::White);
            self.set_pos(BoardPosition::new(1, 7), ChessPieceId::Knight, ChessPieceColor::White);
            self.set_pos(BoardPosition::new(2, 7), ChessPieceId::Bishop, ChessPieceColor::White);
            self.set_pos(BoardPosition::new(3, 7), ChessPieceId::Queen, ChessPieceColor::White);
            self.set_pos(BoardPosition::new(4, 7), ChessPieceId::King, ChessPieceColor::White);
            self.set_pos(BoardPosition::new(5, 7), ChessPieceId::Bishop, ChessPieceColor::White);
            self.set_pos(BoardPosition::new(6, 7), ChessPieceId::Knight, ChessPieceColor::White);
            self.set_pos(BoardPosition::new(7, 7), ChessPieceId::Rook, ChessPieceColor::White);
            for x in 0..8 {
                self.set_pos(BoardPosition::new(x, 6), ChessPieceId::Pawn, ChessPieceColor::White);
            }
        }

        // Move a piece using algebraic notation
        pub fn algebraic_notation_move(&mut self, text: String) -> Result<Option<BoardMove>, String> {
            // Make sure input is not too long
            if text.len() < 2 || text.len() > 10 {
                return Err("Invalid input size".to_string())
            }

            // Remove unnecesary letters
            let text = text.replace(&['(', ')', ',', '\"', '.', ';', 'X', 'x', ':', '='][..], "");

            #[allow(unused_assignments)]
            let mut piece_type: Option<ChessPieceId> = None;
            #[allow(unused_assignments)]
            let mut to_x_input: Option<BoardPosType> = None;
            #[allow(unused_assignments)]
            let mut to_y_input: Option<BoardPosType> = None;
            let mut from_x_input: Option<BoardPosType> = None;
            let mut from_y_input: Option<BoardPosType> = None;
            let mut promote_piece: Option<ChessPieceId> = None;

            let mut char_vec: Vec<char> = text.chars().collect();
            if char_vec.len() < 2 {
                return Err("Could not parse move".to_string());
            }

            // See if player is trying to promote piece
            let promote_piece_result = self.get_piece_type_from_letter(char_vec[char_vec.len() - 1]);
            if promote_piece_result.is_ok() {
                promote_piece = Some(promote_piece_result.unwrap());
                char_vec.resize(char_vec.len() - 1, ' ');
            }

            // See if player specifies a specific piece to move
            let move_piece_result = self.get_piece_type_from_letter(char_vec[0]);
            if move_piece_result.is_ok() {
                piece_type = Some(move_piece_result.unwrap());
                char_vec.remove(0);
            }
            else {
                piece_type = Some(ChessPieceId::Pawn);
            }

            if char_vec.len() < 2 {
                return Err("Could not parse move".to_string());
            }

            if char_vec.len() == 2 {
                to_x_input = Some(BoardPosition::get_coordinate_from_letter(char_vec[0])?);
                to_y_input = Some(BoardPosition::get_coordinate_from_number(char_vec[1])?);
            }
            else if char_vec.len() == 3 {
                let result_letter = BoardPosition::get_coordinate_from_letter(char_vec[0]);
                let result_number = BoardPosition::get_coordinate_from_number(char_vec[0]);
                if result_letter.is_ok() {
                    #[allow(unused_assignments)]
                    {
                        from_x_input = Some(result_letter.unwrap());
                    }
                }
                else if result_number.is_ok() {
                    #[allow(unused_assignments)]
                    {
                        from_y_input = Some(result_number.unwrap());
                    }
                }
                to_x_input = Some(BoardPosition::get_coordinate_from_letter(char_vec[1])?);
                to_y_input = Some(BoardPosition::get_coordinate_from_number(char_vec[2])?);
            }
            else if char_vec.len() == 4 {
                from_x_input = Some(BoardPosition::get_coordinate_from_letter(char_vec[0])?);
                from_y_input = Some(BoardPosition::get_coordinate_from_number(char_vec[1])?);
                to_x_input = Some(BoardPosition::get_coordinate_from_letter(char_vec[2])?);
                to_y_input = Some(BoardPosition::get_coordinate_from_number(char_vec[3])?);
            }
            else {
                return Err("Could not parse move".to_string());
            }

            let mut board_move: Option<BoardMove> = None;


            if from_x_input.is_some()
            && from_y_input.is_some()
            && to_x_input.is_some()
            && to_y_input.is_some() {
                // It is speciefied exactly which piece should move and where
                board_move = Some(BoardMove::new(
                    from_x_input.unwrap(), 
                    from_y_input.unwrap(), 
                    to_x_input.unwrap(), 
                    to_y_input.unwrap()));
            }
            else {
                // Find the moving piece
                // Try to move there, with the limits set by the input
                for from_x in 0..8 {
                    for from_y in 0..8 {
                        // If from position is speciefied, make sure it follows that
                        if from_x_input.is_some() && from_x_input.unwrap() != from_x {
                            continue;
                        }
                        if from_y_input.is_some() && from_y_input.unwrap() != from_y {
                            continue;
                        }
                        // Iterate pieces and see if the piece can move there
                        let from_piece = self.get_board_piece_clone(BoardPosition::new(from_x, from_y));
                        if from_piece.is_some() 
                        && from_piece.unwrap().color == self.turn 
                        && from_piece.unwrap().id == piece_type.unwrap() {
                            // Make a copy of the board and try to move there
                            let mut board_copy = self.clone();
                            let test_move = BoardMove::new(from_x, from_y, to_x_input.unwrap(), to_y_input.unwrap());
                            if board_copy.move_piece(test_move, true, promote_piece).is_ok() {
                                // Make sure there are not multiple pieces that can do that move
                                if board_move.is_some() {
                                    return Err("Unclear which piece is to move".to_string());
                                }
                                board_move = Some(test_move);
                            }
    
                        }
                    }
                }
            }
            
            if board_move.is_none() {
                return Err("Could not do move".to_string());
            }
            if !self.will_require_promotion(board_move.unwrap()) && promote_piece.is_some() {
                return Err("Cannot promote piece".to_string());
            }
            self.move_piece(board_move.unwrap(), true, promote_piece)?;
            return Ok(None)
        }

        // Set all postitions on the board to none
        pub fn empty_board(&mut self) {
            for i in 0..self.board.len() {
                self.board[i] = None;
            }
        }

        pub fn print_board(&mut self) {
            self.print_board_with_possible_moves(None);
        }

        pub fn print_board_with_possible_moves(&mut self, possible_moves_from_pos: Option<BoardPosition>) {
            println!("  a b c d e f g h");
            for y in 0..8 {
                print!("{} ", 8-y);
                for x in 0..8 {
                    let board_move;
                    if possible_moves_from_pos.is_some() {
                        board_move = BoardMove::new(possible_moves_from_pos.unwrap().x, possible_moves_from_pos.unwrap().y, x, y);
                    }
                    else {
                        board_move = BoardMove::new(0, 0, 0, 0);
                    }

                    let mut board_copy = self.clone();
                    let background_color;
                    if possible_moves_from_pos.is_some() && board_copy.move_piece(board_move, true, Some(ChessPieceId::Queen)).is_ok() {
                        // Color square red if piece can move there
                        background_color = ColorTerminal::Red;
                    }
                    else if possible_moves_from_pos.is_some() && board_move.from_pos.x == x && board_move.from_pos.y == y {
                        background_color = ColorTerminal::Green;
                    }
                    else if (x + y) % 2 == 0 {
                        background_color = ColorTerminal::LightBlue;
                    }
                    else {
                        background_color = ColorTerminal::Blue;
                    }
                    // Get a reference to a position on the board
                    let board_ref = *self.get_board_ref(BoardPosition::new(x, y)).unwrap();
                    if board_ref.is_none() {
                        // If there is nothing there, just print two spaces
                        print_color("  ", ColorTerminal::White, background_color);
                    } else {
                        if board_ref.as_ref().unwrap().color == ChessPieceColor::Black {
                            print_color(board_ref.as_ref().unwrap().unicode_char.to_string().as_str(), ColorTerminal::Black, background_color);
                            print_color(" ", ColorTerminal::Black, background_color);
                        }
                        else {
                            print_color(board_ref.as_ref().unwrap().unicode_char.to_string().as_str(), ColorTerminal::White, background_color);
                            print_color(" ", ColorTerminal::White, background_color);
                        }
                    }
                }
                print!("{} ", 8-y);
                println!();
            }
            println!("  a b c d e f g h");
        }

        // Get a reference to a coordinate on the board
        pub fn get_board_ref(&mut self, pos: BoardPosition) -> Result<&mut Option<ChessPiece>, String> {
            if self.inside_board(pos).is_err() {
                return Err("Cannot acces pieces outside board".to_string());
            }
            return Ok(&mut self.board[(pos.x + 8 * pos.y) as usize]);
        }

        // Get a clone of a position on the board
        pub fn get_board_piece_clone(&mut self, pos: BoardPosition) -> Option<ChessPiece> {
            return (*self.get_board_ref(pos).unwrap()).clone();
        }

        // Set a position on the board
        pub fn set_pos(
            &mut self,
            pos: BoardPosition,
            id: ChessPieceId,
            color: ChessPieceColor,
        ) {
            *self.get_board_ref(pos).unwrap() = Some(ChessPiece::new(id, color));
        }

        pub fn set_pos_to_none(
            &mut self,
            pos: BoardPosition,
        ) {
            *self.get_board_ref(pos).unwrap() = None;
        }

        // Returns true if the game is over
        pub fn game_is_over(&mut self) -> bool {
            if self.is_check_mate() || self.is_stale_mate() || self.max_move_count_reached() {
                return true;
            }
            return false;
        }

        // Returns if there is a winner, and what color it is
        pub fn get_winner(&mut self) -> Option<ChessPieceColor> {
            if self.is_check_mate() {
                if self.turn == ChessPieceColor::White {
                    return Some(ChessPieceColor::Black);
                }
                else {
                    return Some(ChessPieceColor::White);
                }
            }
            else {
                return None;
            }
        }

        pub fn is_check(&mut self) -> Option<BoardMove> {
            for x in 0..8 {
                for y in 0..8 {
                    let piece = self.get_board_piece_clone(BoardPosition::new(x, y));
                    // If it is a king
                    if piece.is_some() && piece.unwrap().id == ChessPieceId::King && piece.unwrap().color == self.turn
                    {
                        // Iterate pieces to see if any can capture the king
                        for x2 in 0..8 {
                            for y2 in 0..8 {
                                let board_move = BoardMove::new(x2, y2, x, y);
                                let mut board_copy = (*self).clone();
                                board_copy.end_turn(); // Make it the opponents turn

                                // Checks if the move captures the king
                                // If it takes the king, you dont have to think about check
                                if board_copy.move_piece(board_move, false, Some(ChessPieceId::Queen)).is_ok() {
                                    return Some(board_move);
                                }
                            }
                        }
                    }
                }
            }
            return None;
        }

        pub fn is_check_mate(&mut self) -> bool{
            if self.is_check().is_some() && self.get_possible_moves().is_empty() {
                return true;
            }
            else {
                return false;
            }
        }

        pub fn is_stale_mate(&mut self) -> bool{
            if self.is_check().is_none() && self.get_possible_moves().is_empty() {
                return true;
            }
            else {
                return false;
            }
        }

        pub fn get_possible_moves(&mut self) -> LinkedList<BoardMove> {
            let mut board_moves:LinkedList<BoardMove> = Default::default();
            for x in 0..8 {
                for y in 0..8 {
                    let piece = (self.get_board_ref(BoardPosition::new(x, y)).unwrap()).clone();
                    // If it is a king
                    if piece.is_some() && piece.unwrap().color == self.turn
                    {
                        for x2 in 0..8 {
                            for y2 in 0..8 {
                                // Iterate spaces to see if it can move there
                                let board_move = BoardMove::new(x, y, x2, y2);
                                let mut board_copy = self.clone();
                                if board_copy.move_piece(board_move, true, Some(ChessPieceId::Queen)).is_ok() {
                                    board_moves.push_back(board_move);
                                }
                            }
                        }
                    }
                }
            }
            return board_moves;
        }

        pub fn move_piece(&mut self, board_move: BoardMove, check_for_check: bool, promote_piece: Option<ChessPieceId>) -> Result<(), String> {
            self.is_move(board_move)?;
            self.inside_board(board_move.from_pos)?;
            self.inside_board(board_move.to_pos)?;
            let from_piece = self
                .get_board_ref(board_move.from_pos).unwrap()
                .clone();
            if from_piece.is_none() {
                return Err("No piece on square selected!".to_string());
            }
            self.board_move_not_same_color_pieces(board_move)?;

            // Make sure move does not lead to check
            if check_for_check {
                let mut self_copy = self.clone();
                self_copy.move_piece(board_move, false, Some(ChessPieceId::Queen))?;
                self_copy.turn = self.turn;
                if self_copy.is_check().is_some() {
                    return Err("Move leads to check!".to_string());
                }
            }

            // make sure you are not moving opponents pieces
            if from_piece.as_ref().unwrap().color != self.turn {
                return Err("Cannot move opponents pieces".to_string());
            }

            // make sure move count limit is not reached
            if self.max_move_count_reached() {
                return Err("Maximum move count reached".to_string());
            }

            // Do move depending on piece
            match from_piece.as_ref().unwrap().id {
                ChessPieceId::Bishop => {
                    self.bishop_move(board_move)?;
                }
                ChessPieceId::Rook => {
                    self.rook_move(board_move)?;
                }
                ChessPieceId::King => {
                    let result1 = self.king_move_one(board_move);
                    let result2 = self.king_castle(board_move);
                    if result1.is_err() && result2.is_err() {
                        return Err(
                            (result1.err().unwrap() + result2.err().unwrap().as_str()).to_string()
                        );
                    }
                }
                ChessPieceId::Queen => {
                    self.queen_move(board_move)?;
                }
                ChessPieceId::Knight => {
                    self.knight_move(board_move)?;
                }
                ChessPieceId::Pawn => {
                    let mut promote = false;
                    if self.will_require_promotion(board_move) {
                        if promote_piece.is_some() && promote_piece.unwrap() != ChessPieceId::Pawn && promote_piece.unwrap() != ChessPieceId::King {
                            promote = true;
                        }
                        else {
                            return Err("Move requires promotion, cannot promote to pawn or king".to_string());
                        }
                    }
                    loop {
                        let result1 = self.pawn_one_forward(board_move);
                        if result1.is_ok() {
                            break;
                        }
                        let result2 = self.pawn_two_forward(board_move);
                        if result2.is_ok() {
                            break;
                        }
                        let result3 = self.pawn_take(board_move);
                        if result3.is_ok() {
                            break;
                        }
                        let mut error_message: String = "Moving pawn failed!\n one_forward: ".to_string();
                        error_message += result1.err().unwrap().as_str();
                        error_message += "\n two_forward: ";
                        error_message += result2.err().unwrap().as_str();
                        error_message += "\n take: ";
                        error_message += result3.err().unwrap().as_str();
                        error_message += "\n";
                        return Err(error_message)
                    }
                    if promote {
                        self.get_board_ref(board_move.to_pos).unwrap().as_mut().unwrap().id = promote_piece.unwrap();
                    }
                    // Reset move count after succesfull move with pawn
                    self.reset_move_count_left();
                }
            }
            self.end_turn();
            return Ok(());
        }

        fn promote(
            &mut self,
            pos: BoardPosition,
            to_id: ChessPieceId,
        ) -> Result<(), String> {
            self.inside_board(pos)?;
            if to_id == ChessPieceId::Pawn {
                return Err("Cannot convert to pawn!".to_string());
            }
            if to_id as u32 == ChessPieceId::King as u32 {
                return Err("Cannot convert to king!".to_string());
            }
            let piece = (*self.get_board_ref(pos).unwrap()).clone();
            if piece.is_none() {
                return Err("Cannot convert from nothing!".to_string());
            }
            if piece.unwrap().id != ChessPieceId::Pawn {
                return Err("Cannot convert something other than pawn!".to_string());
            }
            if piece.unwrap().color != self.turn {
                return Err("Cannot convert opponents pieces!".to_string());
            }
            // Make sure piece is in the right place
            if (piece.unwrap().color == ChessPieceColor::Black && pos.y == 7)
                || (piece.unwrap().color as u32 == ChessPieceColor::White as u32 && pos.y == 0)
            {
                // Convert piece
                (*self.get_board_ref(pos).unwrap()).as_mut().unwrap().id = to_id;
                self.end_turn();
                return Ok(());
            } else {
                return Err("Cannot convert pawn not on other side".to_string());
            }
        }
        pub fn get_piece_type_from_letter(&mut self, letter: char) -> Result<ChessPieceId, String> {
            match letter {
                'R' => return Ok(ChessPieceId::Rook),
                'P' => return Ok(ChessPieceId::Pawn),
                'Q' => return Ok(ChessPieceId::Queen),
                'K' => return Ok(ChessPieceId::King),
                'N' => return Ok(ChessPieceId::Knight),
                'B' => return Ok(ChessPieceId::Bishop),
                _ => return Err("No matching type".to_string())
            }
        }

        fn is_unblocked_straight_line(&mut self, board_move: BoardMove) -> Result<(), String> {
            // Make sure it is a straight line
            if board_move.from_pos.x == board_move.to_pos.x {
                for y in 1..(board_move.from_pos.y as i32 - board_move.to_pos.y as i32).abs() {
                    let mut d_y = y;
                    if board_move.to_pos.y < board_move.from_pos.y {
                        d_y = -d_y;
                    }
                    if self
                        .get_board_ref(BoardPosition::new(board_move.from_pos.x, (d_y + board_move.from_pos.y as i32) as u8))
                        .unwrap()
                        .as_ref()
                        .is_some()
                    {
                        return Err("Straight path is blocked".to_string());
                    }
                }
                return Ok(());
            } else if board_move.from_pos.y == board_move.to_pos.y {
                for x in 1..(board_move.from_pos.x as i32 - board_move.to_pos.x as i32).abs() {
                    let mut d_x = x;
                    if board_move.to_pos.x < board_move.from_pos.x {
                        d_x = -d_x;
                    }
                    if self
                        .get_board_ref(BoardPosition::new((board_move.from_pos.x as i32 + d_x) as u8, board_move.from_pos.y))
                        .unwrap()
                        .as_ref()
                        .is_some()
                    {
                        return Err("Straight path is blocked!".to_string());
                    }
                }
                return Ok(());
            }
            return Err("Not a valid straight line!".to_string());
        }

        fn reset_move_count_left(&mut self) {
            self.move_count_left = 100;
        }
        fn max_move_count_reached(&mut self) -> bool {
            return self.move_count_left < 1;
        }

        fn is_unblocked_diagonal_line(&mut self, board_move: BoardMove) -> Result<(), String> {
            // Make sure it is a diagonal
            if (board_move.from_pos.x as i32 - board_move.to_pos.x as i32).abs()
                == (board_move.from_pos.y as i32 - board_move.to_pos.y as i32).abs()
            {
                // Move along diagonal to make sure path is not blocked
                for i in 1..(board_move.from_pos.x as i32 - board_move.to_pos.x as i32).abs() {
                    let mut d_x = i as i32;
                    let mut d_y = i as i32;
                    if board_move.to_pos.x < board_move.from_pos.x {
                        d_x = -d_x;
                    }
                    if board_move.to_pos.y < board_move.from_pos.y {
                        d_y = -d_y;
                    }
                    let pos = BoardPosition::new((d_x + board_move.from_pos.x as i32) as u8, 
                    (d_y + board_move.from_pos.y as i32) as u8);
                    if self
                        .get_board_ref(pos)
                        .unwrap()
                        .as_ref()
                        .is_some()
                    {
                        return Err("Path blocked!".to_string());
                    }
                }
                return Ok(());
            }
            return Err("Not a valid diagonal!".to_string());
        }

        fn is_move(&mut self, board_move: BoardMove) -> Result<(), String> {
            if (board_move.from_pos.x == board_move.to_pos.x) && (board_move.from_pos.y == board_move.to_pos.y) {
                return Err("Cannot do nothing during your turn!".to_string());
            }
            return Ok(());
        }

        fn is_pieces_same_color(&mut self, pos1: BoardPosition, pos2: BoardPosition) -> bool {
            let piece1 = self.get_board_piece_clone(pos1);
            let piece2 = self.get_board_piece_clone(pos2);
            if piece1.is_some() && piece2.is_some() && piece1.unwrap().color == piece2.unwrap().color {
                return true;
            }
            else {
                return false;
            }
        }

        fn is_piece_id(&mut self, pos: BoardPosition, id: ChessPieceId) -> Result<(), String> {
            if self.get_board_ref(pos)?.is_some() && self.get_board_ref(pos)?.unwrap().id == id {
                return Ok(());
            }
            else {
                return Err("Piece id does not match".to_string());
            }
        }
        fn inside_board(&mut self, pos: BoardPosition) -> Result<(), String> {
            if pos.x > 7 || pos.y > 7 {
                return Err("Outside of the board!".to_string());
            }
            return Ok(());
        }

        fn end_turn(&mut self) {
            if self.turn as u32 == ChessPieceColor::White as u32 {
                self.turn = ChessPieceColor::Black;
            } else {
                self.turn = ChessPieceColor::White;
            }
            self.move_count_left = self.move_count_left - 1;
        }

        fn will_require_promotion(&mut self, board_move: BoardMove) -> bool {
            let piece = self.get_board_piece_clone(board_move.from_pos);
            if piece.is_none() {
                return false;
            }
            if piece.unwrap().id != ChessPieceId::Pawn {
                return false;
            }
            if piece.unwrap().color != self.turn {
                return false;
            }
            // Make sure piece is in the right place
            if (piece.unwrap().color == ChessPieceColor::Black && board_move.to_pos.y == 7)
                || (piece.unwrap().color as u32 == ChessPieceColor::White as u32 && board_move.to_pos.y == 0)
            {
                return true;
            } else {
                return false;
            }
        }

        // Just moves the piece without any checking
        fn force_move_piece(&mut self, board_move: BoardMove) {
            if self.get_board_ref(board_move.to_pos).unwrap().is_some(){
                // Reset move count after capture
                self.reset_move_count_left();
            }
            *self.get_board_ref(board_move.to_pos).unwrap() = self
                .get_board_ref(board_move.from_pos)
                .unwrap()
                .take();
            if self.get_board_ref(board_move.to_pos).unwrap().is_some() {
                self.get_board_ref(board_move.to_pos).unwrap().as_mut().unwrap().moved = true;
            }
            self.last_move_passant = false;
            self.last_move = Some(board_move);
        }
        fn board_move_not_same_color_pieces(&mut self, board_move: BoardMove) -> Result<(), String> {
            if self.is_pieces_same_color(board_move.from_pos, board_move.to_pos) {
                return Err("Cannot move piece to piece of same color".to_string())
            }
            else {
                return Ok(());
            }
        }
        fn board_move_is_forward(&mut self, board_move: BoardMove) -> Result<(), String> {
            let piece = self.get_board_ref(board_move.from_pos).unwrap();
            if piece.is_none() {
                return Err("Piece is none".to_string());
            }
            match piece.unwrap().color {
                ChessPieceColor::Black => {
                    if board_move.from_pos.y < board_move.to_pos.y {
                        return Ok(());
                    }
                    else {
                        return Err("Not forward move".to_string());
                    }
                }
                ChessPieceColor::White => {
                    if board_move.from_pos.y > board_move.to_pos.y {
                        return Ok(());
                    }
                    else {
                        return Err("Not forward move".to_string());
                    }
                }
            }
        }

        fn pawn_one_forward(&mut self, board_move: BoardMove) -> Result<(), String> {
            self.is_piece_id(board_move.from_pos, ChessPieceId::Pawn)?;
            self.board_move_is_forward(board_move)?;
            if (board_move.to_pos.y as i32 - board_move.from_pos.y as i32).abs() != 1 {
                return Err("Not moving one forward".to_string());
            }
            if (board_move.to_pos.x as i32 - board_move.from_pos.x as i32).abs() != 0 {
                return Err("Moving to side".to_string());
            }
            if self
                .get_board_ref(board_move.to_pos)
                .unwrap()
                .is_some()
            {
                return Err("Path blocked".to_string());
            }

            self.force_move_piece(board_move);
            return Ok(());
        }

        fn pawn_two_forward(&mut self, board_move: BoardMove) -> Result<(), String> {
            self.is_piece_id(board_move.from_pos, ChessPieceId::Pawn)?;
            self.board_move_is_forward(board_move)?;
            self.is_unblocked_straight_line(board_move)?;

            // Make sure piece is moving two forward
            if (board_move.to_pos.y as i32 - board_move.from_pos.y as i32).abs() != 2 {
                return Err("Not moving two forward".to_string());
            }
            if (board_move.to_pos.x as i32 - board_move.from_pos.x as i32).abs() != 0 {
                return Err("Moving to side".to_string());
            }

            if self
                .get_board_ref(board_move.to_pos)
                .unwrap()
                .is_some()
            {
                return Err("Path blocked".to_string());
            }

            if self.get_board_ref(board_move.from_pos).unwrap().unwrap().moved == true {
                return Err("Cannot move pawn two forward who has already moved".to_string());
            }

            // Move piece
            self.force_move_piece(board_move);
            self.last_move_passant = true;
            
            return Ok(());
        }

        fn pawn_take(&mut self, board_move: BoardMove) -> Result<(), String> {
            self.is_piece_id(board_move.from_pos, ChessPieceId::Pawn)?;
            self.board_move_not_same_color_pieces(board_move)?;
            self.board_move_is_forward(board_move)?;

            // Make sure pawn is moving one forward
            if (board_move.to_pos.y as i32 - board_move.from_pos.y as i32).abs() != 1 {
                return Err("Not moving one forward".to_string());
            }

            // Make sure pawn is moving one to the side
            if (board_move.to_pos.x as i32 - board_move.from_pos.x as i32).abs() != 1 {
                return Err("Not moving one to the side".to_string());
            }

            // Check for pessant
            if self.last_move_passant == true 
                && self.last_move.is_some() 
                && (board_move.to_pos.x as i32 - self.last_move.unwrap().from_pos.x as i32) == 0 
                && (board_move.to_pos.y as i32 - self.last_move.unwrap().from_pos.y as i32).abs() == 1 
                && (board_move.to_pos.y as i32 - self.last_move.unwrap().to_pos.y as i32).abs() == 1 {
                // Remove last moved pawn due to pessant
                *self.get_board_ref(self.last_move.unwrap().to_pos).unwrap() =
                None;
            }
            else if self
                .get_board_ref(board_move.to_pos)
                .unwrap()
                .is_none() {
                return Err("No piece to take".to_string());
            }

            // Move piece
            self.force_move_piece(board_move);
            return Ok(());
        }

        fn rook_move(&mut self, board_move: BoardMove) -> Result<(), String> {
            self.is_piece_id(board_move.from_pos, ChessPieceId::Rook)?;
            self.is_unblocked_straight_line(board_move)?;
            self.board_move_not_same_color_pieces(board_move)?;

            self.force_move_piece(board_move);
            return Ok(());
        }

        fn knight_move(&mut self, board_move: BoardMove) -> Result<(), String> {
            self.is_piece_id(board_move.from_pos, ChessPieceId::Knight)?;
            self.board_move_not_same_color_pieces(board_move)?;

            if ((board_move.from_pos.x as i32 - board_move.to_pos.x as i32).abs() != 1
            || (board_move.from_pos.y as i32 - board_move.to_pos.y as i32).abs() != 2)
                && ((board_move.from_pos.x as i32 - board_move.to_pos.x as i32).abs() != 2
                || (board_move.from_pos.y as i32 - board_move.to_pos.y as i32).abs() != 1) {
                return Err("invalid move".to_string());
            }
            self.force_move_piece(board_move);
            return Ok(());
        }

        fn queen_move(&mut self, board_move: BoardMove) -> Result<(), String> {
            self.is_piece_id(board_move.from_pos, ChessPieceId::Queen)?;
            self.board_move_not_same_color_pieces(board_move)?;
            let diagonal_result = self.is_unblocked_diagonal_line(board_move);
            let straight_result = self.is_unblocked_straight_line(board_move);
            if diagonal_result.is_err() && straight_result.is_err() {
                let mut error_message: String = "Queen: ".to_string();
                error_message += diagonal_result.err().unwrap().as_str();
                error_message += ", ";
                error_message += straight_result.err().unwrap().as_str();
                return Err(error_message);
            }
            self.force_move_piece(board_move);
            return Ok(());
        }

        fn king_move_one(&mut self, board_move: BoardMove) -> Result<(), String> {
            self.is_piece_id(board_move.from_pos, ChessPieceId::King)?;
            self.board_move_not_same_color_pieces(board_move)?;
            if (board_move.from_pos.x as i32 - board_move.to_pos.x as i32).abs() > 1
                || (board_move.from_pos.y as i32 - board_move.to_pos.y as i32).abs() > 1
            {
                return Err("King: cannot move that far!".to_string());
            }
            self.force_move_piece(board_move);
            return Ok(());
        }

        fn king_castle(&mut self, board_move: BoardMove) -> Result<(), String> {
            let from_piece = self.get_board_piece_clone(board_move.from_pos);
            if from_piece.is_none() || (from_piece.unwrap().id != ChessPieceId::King) {
                return Err("Must have king to castle".to_string());
            }
            if from_piece.unwrap().moved == true {
                return Err("Cannot castle with moved piece".to_string());
            }
            if (board_move.to_pos.x as i32- board_move.from_pos.x as i32).abs() != 2 {
                return Err("Castle requires piece to move two to the side".to_string());
            }
            if (board_move.to_pos.y as i32- board_move.from_pos.y as i32).abs() != 0 {
                return Err("Cannot move forwards while casteling".to_string());
            }
            // Find the rook
            let direction: i32;
            if board_move.to_pos.x > board_move.from_pos.x {
                direction = 1;
            }
            else {
                direction = -1;
            }
            // Make sure it is not check on any of the squares king is moving on
            for i in 0..3 {
                // Make a clone of the board
                let mut board_copy = self.clone();

                // Move the king and check for check
                board_copy.set_pos_to_none(board_move.from_pos);
                let pos = BoardPosition::new(
                    (board_move.from_pos.x as i32+i*direction) as BoardPosType, 
                    board_move.to_pos.y);
                board_copy.set_pos(pos, ChessPieceId::King, self.turn);
                if board_copy.is_check().is_some() {
                    return Err("Cannot castle on checked square".to_string());
                }

            }
            for i in 1..5 {
                let rook_x = (board_move.from_pos.x as i32+i*direction) as BoardPosType;
                let rook_y = board_move.from_pos.y;
                let rook_pos = BoardPosition::new(rook_x, rook_y);
                if self.inside_board(rook_pos).is_err() {
                    break;
                }
                if self.is_piece_id(BoardPosition::new(rook_x, rook_y), ChessPieceId::Rook).is_ok() {
                    if self.get_board_ref(BoardPosition::new(rook_x, rook_y)).unwrap().unwrap().moved == true {
                        return Err("Cannot castle with moved rook".to_string());
                    }
                    
                    // Move king and rook
                    self.force_move_piece(board_move);
                    let rook_move = BoardMove::new(rook_x, rook_y, 
                        (board_move.from_pos.x as i32 + direction) as BoardPosType, rook_y);
                    self.force_move_piece(rook_move);
                    return Ok(());
                }
            }
            return Err("No rook to move with".to_string());
        }

        fn bishop_move(&mut self, board_move: BoardMove) -> Result<(), String> {
            self.is_unblocked_diagonal_line(board_move)?;
            self.force_move_piece(board_move);
            return Ok(());
        }
    }
}