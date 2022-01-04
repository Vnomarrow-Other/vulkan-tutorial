pub extern crate chess_engine;
use std::{
    io::{self, BufRead},
};

fn print_game_info(game: &mut chess_engine::chess_game::Game) {
    game.print_board();
    if game.game_is_over() {
        let winner = game.get_winner();
        println!("Game is over!");
        if winner == None {
            println!("Draw");
        }
        else if winner.unwrap() == ChessPieceColor::White {
            println!("White wins!");
        }
        else if winner.unwrap() == ChessPieceColor::Black {
            println!("Black wins!");
        }
        return;
    }
    use chess_engine::chess_game::*;
    if game.turn as u32 == ChessPieceColor::Black as u32 {
        println!("Black's turn");
    } else {
        println!("White's turn");
    }
    let check = game.is_check();
    if check.is_some() {
        println!("Check!");
    }
    println!();
    println!("Algebraic notation, ex Na3 to move knight to a3");
    println!("Algebraic notation, specifying move exactly ex. Nb1b3 will show debug messages");
    println!("Or, print possible moves: ex 'moves a2' to print all moves for a2");
}

fn main() {
    let mut game = chess_engine::chess_game::Game::new();
    game.set_up_board();
    loop {
        let stdin = io::stdin();
        print_game_info(&mut game);
        for line in stdin.lock().lines().map(|l| l.unwrap()) {
            let user_input: Vec<String> =
                line.split_whitespace().map(|num| num.to_string()).collect();
            if user_input.len() == 1 {
                // Treat it as a algebraic move
                let result = game.algebraic_notation_move(user_input[0].clone());
                if result.is_ok() {
                    println!("Move Succesfull!");
                } else {
                    println!("Move Failed!");
                    let error_message = result.err().unwrap();
                    println!("{}", error_message);
                }
                print_game_info(&mut game);
            }
            else if user_input.len() == 2 {
                // Print all possible moves for square
                if user_input[0] == "moves" {
                    let position = user_input[1].clone();
                    if position.len() != 2 {
                        println!("Invalid input");
                    }
                    let char_vec: Vec<char> = position.chars().collect();
                    let result_letter = chess_engine::chess_game::BoardPosition::get_coordinate_from_letter(char_vec[0]);
                    let result_number = chess_engine::chess_game::BoardPosition::get_coordinate_from_number(char_vec[1]);
                    if result_letter.is_ok() && result_number.is_ok() {
                        let pos = chess_engine::chess_game::BoardPosition::new(result_letter.unwrap(), result_number.unwrap());
                        game.print_board_with_possible_moves(Some(pos));
                    }
                    else {
                        println!("Invalid input");
                    }
                }
                else {
                    println!("Invalid input");
                }
            } else {
                println!("Invalid input");
            }
        }
    }
}
