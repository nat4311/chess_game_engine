#pragma once

#include "engine.h"

const U64 rank_1 = A1|B1|C1|D1|E1|F1|G1|H1;
const U64 rank_2 = A2|B2|C2|D2|E2|F2|G2|H2;
const U64 rank_7 = A7|B7|C7|D7|E7|F7|G7|H7;
const U64 rank_8 = A8|B8|C8|D8|E8|F8|G8|H8;
const U64 not_rank_1 = ~rank_1;
const U64 not_rank_2 = ~rank_2;
const U64 not_rank_7 = ~rank_7;
const U64 not_rank_8 = ~rank_8;
const U64 not_a_file = ~(A1|A2|A3|A4|A5|A6|A7|A8);
const U64 not_ab_file = ~(~not_a_file|B1|B2|B3|B4|B5|B6|B7|B8);
const U64 not_h_file = ~(H1|H2|H3|H4|H5|H6|H7|H8);
const U64 not_gh_file = ~(~not_h_file|G1|G2|G3|G4|G5|G6|G7|G8);

void init_attacks();

// side is WHITE or BLACK (from enum), square is 0-63
U64 get_pawn_attacks(int side, int square);

// square is 0-63
U64 get_knight_attacks(int square);

// square is 0-63
U64 get_king_attacks(int square);

// square is 0-63, occupancy is for both sides
U64 get_bishop_attacks(int square, U64 occupancy);

// square is 0-63, occupancy is for both sides
U64 get_rook_attacks(int square, U64 occupancy);

// square is 0-63, occupancy is for both sides
U64 get_queen_attacks(int square, U64 occupancy);
