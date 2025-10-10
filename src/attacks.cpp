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

/*/////////////////////////////////////////////////////////////////////////////
                                pawn attacks
/*/////////////////////////////////////////////////////////////////////////////

// index: side, square
U64 pawn_attacks[2][64];

void init_pawn_attacks() {
    for (int sq=0; sq<64; sq++) {
        pawn_attacks[WHITE][sq] = 0;
        pawn_attacks[BLACK][sq] = 0;
        U64 sq_bitmask = sq_bit[sq];

        if ((sq_bitmask & rank_8) || (sq_bitmask & rank_1)) { continue; }

        if (sq_bitmask & not_a_file) {
            pawn_attacks[WHITE][sq] |= sq_bitmask >> 9;
            pawn_attacks[BLACK][sq] |= sq_bitmask << 7;
        }

        if (sq_bitmask & not_h_file) {
            pawn_attacks[WHITE][sq] |= sq_bitmask >> 7;
            pawn_attacks[BLACK][sq] |= sq_bitmask << 9;
        }
    }
}

/*/////////////////////////////////////////////////////////////////////////////
                                knight attacks
/*/////////////////////////////////////////////////////////////////////////////

// index: square
U64 knight_attacks[64];

void init_knight_attacks() {
    for (int sq=0; sq<64; sq++) {
        knight_attacks[sq] = 0;
        U64 sq_bitmask = sq_bit[sq];

        if (sq_bitmask & not_rank_8) {
            if (sq_bitmask & not_ab_file) { knight_attacks[sq] |= sq_bitmask >> 10; }
            if (sq_bitmask & not_gh_file) { knight_attacks[sq] |= sq_bitmask >> 6; }
            if (sq_bitmask & not_rank_7) {
                if (sq_bitmask & not_a_file) { knight_attacks[sq] |= sq_bitmask >> 17; }
                if (sq_bitmask & not_h_file) { knight_attacks[sq] |= sq_bitmask >> 15; }
            }
        }

        if (sq_bitmask & not_rank_1) {
            if (sq_bitmask & not_ab_file) { knight_attacks[sq] |= sq_bitmask << 6; }
            if (sq_bitmask & not_gh_file) { knight_attacks[sq] |= sq_bitmask << 10; }
            if (sq_bitmask & not_rank_2) {
                if (sq_bitmask & not_a_file) { knight_attacks[sq] |= sq_bitmask << 15; }
                if (sq_bitmask & not_h_file) { knight_attacks[sq] |= sq_bitmask << 17; }
            }
        }
    }
}

/*/////////////////////////////////////////////////////////////////////////////
                                king attacks
/*/////////////////////////////////////////////////////////////////////////////

// index: square
U64 king_attacks[64];

void init_king_attacks() {
    for (int sq=0; sq<64; sq++) {
        king_attacks[sq] = 0;
        U64 sq_bitmask = sq_bit[sq];

        if (sq_bitmask & not_rank_8) {
            king_attacks[sq] |= sq_bitmask >> 8;
            if (sq_bitmask & not_a_file) { king_attacks[sq] |= sq_bitmask >> 9; }
            if (sq_bitmask & not_h_file) { king_attacks[sq] |= sq_bitmask >> 7; }
        }

        if (sq_bitmask & not_a_file) { king_attacks[sq] |= sq_bitmask >> 1; }
        if (sq_bitmask & not_h_file) { king_attacks[sq] |= sq_bitmask << 1; }

        if (sq_bitmask & not_rank_1) {
            king_attacks[sq] |= sq_bitmask << 8;
            if (sq_bitmask & not_a_file) { king_attacks[sq] |= sq_bitmask << 7; }
            if (sq_bitmask & not_h_file) { king_attacks[sq] |= sq_bitmask << 9; }
        }
    }
}

