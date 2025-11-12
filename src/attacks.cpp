/* references:
 * https://www.chessprogramming.org/Looking_for_Magics
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "engine.h"
#include "attacks.h"
#include "random_numbers.h"
#include <assert.h>
#include <unistd.h>
#include <string.h>

/*///////////////////////////////////////////////////////////////////////////////
                              Section: pawn attacks
/*///////////////////////////////////////////////////////////////////////////////

// index: side, square
static U64 pawn_attacks[2][64];

U64 get_pawn_attacks(int side, int square) {
    return pawn_attacks[side][square];
}

static void init_pawn_attacks() {
    for (int sq=0; sq<64; sq++) {
        pawn_attacks[WHITE][sq] = 0ULL;
        pawn_attacks[BLACK][sq] = 0ULL;
        U64 sq_bitmask = sq_bit[sq];

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

/*///////////////////////////////////////////////////////////////////////////////
                              Section: knight attacks
/*///////////////////////////////////////////////////////////////////////////////

// index: square
static U64 knight_attacks[64];

U64 get_knight_attacks(int square) {
    return knight_attacks[square];
}

static void init_knight_attacks() {
    for (int sq=0; sq<64; sq++) {
        knight_attacks[sq] = 0ULL;
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

/*///////////////////////////////////////////////////////////////////////////////
                              Section: king attacks
/*///////////////////////////////////////////////////////////////////////////////

// index: square
static U64 king_attacks[64];

U64 get_king_attacks(int square) {
    return king_attacks[square];
}

static void init_king_attacks() {
    for (int sq=0; sq<64; sq++) {
        king_attacks[sq] = 0ULL;
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

/*///////////////////////////////////////////////////////////////////////////////
                              Section: bishop attacks
/*///////////////////////////////////////////////////////////////////////////////

static int bishop_n_relevant_occupancies[64] = {
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6,
};
constexpr int bishop_max_relevant_occupancies = 9;
constexpr int bishop_magic_indices = 1<<bishop_max_relevant_occupancies;

// index: square
static U64 bishop_relevant_bits_masks[64];
// indices: square, magic_index
static U64 bishop_attack_tables[64][bishop_magic_indices];
// index: square
static U64 bishop_magic_numbers[64];

U64 get_bishop_attacks(int square, U64 occupancy_both) {
    U64 magic_index = ((occupancy_both & bishop_relevant_bits_masks[square]) * bishop_magic_numbers[square]) >> (64-bishop_max_relevant_occupancies);
    return bishop_attack_tables[square][magic_index];
}

static void init_bishop_masks() {
    for (int sq=0; sq<64; sq++) {
        int x0 = sq % 8;
        int y0 = sq / 8;
        int x, y;
        for (x=x0+1, y=y0+1; x<7 && y<7; x++, y++) { bishop_relevant_bits_masks[sq] |= sq_bit[x + y*8]; }
        for (x=x0+1, y=y0-1; x<7 && y>0; x++, y--) { bishop_relevant_bits_masks[sq] |= sq_bit[x + y*8]; }
        for (x=x0-1, y=y0+1; x>0 && y<7; x--, y++) { bishop_relevant_bits_masks[sq] |= sq_bit[x + y*8]; }
        for (x=x0-1, y=y0-1; x>0 && y>0; x--, y--) { bishop_relevant_bits_masks[sq] |= sq_bit[x + y*8]; }
    }
}

static U64 get_bishop_attacks_slow(int square, U64 occupancy_both) {
    U64 attacks = 0ULL;
    int x0 = square % 8;
    int y0 = square / 8;
    int x, y, sq;

    for (x=x0+1, y=y0+1; x<=7 && y<=7; x++, y++) {
        sq = x + 8*y;
        attacks |= sq_bit[sq];
        if (sq_bit[sq] & occupancy_both) { break; }
    }

    for (x=x0+1, y=y0-1; x<=7 && y>=0; x++, y--) {
        sq = x + 8*y;
        attacks |= sq_bit[sq];
        if (sq_bit[sq] & occupancy_both) { break; }
    }

    for (x=x0-1, y=y0+1; x>=0 && y<=7; x--, y++) {
        sq = x + 8*y;
        attacks |= sq_bit[sq];
        if (sq_bit[sq] & occupancy_both) { break; }
    }

    for (x=x0-1, y=y0-1; x>=0 && y>=0; x--, y--) {
        sq = x + 8*y;
        attacks |= sq_bit[sq];
        if (sq_bit[sq] & occupancy_both) { break; }
    }
    
    return attacks;
}

// indices: square, occupancy_index
static U64 bishop_relevant_occupancies[64][bishop_magic_indices];

// for converting occupancy_index into an occupancy
static U64 bishop_relevant_occupancy(int square, int occupancy_index) {
    U64 relevant_occupancy = 0;
    int sqs[9];
    int sqs_index = 0;
    int x0 = square % 8;
    int y0 = square / 8;
    int x, y;

    for (x=x0-1,y=y0-1; x>0 && y>0; x--,y--) { sqs[sqs_index++] = x+y*8; }
    for (x=x0-1,y=y0+1; x>0 && y<7; x--,y++) { sqs[sqs_index++] = x+y*8; }
    for (x=x0+1,y=y0-1; x<7 && y>0; x++,y--) { sqs[sqs_index++] = x+y*8; }
    for (x=x0+1,y=y0+1; x<7 && y<7; x++,y++) { sqs[sqs_index++] = x+y*8; }

    sqs_index = 0;
    while (occupancy_index) {
        if (occupancy_index & 1) {
            relevant_occupancy |= sq_bit[sqs[sqs_index]];
        }
        occupancy_index >>= 1;
        sqs_index++;
    }

    return relevant_occupancy;
}

static void init_bishop_relevant_occupancies() {
    for (int sq=0; sq<64; sq++) {
        int n_relevant_occupancies = bishop_n_relevant_occupancies[sq];
        for (int occupancy_index=0; occupancy_index < (1<<n_relevant_occupancies); occupancy_index++) {
            bishop_relevant_occupancies[sq][occupancy_index] = bishop_relevant_occupancy(sq, occupancy_index);
        } 
    }
}

static void bishop_visual_test(int sq) {
    init_bishop_relevant_occupancies();
    for (int i=0; i<(1<<bishop_n_relevant_occupancies[sq]); i++) {
        print_bitboard(bishop_relevant_occupancies[sq][i], sq);
        usleep(100000);
    }
}

static void find_bishop_magic_numbers() {
    for (int sq=0; sq<64; sq++) {
        U64 mask = bishop_relevant_bits_masks[sq];
        int max_occupancy_index = 1<<bishop_n_relevant_occupancies[sq];
        // print_bitboard(mask, sq);
        for (U64 i=0; i<100000000ULL; i++) {
            U64 magic_number_candidate = random_U64_few_bits();
            if (bit_count((magic_number_candidate * mask) & 0xff00000000000000) < 6) continue;
            // printf("candidate: %llu\n", magic_number_candidate);

            int fail = 0;
            U64 attack_table[bishop_magic_indices] = {0};
            for (int occupancy_index=0; occupancy_index<max_occupancy_index; occupancy_index++) {
                U64 relevant_occupancy = bishop_relevant_occupancies[sq][occupancy_index];
                int magic_index = (relevant_occupancy * magic_number_candidate) >> (64-bishop_max_relevant_occupancies);
                if (attack_table[magic_index] == 0) {
                    attack_table[magic_index] = get_bishop_attacks_slow(sq, relevant_occupancy);
                }
                else if (attack_table[magic_index] != get_bishop_attacks_slow(sq, relevant_occupancy)) {
                    fail = 1;
                    break;
                }
            }
            if (!fail) {
                // printf("%.2d  0x%llx,\n", sq, magic_number_candidate);
                bishop_magic_numbers[sq] = magic_number_candidate;
                memcpy(bishop_attack_tables[sq], attack_table, bishop_magic_indices * sizeof(U64));
                break;
            }
        }
    }
}

static void init_bishop_attacks() {
    init_bishop_masks();
    init_bishop_relevant_occupancies();
    find_bishop_magic_numbers();
}

/*///////////////////////////////////////////////////////////////////////////////
                              Section: rook attacks
/*///////////////////////////////////////////////////////////////////////////////

static int rook_n_relevant_occupancies[64] = {
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12,
};
static constexpr int rook_max_relevant_occupancies = 12;
static constexpr int rook_magic_indices = 1 << rook_max_relevant_occupancies;

// index: square
static U64 rook_relevant_bits_masks[64];
// indices: square, magic_index
static U64 rook_attack_tables[64][rook_magic_indices];
// index: square
static U64 rook_magic_numbers[64];

U64 get_rook_attacks(int square, U64 occupancy_both) {
    U64 magic_index = ((occupancy_both & rook_relevant_bits_masks[square]) * rook_magic_numbers[square]) >> (64-rook_max_relevant_occupancies);
    return rook_attack_tables[square][magic_index];
}

static void init_rook_masks() {
    for (int sq=0; sq<64; sq++) {
        int x0 = sq % 8;
        int y0 = sq / 8;
        int x, y;
        for (x=x0+1, y=y0; x<7; x++) { rook_relevant_bits_masks[sq] |= sq_bit[x + y*8]; }
        for (x=x0-1, y=y0; x>0; x--) { rook_relevant_bits_masks[sq] |= sq_bit[x + y*8]; }
        for (x=x0, y=y0+1; y<7; y++) { rook_relevant_bits_masks[sq] |= sq_bit[x + y*8]; }
        for (x=x0, y=y0-1; y>0; y--) { rook_relevant_bits_masks[sq] |= sq_bit[x + y*8]; }
    }
}

static U64 get_rook_attacks_slow(int square, U64 occupancy_both) {
    U64 attacks = 0ULL;
    int x0 = square % 8;
    int y0 = square / 8;
    int x, y, sq;

    for (x=x0+1, y=y0; x<=7; x++) {
        sq = x + 8*y;
        attacks |= sq_bit[sq];
        if (sq_bit[sq] & occupancy_both) { break; }
    }

    for (x=x0-1, y=y0; x>=0; x--) {
        sq = x + 8*y;
        attacks |= sq_bit[sq];
        if (sq_bit[sq] & occupancy_both) { break; }
    }

    for (x=x0, y=y0+1; y<=7; y++) {
        sq = x + 8*y;
        attacks |= sq_bit[sq];
        if (sq_bit[sq] & occupancy_both) { break; }
    }

    for (x=x0, y=y0-1; y>=0; y--) {
        sq = x + 8*y;
        attacks |= sq_bit[sq];
        if (sq_bit[sq] & occupancy_both) { break; }
    }

    return attacks;
}

// indices: square, occupancy_index
static U64 rook_relevant_occupancies[64][rook_magic_indices];

// for converting occupancy_index into an occupancy
static U64 rook_relevant_occupancy(int square, int occupancy_index) {
    U64 relevant_occupancy = 0;
    int sqs[12];
    int sqs_index = 0;
    int x0 = square % 8;
    int y0 = square / 8;
    int x, y;

    for (x=x0-1,y=y0; x>0; x--) { sqs[sqs_index++] = x+y*8; }
    for (x=x0+1,y=y0; x<7; x++) { sqs[sqs_index++] = x+y*8; }
    for (x=x0,y=y0-1; y>0; y--) { sqs[sqs_index++] = x+y*8; }
    for (x=x0,y=y0+1; y<7; y++) { sqs[sqs_index++] = x+y*8; }

    sqs_index = 0;
    while (occupancy_index) {
        if (occupancy_index & 1) {
            relevant_occupancy |= sq_bit[sqs[sqs_index]];
        }
        occupancy_index >>= 1;
        sqs_index++;
    }

    return relevant_occupancy;
}

static void init_rook_relevant_occupancies() {
    for (int sq=0; sq<64; sq++) {
        int n_relevant_occupancies = rook_n_relevant_occupancies[sq];
        for (int occupancy_index=0; occupancy_index < (1<<n_relevant_occupancies); occupancy_index++) {
            rook_relevant_occupancies[sq][occupancy_index] = rook_relevant_occupancy(sq, occupancy_index);
        } 
    }
}

static void rook_visual_test(int sq) {
    init_rook_relevant_occupancies();
    for (int i=0; i<(1<<rook_n_relevant_occupancies[sq]); i++) {
        print_bitboard(rook_relevant_occupancies[sq][i], sq);
        usleep(10000);
    }
}

static void find_rook_magic_numbers() {
    for (int sq=0; sq<64; sq++) {
        U64 mask = rook_relevant_bits_masks[sq];
        int max_occupancy_index = 1<<rook_n_relevant_occupancies[sq];
        // print_bitboard(mask, sq);
        for (U64 i=0; i<100000000ULL; i++) {
            U64 magic_number_candidate = random_U64_few_bits();
            if (bit_count((magic_number_candidate * mask) & 0xff00000000000000) < 6) continue;
            // printf("candidate: %llu\n", magic_number_candidate);

            int fail = 0;
            U64 attack_table[rook_magic_indices] = {0};
            for (int occupancy_index=0; occupancy_index<max_occupancy_index; occupancy_index++) {
                U64 relevant_occupancy = rook_relevant_occupancies[sq][occupancy_index];
                int magic_index = (relevant_occupancy * magic_number_candidate) >> (64-rook_max_relevant_occupancies);
                if (attack_table[magic_index] == 0) {
                    attack_table[magic_index] = get_rook_attacks_slow(sq, relevant_occupancy);
                }
                else if (attack_table[magic_index] != get_rook_attacks_slow(sq, relevant_occupancy)) {
                    fail = 1;
                    break;
                }
            }
            if (!fail) {
                // printf("%.2d  0x%llx,\n", sq, magic_number_candidate);
                rook_magic_numbers[sq] = magic_number_candidate;
                memcpy(rook_attack_tables[sq], attack_table, rook_magic_indices * sizeof(U64));
                break;
            }
        }
    }
}

static void init_rook_attacks() {
    init_rook_masks();
    init_rook_relevant_occupancies();
    find_rook_magic_numbers();
}

/*/////////////////////////////////////////////////////////////////////////////
                              Section: queen attacks
/*/////////////////////////////////////////////////////////////////////////////

U64 get_queen_attacks(int square, U64 occupancy_both) {
    return get_bishop_attacks(square, occupancy_both) | get_rook_attacks(square, occupancy_both);
}

/*/////////////////////////////////////////////////////////////////////////////
                              Section: init and main
/*/////////////////////////////////////////////////////////////////////////////

void init_attacks() {
    init_pawn_attacks();
    init_knight_attacks();
    init_king_attacks();
    init_bishop_attacks();
    init_rook_attacks();
}

#ifndef MAIN
// for testing
int main() {
    init_attacks();

    int sq = e6;
    U64 occupancy_both = B7 | C6 | D5 | E4 | F3 | G2 | H1 | B8 | A1;
    print_bitboard(occupancy_both, sq);
    U64 attacks = get_rook_attacks(sq, occupancy_both);
    print_bitboard(attacks, sq);
    U64 attacks3 = get_bishop_attacks(sq, occupancy_both);
    print_bitboard(attacks3, sq);
    // U64 attacks4 = get_queen_attacks(sq, occupancy);
    // print_bitboard(attacks4, sq);
    // U64 attacks2 = get_rook_attacks_slow(sq, occupancy);
    // print_bitboard(attacks2, sq);
    
    return 0;
}
#endif
