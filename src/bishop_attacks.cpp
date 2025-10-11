/* references:
 * https://www.chessprogramming.org/Looking_for_Magics
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "engine.h"
#include "random_numbers.cpp"
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>


// index: square
U64 bishop_relevant_bits_masks[64];
// indices: square, magic_index
U64 bishop_attack_tables[64][512];
// index: square
U64 bishop_magic_numbers[64];

static void init_bishop_masks() {
    for (int sq=0; sq<64; sq++) {
        int x0 = sq % 8;
        int y0 = sq / 8;
        int x, y;
        for (x=x0+1, y=y0+1; x<7 && y<7; x++, y++) { bishop_relevant_bits_masks[sq] |= 1ULL << (x + y*8); }
        for (x=x0+1, y=y0-1; x<7 && y>0; x++, y--) { bishop_relevant_bits_masks[sq] |= 1ULL << (x + y*8); }
        for (x=x0-1, y=y0+1; x>0 && y<7; x--, y++) { bishop_relevant_bits_masks[sq] |= 1ULL << (x + y*8); }
        for (x=x0-1, y=y0-1; x>0 && y>0; x--, y--) { bishop_relevant_bits_masks[sq] |= 1ULL << (x + y*8); }
    }
}

U64 get_bishop_attacks(int square, U64 occupancy) {
    U64 magic_index = ((occupancy & bishop_relevant_bits_masks[square]) * bishop_magic_numbers[square]) >> 55;
    return bishop_attack_tables[square][magic_index];
}

/*/////////////////////////////////////////////////////////////////////////////
                                   magic number generation
/*/////////////////////////////////////////////////////////////////////////////

static U64 get_bishop_attacks_slow(int square, U64 occupancy) {
    U64 attacks = 0ULL;
    int x0 = square % 8;
    int y0 = square / 8;
    int x, y, sq;

    for (x=x0+1, y=y0+1; x<=7 && y<=7; x++, y++) {
        sq = x + 8*y;
        attacks |= 1ULL << sq;
        if (sq_bit[sq] & occupancy) { break; }
    }

    for (x=x0+1, y=y0-1; x<=7 && y>=0; x++, y--) {
        sq = x + 8*y;
        attacks |= 1ULL << sq;
        if (sq_bit[sq] & occupancy) { break; }
    }

    for (x=x0-1, y=y0+1; x>=0 && y<=7; x--, y++) {
        sq = x + 8*y;
        attacks |= 1ULL << sq;
        if (sq_bit[sq] & occupancy) { break; }
    }

    for (x=x0-1, y=y0-1; x>=0 && y>=0; x--, y--) {
        sq = x + 8*y;
        attacks |= 1ULL << sq;
        if (sq_bit[sq] & occupancy) { break; }
    }
    
    return attacks;
}


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

// indices: square, occupancy_index
static U64 bishop_relevant_occupancies[64][512];

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
            relevant_occupancy |= (1ULL << sqs[sqs_index]);
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
            U64 attack_table[512] = {0};
            for (int occupancy_index=0; occupancy_index<max_occupancy_index; occupancy_index++) {
                U64 relevant_occupancy = bishop_relevant_occupancies[sq][occupancy_index];
                int magic_index = (relevant_occupancy * magic_number_candidate) >> 55;
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
                memcpy(bishop_attack_tables[sq], attack_table, 512 * sizeof(U64));
                break;
            }
        }
    }
}

/*/////////////////////////////////////////////////////////////////////////////
                                   main
/*/////////////////////////////////////////////////////////////////////////////

void init_bishop_attacks() {
    init_bishop_masks();
    init_bishop_relevant_occupancies();
    find_bishop_magic_numbers();
}

// for testing
int main() {
    init_bishop_attacks();

    int sq = a8;
    U64 occupancy = 0*B7 | 0*C6 | 0*D5 | 0*E4 | 0*F3 | 0*G2 | H1 |  B8 | A3;
    U64 attacks = get_bishop_attacks(sq, occupancy);
    U64 attacks2 = get_bishop_attacks_slow(sq, occupancy);
    print_bitboard(attacks, sq);
    print_bitboard(attacks2, sq);
    
    return 0;
}
