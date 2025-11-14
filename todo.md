# overall
- 

# alphazero
- preallocate giant tensor for inputs?
    ((N+7)*14)x8x8 where N is max plausible number of chess turns - can allocate more if needed in rare cases
    is this actually faster?
- implement GameStateNode stuff needed for MCTS in BoardState
    functions
        new_node() should return a new BoardState
        model_input() as 21x8x8 = 14x8x8 + 7x8x8 (see feature_channels description below)
            NOTE: update the indexing values if time_history is reduced
            this will write to the preallocated giant tensor at [(half_turn+7)*14:that+21, :, :]
            the actual model input will be giant_tensor[half_turn*14:that+119, :, :]
            this way i don't have to keep making new tensors
        legal_moves() as 73x8x8 mask for policy output (valid_moves on connect 4 proj)
    attributes
        parent-child pointers
        visit_counts
        etc
- implement MCTS
- cache the conversion from U32 move to 73x8x8 index (d,y,x)
