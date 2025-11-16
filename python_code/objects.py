import pickle
import os

"""#############################################################
                Section: Objects and savefiles
#############################################################"""

# for converting U32 move (game engine moves list) to 73x8x8 move (policy head output)
# reverse should happen with a temporary dict created when masking the policy head outputs
U32_move_to_policy_output_savefile = "U32_move_to_policy_output.pickle"
# indexed by (73, 8, 8) tuple
U32_move_to_policy_output_dict = dict()

"""#############################################################
                    Section: Functions
#############################################################"""

def set_directory():
    root_dir = __file__
    while root_dir[-17:] != "chess_game_engine":
        root_dir = root_dir[:-1]
        if len(root_dir) == 0:
            raise Exception("could not find project root directory")
    os.chdir(root_dir + r"/python_code/saved_objects")

def load_objects():
    set_directory()
    print()
    print("    Loading objects...")

    if os.path.exists(U32_move_to_policy_output_savefile):
        with open(U32_move_to_policy_output_savefile, 'rb') as f:
            U32_move_to_policy_output_dict = pickle.load(f)
            print(f"    {U32_move_to_policy_output_savefile} loaded")
    else:
        print(f"X   {U32_move_to_policy_output_savefile} not found")

def save_objects():
    set_directory()
    print("    Saving objects...")

    with open(U32_move_to_policy_output_savefile, 'wb') as f:
        pickle.dump(U32_move_to_policy_output_dict, f)
        print(f"    {U32_move_to_policy_output_savefile} saved")

    print()

if __name__ == "__main__":
    load_objects()
    save_objects()
