# helper functions
from utils.arguments import parse_args
from utils.init_models import load_models
from inference import generate_transition

"""
** Visit utils/arguments.py to view & customize input parameters **
"""

if __name__ == "__main__":
    # Parse command-line arguments - get all the variables needed for script
    args = parse_args()

    # show progress
    progress = True

    # load base models
    FCVG_model, GlueStick_model, SEARAFT_model = load_models(args, progress=progress)

    # generate the transition
    generate_transition(args, GlueStick_model, FCVG_model, SEARAFT_model, 
                        progress=progress, visualize=True)