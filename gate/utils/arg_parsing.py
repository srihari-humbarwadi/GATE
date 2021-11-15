import argparse
import logging
import pprint
import sys

from rich import print
from rich.logging import RichHandler

from gate.utils.storage import load_dict_from_json

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logging = logging.getLogger("rich")


# 3. priority should be defaults, json file, then any additional command line arguments
def merge_json_with_mutable_arguments(json_file_path, arg_dict):
    config_dict = load_dict_from_json(json_file_path)
    arguments_passed_to_command_line = get_arguments_passed_on_command_line(
        arg_dict=arg_dict
    )
    print(
        "arguments_passed_to_command_line",
        arguments_passed_to_command_line,
        sys.argv,
    )
    for key in config_dict.keys():
        if key in arguments_passed_to_command_line:
            config_dict[key] = arg_dict[key]

    return config_dict


class DictWithDotNotation(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return "<DictWithDotNotation " + dict.__repr__(self) + ">"


def get_arguments_passed_on_command_line(arg_dict):
    return [
        option.lower()
        for command_line_argument in sys.argv[1:]
        for option in arg_dict.keys()
        if command_line_argument.lower().replace("--", "") == option.lower()
    ]


def process_args(parser):
    args = parser.parse_args()

    if args.filepath_to_arguments_json_config is not None:
        args_dict = merge_json_with_mutable_arguments(
            json_file_path=args.filepath_to_arguments_json_config,
            arg_dict=vars(args),
        )
        args = DictWithDotNotation(args_dict)

    if isinstance(args, argparse.Namespace):
        args = vars(args)

    args_tree_like_structure = {}

    for key, value in args.items():
        if "." in key:
            top_level_key = key.split(".")[0]
            lower_level_key = key.replace(key.split(".")[0] + ".", "")

            if top_level_key in args_tree_like_structure:
                args_tree_like_structure[top_level_key][lower_level_key] = value
            else:
                args_tree_like_structure[top_level_key] = DictWithDotNotation(
                    {lower_level_key: value}
                )

        else:
            args_tree_like_structure[key] = value

    for key, value in args_tree_like_structure.items():
        if isinstance(value, dict):
            args_tree_like_structure[key] = DictWithDotNotation(value)

    args = DictWithDotNotation(args_tree_like_structure)
    arg_summary_string = pprint.pformat(args, indent=4)
    logging.info(arg_summary_string)

    return args