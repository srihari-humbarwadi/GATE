from typing import Any, Dict, Tuple, Union

# def instantiate_class(args, kwargs) -> Any:
#     """Instantiates a class with the given args and init.
#
#     Args:
#         args: Positional arguments required for instantiation.
#         init: Dict of the form {"class_path":...,"init_args":...}.
#
#     Returns:
#         The instantiated class object.
#     """
#     kwargs = init.get("init_args", {})
#     if not isinstance(args, tuple):
#         args = (args,)
#     class_module, class_name = init["class_path"].rsplit(".", 1)
#     module = __import__(class_module, fromlist=[class_name])
#     args_class = getattr(module, class_name)
#     return args_class(*args, **kwargs)
