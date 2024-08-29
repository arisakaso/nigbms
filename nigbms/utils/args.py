def arrange_sweep_args_for_hydra():
    import sys

    new_cmd_line_args = []
    for arg in sys.argv:
        if "=" in arg:  # catch x=y
            arg = "++" + arg  # Appending or overriding. see https://hydra.cc/docs/advanced/override_grammar/basic/
        # Try and catch the wandb agent formatted args
        if "={" in arg:
            arg = arg.replace("'", "")
        new_cmd_line_args.append(arg)
    sys.argv = new_cmd_line_args
