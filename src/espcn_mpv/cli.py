"""Entry point: dispatch to the train / export / gen-glsl subcommands."""

import sys

USAGE = """usage: espcn-mpv <command> [options]

commands:
  train      train an ESPCN model on a folder of images
  export     export a trained checkpoint to an .npz weight file
  gen-glsl   generate an mpv custom shader from an .npz weight file

Run `espcn-mpv <command> --help` for the options of a command.
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE, end="")
        sys.exit(0)

    # Pop the subcommand so the target module's argparse never sees it.
    command = sys.argv.pop(1)

    if command == "train":
        from . import train

        train.main()
    elif command == "export":
        from . import export_weights

        export_weights.main()
    elif command == "gen-glsl":
        from . import gen_glsl

        gen_glsl.main()
    else:
        print(f"unknown command: {command}\n", file=sys.stderr)
        print(USAGE, end="", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
