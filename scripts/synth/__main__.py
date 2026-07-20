import asyncio
import sys

from scripts.synth.cli import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
