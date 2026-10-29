from __future__ import annotations

import sys

from research_team.__main__ import main as cli_main
from research_team.chat import chat_main


def main() -> int:
    if len(sys.argv) > 1:
        return cli_main(sys.argv[1:])
    return chat_main()


if __name__ == "__main__":
    raise SystemExit(main())
