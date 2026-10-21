"""python -m ats_analyzer.web [--host 127.0.0.1] [--port 8000]"""

import argparse


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(prog="python -m ats_analyzer.web")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only behind a proxy you control")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run("ats_analyzer.web.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
