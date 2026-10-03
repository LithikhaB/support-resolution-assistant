"""Run a bounded customer conversation from CLI arguments or a JSON request file."""

import argparse
from pathlib import Path

from fastapi import HTTPException
from pydantic import ValidationError

from app.api.resolution import run_request
from app.resolution.conversation import ConversationRequest, CustomerTurn, resolve_conversation
from app.resolution.service import get_resolution_service


def main():
    """Print per-issue drafts with optional replayable JSON and explicit reply targeting."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query")
    source.add_argument("--request", type=Path)
    parser.add_argument("--reply", action="append", default=[])
    parser.add_argument("--issue-id", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.request and args.reply:
        parser.error("put replies inside the request file when using --request")
    try:
        request = (
            ConversationRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
            if args.request
            else ConversationRequest(
                query=args.query,
                turns=[CustomerTurn(issue_id=args.issue_id, message=m) for m in args.reply],
            )
        )
        result = run_request(lambda: resolve_conversation(request, get_resolution_service()))
    except (ValidationError, OSError) as exc:
        parser.exit(2, f"Invalid conversation input: {exc}\n")
    except HTTPException as exc:
        parser.exit(1, f"{exc.detail}\n")
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        for issue in result.issues:
            print(f"Issue {issue.issue_id}: {issue.complaint}\n{issue.resolution.draft}\n")


if __name__ == "__main__":
    main()
