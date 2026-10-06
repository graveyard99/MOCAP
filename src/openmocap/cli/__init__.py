"""Automation interface using the same application service as the desktop."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from openmocap.application import ProjectService, STAGES, installation_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openmocap", description="Geometry-first metric motion capture"
    )
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--json", action="store_true", help="JSON output (the default)")
    gui = commands.add_parser("gui")
    gui.add_argument("project", nargs="?", type=Path)
    demo = commands.add_parser("demo")
    demo.add_argument("--output", type=Path, default=Path.cwd() / "outputs" / "demo")
    demo.add_argument("--cameras", type=int, default=8)
    demo.add_argument("--frames", type=int, default=36)
    demo.add_argument("--moving", action="store_true")
    demo.add_argument("--no-fbx", action="store_true")
    project = commands.add_parser("project")
    project_commands = project.add_subparsers(dest="action", required=True)
    init = project_commands.add_parser("init")
    init.add_argument("path", type=Path)
    init.add_argument("--name", default="Capture")
    init.add_argument("--cameras", type=int, default=2)
    camera = commands.add_parser("import-camera")
    camera.add_argument("project", type=Path)
    camera.add_argument("camera_id")
    camera.add_argument("source", type=Path)
    calibration = commands.add_parser("import-calibration")
    calibration.add_argument("project", type=Path)
    calibration.add_argument("calibration", type=Path)
    for name in ["run", "qc"] + [s for s in STAGES if s not in {"qc", "export"}]:
        command = commands.add_parser(name)
        command.add_argument("project", type=Path)
    export = commands.add_parser("export")
    export.add_argument("project", type=Path)
    export.add_argument("--format", choices=["npz", "fbx", "bvh", "usd"], default="fbx")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--fps", type=float)
    export.add_argument("--start", type=float)
    export.add_argument("--end", type=float)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    service = ProjectService()

    def progress(event: dict) -> None:
        if event.get("status"):
            print(
                f"{event['stage']}: {event['status']} · {event.get('message', '')}", file=sys.stderr
            )

    try:
        if args.command == "gui":
            from openmocap.ui.app import launch

            return launch(str(args.project) if args.project else None)
        if args.command == "doctor":
            result = service.doctor()
            print(json.dumps(result, indent=2))
            return 0 if result["healthy"] else 1
        if args.command == "demo":
            project = service.create_example(
                args.output, camera_count=args.cameras, frames=args.frames, moving=args.moving
            )
            result = service.run(project, callback=progress)
            result["exports"] = [
                service.export(project, "npz", args.output / "exports" / "actor.npz"),
                service.export(project, "bvh", args.output / "exports" / "actor.bvh"),
            ]
            if not args.no_fbx:
                result["exports"].append(
                    service.export(project, "fbx", args.output / "exports" / "actor.fbx")
                )
        elif args.command == "project":
            result = service.create_project(args.path, args.name, camera_count=args.cameras)
        else:
            project = service.open_project(args.project)
            if args.command == "import-camera":
                service.import_camera(project, args.camera_id, args.source)
                result = {"project": project["project_file"], "camera": args.camera_id}
            elif args.command == "import-calibration":
                service.import_calibration(project, args.calibration)
                result = {"project": project["project_file"], "calibration": "imported"}
            elif args.command == "export":
                result = service.export(
                    project,
                    args.format,
                    args.output,
                    fps=args.fps or project["output_fps"],
                    start=args.start,
                    end=args.end,
                )
            else:
                result = service.run(
                    project,
                    callback=progress,
                    stage=None if args.command == "run" else args.command,
                )
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
        PermissionError,
        InterruptedError,
    ) as error:
        logging.error("%s", error)
        if args.verbose:
            logging.exception("Processing failed")
        return 2


__all__ = ["main", "installation_root"]
