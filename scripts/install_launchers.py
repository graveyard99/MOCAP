"""Install relocatable Linux launchers inside the installation root only."""

from pathlib import Path
import os

repo = Path(__file__).resolve().parents[1]
root = Path(os.environ.get("OPENMOCAP_INSTALL_ROOT", str(repo.parent))).resolve()
launcher = root / "OpenMocap.sh"
launcher.write_text(
    '#!/usr/bin/env bash\nset -euo pipefail\nLAUNCHER_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nexec "$LAUNCHER_DIRECTORY/openmocap-vfx/scripts/launch.sh" "$@"\n'
)
launcher.chmod(0o755)
(root / "OpenMocap.desktop").write_text(
    "[Desktop Entry]\nType=Application\nName=OpenMocap VFX\n"
    "Comment=Geometry-first multi-camera motion capture\n"
    f'Exec="{launcher}"\nPath={repo}\nTerminal=false\nCategories=Graphics;3DGraphics;\n'
)
(root / "OpenMocap.desktop").chmod(0o755)
print(f"GUI launcher: {launcher}")
