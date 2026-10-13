"""Database migration utilities."""

import os
import subprocess
from typing import Optional


def run_alembic_upgrade(revision: str = "head") -> bool:
    """Run Alembic upgrade to the specified revision.

    Args:
        revision: Target revision (default: "head" for latest)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Set up environment
        env = os.environ.copy()
        backend_dir = os.path.dirname(os.path.dirname(__file__))
        env["PYTHONPATH"] = backend_dir + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            ["alembic", "upgrade", revision],
            cwd=backend_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"Alembic upgrade failed: {result.stderr}")
            return False

        print(f"Alembic upgrade successful: {result.stdout}")
        return True

    except subprocess.TimeoutExpired:
        print("Alembic upgrade timed out")
        return False
    except Exception as exc:
        print(f"Alembic upgrade error: {exc}")
        return False


def run_alembic_downgrade(revision: str = "-1") -> bool:
    """Run Alembic downgrade to the specified revision.

    Args:
        revision: Target revision (default: "-1" for previous)

    Returns:
        True if successful, False otherwise
    """
    try:
        env = os.environ.copy()
        backend_dir = os.path.dirname(os.path.dirname(__file__))
        env["PYTHONPATH"] = backend_dir + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            ["alembic", "downgrade", revision],
            cwd=backend_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"Alembic downgrade failed: {result.stderr}")
            return False

        print(f"Alembic downgrade successful: {result.stdout}")
        return True

    except subprocess.TimeoutExpired:
        print("Alembic downgrade timed out")
        return False
    except Exception as exc:
        print(f"Alembic downgrade error: {exc}")
        return False


def get_current_revision() -> Optional[str]:
    """Get the current Alembic revision."""
    try:
        env = os.environ.copy()
        backend_dir = os.path.dirname(os.path.dirname(__file__))
        env["PYTHONPATH"] = backend_dir + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            ["alembic", "current"],
            cwd=backend_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip()

    except Exception:
        return None