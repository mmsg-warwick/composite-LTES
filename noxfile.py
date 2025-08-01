import os
from pathlib import Path

import nox

# Options to modify nox behaviour
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True

VENV_DIR = Path("./venv").resolve()


@nox.session(name="docs")
def build_docs(session: nox.Session) -> None:
    """Build the documentation and load it in a browser tab, rebuilding on changes."""
    session.install("setuptools")
    session.install("-e", ".[docs]")
    with session.chdir("docs/"):
        # For local development
        if session.interactive:
            session.run(
                "sphinx-autobuild",
                "-j",
                "auto",
                "--open-browser",
                "-qT",
                ".",
                "build/html/",
            )
        # For CI testing if documentation builds
        else:
            session.run(
                "sphinx-build",
                "-b",
                "html",
                "-W",
                "--keep-going",
                ".",
                "build/html/",
            )


def install_and_run_tests(session, test_dir):
    """Install dependencies and run tests in the specified directory."""
    session.install("setuptools", silent=False)
    session.install("-e", ".[dev]", silent=False)
    session.run("pytest", test_dir)


@nox.session(name="generated-project-tests")
def run_generated_project_tests(session):
    """Run the tests for testing units inside generated project"""
    install_and_run_tests(session, "tests/generated_project_tests")


@nox.session(name="user-tests")
def run_user_tests(session):
    """Run user written tests"""
    install_and_run_tests(session, "tests/user_tests")


@nox.session(name="coverage")
def run_coverage(session):
    """Run the coverage tests and generate an XML report."""
    session.install("setuptools", silent=False)
    session.install("coverage", silent=False)
    session.install("-e", ".[dev]", silent=False)
    session.run(
        "pytest", "--cov=src/encapsulated_ltes", "--cov-report=xml", "tests/user_tests"
    )
    # session.posargs.append("--cov=src/electrode_coating")
    # session.posargs.append("--cov-report=xml")
    # run_user_tests(session)


@nox.session(name="dev")
def set_dev(session):
    """Install pybamm-cookie in editable mode"""
    session.install("virtualenv")
    session.run("virtualenv", os.fsdecode(VENV_DIR), silent=True)
    python = os.fsdecode(VENV_DIR.joinpath("bin/python"))
    session.run(python, "-m", "pip", "install", "-e", ".[dev]")
