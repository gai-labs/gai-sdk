import nox

@nox.session(python=["3.10"])
def tests(session):
    # Install all dependencies and the package itself
    session.install(".")
    session.install(".[dev]")
    session.install(".[cli]")
    session.install(".[ttt]")

    # Run tests
    #session.run("pytest", "--cov")