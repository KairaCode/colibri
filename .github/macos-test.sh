set -eu
cd c
make tests/test_compat_env && ./tests/test_compat_env
python3 -c "import platform; print('python', platform.python_version(), platform.machine())"
