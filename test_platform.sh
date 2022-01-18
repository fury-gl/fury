error_code=0
for file in `find . -name 'test_*.py' -print`;
do
    if coverage run -m -p pytest -svv $file; then
    error_code=1
    fi
done
coverage combine .
coverage report -m

exit $error_code

