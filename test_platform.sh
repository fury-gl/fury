error_code=0
for file in `find . -name 'test_*.py' -print`;
do
    if pytest -svv $file; then
    error_code=1
    fi
done

exit $error_code

