for file in `find . -name 'test_*.py' -print`;
do
    pytest -svv $file;
done
