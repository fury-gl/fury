  - pip install nose coverage coveralls codecov
  - mkdir for_testing
  - cd for_testing
  - echo backend:Agg > matplotlibrc
  - if exist ../.coveragerc (cp ../.coveragerc .) else (echo no .coveragerc)
  - ps: |
        if ($env:COVERAGE)
        {
          $env:COVER_ARGS = "--with-coverage --cover-package dipy"
        }
  - cmd: echo %COVER_ARGS%
  - nosetests --with-doctest --verbose %COVER_ARGS% dipy