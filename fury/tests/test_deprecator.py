
import numpy.testing as npt
import fury
from fury.deprecator import (cmp_pkg_version, _add_dep_doc, _ensure_cr,
                             deprecate_with_version)


def test_cmp_pkg_version():
    # Test version comparator
    npt.assert_equal(cmp_pkg_version(fury.__version__), 0)
    npt.assert_equal(cmp_pkg_version('0.0'), -1)
    npt.assert_equal(cmp_pkg_version('1000.1000.1'), 1)
    npt.assert_equal(cmp_pkg_version(fury.__version__, fury.__version__), 0)
    for test_ver, pkg_ver, exp_out in (('1.0', '1.0', 0),
                                       ('1.0.0', '1.0', 0),
                                       ('1.0', '1.0.0', 0),
                                       ('1.1', '1.1', 0),
                                       ('1.2', '1.1', 1),
                                       ('1.1', '1.2', -1),
                                       ('1.1.1', '1.1.1', 0),
                                       ('1.1.2', '1.1.1', 1),
                                       ('1.1.1', '1.1.2', -1),
                                       ('1.1', '1.1dev', 1),
                                       ('1.1dev', '1.1', -1),
                                       ('1.2.1', '1.2.1rc1', 1),
                                       ('1.2.1rc1', '1.2.1', -1),
                                       ('1.2.1rc1', '1.2.1rc', 1),
                                       ('1.2.1rc', '1.2.1rc1', -1),
                                       ('1.2.1rc1', '1.2.1rc', 1),
                                       ('1.2.1rc', '1.2.1rc1', -1),
                                       ('1.2.1b', '1.2.1a', 1),
                                       ('1.2.1a', '1.2.1b', -1),
                                       ):
        npt.assert_equal(cmp_pkg_version(test_ver, pkg_ver), exp_out)

    npt.assert_raises(ValueError, cmp_pkg_version, 'foo.2')
    npt.assert_raises(ValueError, cmp_pkg_version, 'foo.2', '1.0')
    npt.assert_raises(ValueError, cmp_pkg_version, '1.0', 'foo.2')
    npt.assert_raises(ValueError, cmp_pkg_version, 'foo')


def test__ensure_cr():
    # Make sure text ends with carriage return
    npt.assert_equal(_ensure_cr('  foo'), '  foo\n')
    npt.assert_equal(_ensure_cr('  foo\n'), '  foo\n')
    npt.assert_equal(_ensure_cr('  foo  '), '  foo\n')
    npt.assert_equal(_ensure_cr('foo  '), 'foo\n')
    npt.assert_equal(_ensure_cr('foo  \n bar'), 'foo  \n bar\n')
    npt.assert_equal(_ensure_cr('foo  \n\n'), 'foo\n')


def test__add_dep_doc():
    # Test utility function to add deprecation message to docstring
    npt.assert_equal(_add_dep_doc('', 'foo'), 'foo\n')
    npt.assert_equal(_add_dep_doc('bar', 'foo'), 'bar\n\nfoo\n')
    npt.assert_equal(_add_dep_doc('   bar', 'foo'), '   bar\n\nfoo\n')
    npt.assert_equal(_add_dep_doc('   bar', 'foo\n'), '   bar\n\nfoo\n')
    npt.assert_equal(_add_dep_doc('bar\n\n', 'foo'), 'bar\n\nfoo\n')
    npt.assert_equal(_add_dep_doc('bar\n    \n', 'foo'), 'bar\n\nfoo\n')
    npt.assert_equal(_add_dep_doc(' bar\n\nSome explanation', 'foo\nbaz'),
                     ' bar\n\nfoo\nbaz\n\nSome explanation\n')
    npt.assert_equal(_add_dep_doc(' bar\n\n  Some explanation', 'foo\nbaz'),
                     ' bar\n  \n  foo\n  baz\n  \n  Some explanation\n')
