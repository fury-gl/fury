#!/usr/bin/env python
"""Simple tools to query github.com and gather repo information.

Taken from ipython

"""
# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

from __future__ import print_function
from future.standard_library import hooks
from distutils.version import LooseVersion

import json
import re
import sys
# import argparse
import operator

from datetime import datetime, timedelta
from subprocess import check_output

with hooks():
    from urllib.request import urlopen

# ----------------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------------

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
PER_PAGE = 100

element_pat = re.compile(r'<(.+?)>')
rel_pat = re.compile(r'rel=[\'"](\w+)[\'"]')

LAST_RELEASE = datetime(2015, 3, 18)

core_team_info = [{'usernames': ['skoudoro', 'skab12'],
                   'fullname': 'Serge Koudoro',
                   'affiliation': 'Indiana University, IN, USA',
                   },
                  {'usernames': ['garyfallidis', ],
                   'fullname': 'Eleftherios Garyfallidis',
                   'affiliation': 'Indiana University, IN, USA',
                   },
                  ]

extra_release_information = {"v0.1.0": "",
                             "v0.1.1": "",
                             "v0.1.2": "",
                             "v0.1.3": "",
                             "v0.1.4": "",
                             }

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------


def parse_link_header(headers):
    link_s = headers.get('link', '')
    urls = element_pat.findall(link_s)
    rels = rel_pat.findall(link_s)
    d = {}
    for rel, url in zip(rels, urls):
        d[rel] = url
    return d


def get_paged_request(url):
    """Get a full list, handling APIv3's paging."""
    results = []
    while url:
        print("fetching %s" % url, file=sys.stderr)
        f = urlopen(url)
        results.extend(json.load(f))
        links = parse_link_header(f.headers)
        url = links.get('next')
    return results


def get_issues(project="fury-gl/fury", state="closed", pulls=False):
    """Get a list of the issues from the Github API."""
    which = 'pulls' if pulls else 'issues'
    url = "https://api.github.com/repos/%s/%s?state=%s&per_page=%i" % \
        (project, which, state, PER_PAGE)
    return get_paged_request(url)


def get_tags(project="fury-gl/fury"):
    """Get a list of the tags from the Github API."""
    url = "https://api.github.com/repos/{}/tags".format(project)
    return get_paged_request(url)


def fetch_basic_stats(project="fury-gl/fury"):
    """Fetch the basic stats.

    Returns
    -------
    basic_stats : dict
        A dictionary containing basic statistics. For example:
        {   'subscribers': 41,
            'forks': 142,
            'forks_url': 'https://github.com/fury-gl/fury/network'
            'watchers': 94,
            'open_issues': 154,
            'stars': 94,
            'stars_url': 'https://github.com/fury-gl/fury/stargazers'
        }

    """
    desired_keys = ["stargazers_count", "stargazers_url", "watchers_count",
                    "watchers_url", "forks_count", "forks_url", "open_issues",
                    "issues_url", "subscribers_count", "subscribers_url"]
    url = "https://api.github.com/repos/{}".format(project)
    f = urlopen(url)
    r_json = json.load(f)
    basic_stats = dict((k, r_json[k]) for k in desired_keys if k in r_json)
    return basic_stats


def fetch_contributor_stats(project="fury-gl/fury"):
    """Fetch stats of contributors.

    Returns
    -------
    contributor_stats : dict
        A dictionary containing contributor statistics. For example:
        {'total_contributors': 50,
         'total_commits': 6031,
         'contributors': [ {
            "user_name": "Garyfallidis"
            "avatar_url":"https://avatars.githubusercontent.com/u/134276?v=3",
            "html_url": "https://github.com/Garyfallidis",
            "total_commits": 1389,
            "total_additions": 116712,
            "total_deletions": 70340,
            "weekly_commits": [
                        {
                        "w": "1367712000",
                        "a": 6898,
                        "d": 77,
                        "c": 10
                        },
                    ]
            },
            ]
        }

    """
    url = "https://api.github.com/repos/{}/stats/contributors".format(project)
    f = urlopen(url)
    r_json = json.load(f)

    contributor_stats = {}
    contributor_stats["total_contributors"] = len(r_json)
    contributor_stats["contributors"] = []

    cumulative_commits = 0
    desired_keys = ["login", "avatar_url", "html_url"]
    for contributor in r_json:
        contributor_dict = dict((k, contributor["author"][k])
                                for k in desired_keys
                                if k in contributor["author"])

        # import ipdb;ipdb.set_trace()
        # check if "author" is null
        if not contributor_dict["login"]:
            continue

        # Replace key name
        contributor_dict["username"] = contributor_dict.pop("login")
        contributor_dict["nb_commits"] = contributor["total"]

        # Update total commits
        cumulative_commits += contributor_dict["nb_commits"]

        total_additions = 0
        total_deletions = 0
        for week in contributor["weeks"]:
            total_additions += week['a']
            total_deletions += week['d']

        contributor_dict["total_additions"] = total_additions
        contributor_dict["total_deletions"] = total_deletions
        # contributor_dict["weekly_commits"] = contributor["weeks"]
        contributor_stats["contributors"].insert(0, contributor_dict)

    contributor_stats["total_commits"] = cumulative_commits
    return contributor_stats


def _parse_datetime(s):
    """Parse dates in the format returned by the Github API."""
    if s:
        return datetime.strptime(s, ISO8601)
    else:
        return datetime.fromtimestamp(0)


def issues2dict(issues):
    """Convert a list of issues to a dict, keyed by issue number."""
    idict = {}
    for i in issues:
        idict[i['number']] = i
    return idict


def is_pull_request(issue):
    """Return True if the given issue is a pull request."""
    return 'pull_request_url' in issue


def issues_closed_since(period=LAST_RELEASE, project="fury-gl/fury",
                        pulls=False):
    """Get all issues closed since a particular point in time.

    period can either be a datetime object, or a timedelta object. In the
    latter case, it is used as a time before the present.

    """
    which = 'pulls' if pulls else 'issues'

    if isinstance(period, timedelta):
        period = datetime.now() - period
    url = "https://api.github.com/repos/%s/%s?state=closed&sort=updated&since=%s&per_page=%i" % (project, which, period.strftime(ISO8601), PER_PAGE)
    allclosed = get_paged_request(url)
    # allclosed = get_issues(project=project, state='closed', pulls=pulls,
    #                        since=period)
    filtered = [i for i in allclosed
                if _parse_datetime(i['closed_at']) > period]

    # exclude rejected PRs
    if pulls:
        filtered = [pr for pr in filtered if pr['merged_at']]

    return filtered


def sorted_by_field(issues, field='closed_at', reverse=False):
    """Return a list of issues sorted by closing date date."""
    return sorted(issues, key=lambda i: i[field], reverse=reverse)


def report(issues, show_urls=False):
    """Summary report about a list of issues, printing number and title."""
    # titles may have unicode in them, so we must encode everything below
    if show_urls:
        for i in issues:
            role = 'ghpull' if 'merged_at' in i else 'ghissue'
            print('* :%s:`%d`: %s' % (role, i['number'],
                                      i['title'].encode('utf-8')))
    else:
        for i in issues:
            print('* %d: %s' % (i['number'], i['title'].encode('utf-8')))


def get_all_versions(ignore='', project="fury-gl/fury"):
    """Return all releases version.

    Parameters
    ----------
    ignore: str
        skip a version number between (default: '')
        you can skip minor, or micro version number, it
        will be replace by x
    project: str
        repo path

    Returns
    -------
    l_version: list of str
        versions number list

    """
    tags = get_tags(project=project)
    l_version = [t['name']for t in tags]

    if ignore.lower() in ['micro', 'minor']:
        l_version = list(set([re.sub(r'(\d+)$', 'x', v) for v in l_version]))

    if ignore.lower() == 'minor':
        l_version = list(set([re.sub(r'\.(\d+)\.', '.x.', v)
                              for v in l_version]))

    return l_version


def version_compare(current_version, version_number, op='eq',
                    all_versions=None):
    """Compare doc version. This is afilter for sphinx template."""
    p = re.compile(r'(\d+)\.(\d+)')
    d_operator = {'<': operator.lt,
                  'lt': operator.lt,
                  '<=': operator.le,
                  'le': operator.le,
                  '>': operator.gt,
                  'gt': operator.gt,
                  '>=': operator.ge,
                  'ge': operator.ge,
                  '==': operator.eq,
                  '=': operator.eq,
                  'eq': operator.eq,
                  '!=': operator.ne,
                  }
    # Setup default value to op
    if op not in d_operator.keys():
        op = 'eq'

    # check dev page
    if current_version.lower() == 'dev':
        return 'post' in version_number

    # major and minor extraction
    current = p.search(current_version)
    ref = p.search(version_number)

    # Check if it is the latest release
    all_versions = all_versions or get_all_versions()
    if current_version.lower() == 'latest':
        last_version = sorted(all_versions)[0]
        last_version = p.search(last_version)
        if LooseVersion(last_version.group()) ==  \
           LooseVersion(ref.group()) and \
           'post' not in version_number:
            return True
        return False

    if 'post' in version_number:
        return False

    return d_operator[op](LooseVersion(current.group()),
                          LooseVersion(ref.group()))


def generate_release_information():
    """Generate release information."""
    pass


def github_stats():
    """Get release github stats."""
    # Whether to add reST urls for all issues in printout.
    show_urls = True

    # By default, search one month back
    tag = None
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except Exception:
            tag = sys.argv[1]
    else:
        tag = check_output(['git', 'describe', '--abbrev=0']).strip()

    if tag:
        cmd = ['git', 'log', '-1', '--format=%ai', tag]
        tagday, _ = check_output(cmd).strip().rsplit(' ', 1)
        since = datetime.strptime(tagday, "%Y-%m-%d %H:%M:%S")
    else:
        since = datetime.now() - timedelta(days=days)

    print("fetching GitHub stats since %s (tag: %s)" % (since, tag),
          file=sys.stderr)
    # turn off to play interactively without redownloading, use %run -i
    if 1:
        issues = issues_closed_since(since, pulls=False)
        pulls = issues_closed_since(since, pulls=True)

    # For regular reports, it's nice to show them in reverse chronological
    # order
    issues = sorted_by_field(issues, reverse=True)
    pulls = sorted_by_field(pulls, reverse=True)

    n_issues, n_pulls = map(len, (issues, pulls))
    n_total = n_issues + n_pulls

    # Print summary report we can directly include into release notes.
    print()
    since_day = since.strftime("%Y/%m/%d")
    today = datetime.today().strftime("%Y/%m/%d")
    print("GitHub stats for %s - %s (tag: %s)" % (since_day, today, tag))
    print()
    print("These lists are automatically generated, and may be incomplete or"
          "contain duplicates.")
    print()
    if tag:
        # print git info, in addition to GitHub info:
        since_tag = tag+'..'
        cmd = ['git', 'log', '--oneline', since_tag]
        ncommits = len(check_output(cmd).splitlines())

        author_cmd = ['git', 'log', '--format=* %aN', since_tag]
        all_authors = check_output(author_cmd).splitlines()
        unique_authors = sorted(set(all_authors))

        if len(unique_authors) == 0:
            print("No commits during this period.")
        else:
            print("The following %i authors contributed %i commits." %
                  (len(unique_authors), ncommits))
            print()
            print('\n'.join(unique_authors))
            print()

            print()
            print("We closed a total of %d issues, %d pull requests and %d"
                  " regular issues;\nthis is the full list (generated with"
                  " the script \n:file:`tools/github_stats.py`):" %
                  (n_total, n_pulls, n_issues))
            print()
            print('Pull Requests (%d):\n' % n_pulls)
            report(pulls, show_urls)
            print()
            print('Issues (%d):\n' % n_issues)
            report(issues, show_urls)


# ----------------------------------------------------------------------------
# Sphinx connection
# ----------------------------------------------------------------------------
def add_jinja_filters(app):
    app.builder.templates.environment.filters['version_compare'] = \
        version_compare


def setup(app):
    """
    - Create releases information
    - Collect and clean authors
    - Adds extra jinja filters.
    """
    app.connect("builder-inited", add_jinja_filters)
    app.add_stylesheet("css/custom_github.css")


# ----------------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--use_date")
    # github_stats()
    # import ipdb;ipdb.set_trace()
    # generate_release_information()
    # get_all_versions()
    # get_all_versions('minor')
    # get_all_versions('micro')
    # print(fetch_basic_stats())
    print(fetch_contributor_stats())
